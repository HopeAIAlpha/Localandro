package com.localandro.gemma4e2b.agent

import android.util.Log
import com.localandro.gemma4e2b.domain.model.Message
import com.localandro.gemma4e2b.domain.model.MessageRole
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import com.localandro.gemma4e2b.inference.InferenceEngineDisconnectedException
import com.localandro.gemma4e2b.memory.LongTermMemory
import com.localandro.gemma4e2b.memory.SlidingWindowContext
import com.localandro.gemma4e2b.security.SecurityPolicy
import com.localandro.gemma4e2b.tools.ToolRegistry
import com.localandro.gemma4e2b.tools.ToolResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.update

/**
 * Core agentic orchestration engine implementing a deterministic
 * Finite-State-Machine loop:
 *
 * ```
 * IDLE → PLANNING → EXECUTING → OBSERVING → REFINING → IDLE
 *                       ↑_________________________↓
 *                     (loop if model requests more tools)
 * ```
 *
 * The orchestrator coordinates:
 * - The on-device LLM ([InferenceRepository]) for planning & refinement.
 * - The [ToolRegistry] for resolving and executing tool calls.
 * - The [SecurityPolicy] for gating destructive actions.
 * - The [SlidingWindowContext] for token-budgeted context management.
 * - The [LongTermMemory] for persistent fact storage.
 *
 * @param inferenceRepository On-device LLM engine.
 * @param toolRegistry        Registry of available tools.
 * @param securityPolicy      Security gatekeeper.
 * @param longTermMemory      Persistent fact store.
 * @param slidingWindowContext Token-budgeted context builder.
 */
class ActionOrchestrator(
    private val inferenceRepository: InferenceRepository,
    private val toolRegistry: ToolRegistry,
    private val securityPolicy: SecurityPolicy,
    private val longTermMemory: LongTermMemory,
    private val slidingWindowContext: SlidingWindowContext = SlidingWindowContext()
) {

    companion object {
        private const val TAG = "ActionOrchestrator"

        /**
         * Maximum Plan→Execute→Observe cycles per user request.
         * Kept low (3) for the tiny E2B model to avoid context exhaustion.
         */
        const val MAX_ITERATIONS = 3

        /**
         * Maximum characters the model may generate in a single response.
         * Prevents runaway generation when the model enters a repetition loop.
         * ~500 tokens at ~4 chars/token.
         */
        private const val MAX_OUTPUT_CHARS = 2000

        /**
         * Maximum characters for the prompt sent to the native engine.
         * Leaves room for ~512 tokens of generation within the 4096 context.
         */
        private const val MAX_PROMPT_CHARS = 12000

        /**
         * Minimum number of characters before repetition detection kicks in.
         */
        private const val REPETITION_CHECK_THRESHOLD = 60
    }

    private val _agentState = MutableStateFlow(AgentState())
    val agentState: StateFlow<AgentState> = _agentState.asStateFlow()

    private val conversationHistory = mutableListOf<Message>()

    /**
     * Processes a user message through the agentic loop.
     *
     * @param userText     The user's input text.
     * @param onToken      Callback invoked for each streamed token (for UI).
     * @param onToolCall   Callback when a tool is about to execute.
     * @param onComplete   Callback with the final assembled response.
     * @param onError      Callback on unrecoverable error.
     */
    suspend fun processUserMessage(
        userText: String,
        onToken: (String) -> Unit = {},
        onToolCall: (String) -> Unit = {},
        onComplete: (Message) -> Unit = {},
        onError: (String) -> Unit = {}
    ) {
        // Add user message to history.
        val userMessage = Message(role = MessageRole.USER, content = userText)
        conversationHistory.add(userMessage)

        var iterations = 0

        try {
            while (iterations < MAX_ITERATIONS) {
                iterations++
                _agentState.update {
                    it.copy(
                        phase = AgentPhase.PLANNING,
                        iterations = iterations
                    )
                }

                // Build context with sliding window.
                val systemPrompt = buildSystemPrompt()
                val contextMessages = slidingWindowContext.buildContext(
                    systemPrompt = systemPrompt,
                    messages = conversationHistory,
                    memoryContext = longTermMemory.buildMemoryContext()
                )

                // Build the prompt from context messages.
                var prompt = buildPromptFromContext(contextMessages)

                // Safety: truncate prompt if it exceeds the native engine's
                // capacity to prevent SIGSEGV from context overflow.
                if (prompt.length > MAX_PROMPT_CHARS) {
                    Log.w(TAG, "Prompt too long (${prompt.length} chars), truncating to $MAX_PROMPT_CHARS")
                    prompt = prompt.takeLast(MAX_PROMPT_CHARS)
                }

                Log.d(TAG, "Iteration $iterations — sending prompt (${prompt.length} chars)")

                // Stream model response.
                val fullResponse = StringBuilder()
                var repetitionDetected = false
                var engineCrashed = false

                try {
                    inferenceRepository.streamResponse(prompt)
                        .flowOn(Dispatchers.IO)
                        .catch { e ->
                            if (e is InferenceEngineDisconnectedException) {
                                // The inference engine process died.  Signal
                                // recovery below instead of treating it as
                                // a fatal error.
                                engineCrashed = true
                                Log.w(TAG, "Inference engine disconnected", e)
                                return@catch
                            }
                            Log.e(TAG, "Inference error on iteration $iterations", e)
                            _agentState.update {
                                it.copy(
                                    phase = AgentPhase.ERROR,
                                    errorMessage = e.localizedMessage ?: "Inference error"
                                )
                            }
                            onError(e.localizedMessage ?: "Inference error")
                        }
                        .onCompletion { /* handled below */ }
                        .collect { token ->
                            fullResponse.append(token)

                            // Safety: stop collecting if output is too long or
                            // the model is stuck in a repetitive loop.
                            if (fullResponse.length > MAX_OUTPUT_CHARS) {
                                Log.w(TAG, "Output exceeded $MAX_OUTPUT_CHARS chars — truncating")
                                return@collect
                            }
                            if (isRepeating(fullResponse.toString())) {
                                Log.w(TAG, "Repetition detected — stopping generation")
                                repetitionDetected = true
                                return@collect
                            }

                            // Only emit user-visible text to the UI — never leak
                            // tool-call delimiters or JSON payloads.
                            val visibleText = ToolCallParser.getVisibleText(fullResponse.toString())
                            onToken(visibleText)
                        }
                } catch (e: InferenceEngineDisconnectedException) {
                    // streamResponse() itself may throw before returning a
                    // Flow (e.g. when serviceMessenger is null).
                    engineCrashed = true
                    Log.w(TAG, "Inference engine disconnected (direct throw)", e)
                }

                // ── Crash recovery ──────────────────────────────────
                // If the inference engine process crashed, transition to
                // PLANNING and retry.  RemoteInferenceRepository handles
                // the actual service reconnection transparently.
                if (engineCrashed) {
                    Log.w(TAG, "Recovering from engine crash — returning to PLANNING (iteration $iterations)")
                    _agentState.update { it.copy(phase = AgentPhase.PLANNING) }
                    delay(1500) // Brief pause to let the service process restart
                    continue
                }

                if (_agentState.value.phase == AgentPhase.ERROR) return

                // If repetition was detected, trim the response to just
                // the non-repeating prefix and treat it as a final answer.
                if (repetitionDetected) {
                    val trimmed = trimRepetition(fullResponse.toString())
                    val cleanResponse = ToolCallParser.stripToolTokens(trimmed).trim()
                    val modelMessage = Message(
                        role = MessageRole.MODEL,
                        content = cleanResponse.ifEmpty {
                            "I apologize, I wasn't able to process that request properly. Could you try rephrasing?"
                        }
                    )
                    conversationHistory.add(modelMessage)
                    _agentState.update { it.copy(phase = AgentPhase.IDLE) }
                    onComplete(modelMessage)
                    return
                }

                val responseText = fullResponse.toString()

                // Check for tool call in the response.
                val toolCallResult = ToolCallParser.extractToolCall(responseText)

                if (toolCallResult != null) {
                    val (parsedCall, remaining) = toolCallResult

                    Log.d(TAG, "Tool call detected: ${parsedCall.name}")

                    // Add the model's planning message (text before tool call).
                    val planText = ToolCallParser.getVisibleText(responseText).trim()
                    if (planText.isNotEmpty()) {
                        _agentState.update {
                            it.copy(planSummary = planText)
                        }
                    }

                    // Record the raw model output (including the tool-call
                    // token) so it can be replayed into context correctly.
                    conversationHistory.add(
                        Message(
                            role = MessageRole.TOOL_CALL,
                            content = "<|tool_call>call:${parsedCall.name}{${
                                parsedCall.arguments.entries.joinToString(",") { (k, v) ->
                                    "$k:<|\"|>$v<|\"|>"
                                }
                            }}<tool_call|>"
                        )
                    )

                    // ── EXECUTING phase ─────────────────────────────
                    _agentState.update {
                        it.copy(
                            phase = AgentPhase.EXECUTING,
                            activeTool = parsedCall.name
                        )
                    }
                    onToolCall(parsedCall.name)

                    val tool = toolRegistry.resolve(parsedCall.name)
                    val executionResult: ToolResult = if (tool == null) {
                        ToolResult.Failure("Unknown tool: ${parsedCall.name}")
                    } else {
                        // Security gate.
                        val denied = securityPolicy.validate(tool, parsedCall.arguments)
                        denied ?: tool.execute(parsedCall.arguments)
                    }

                    // ── OBSERVING phase ─────────────────────────────
                    _agentState.update {
                        it.copy(
                            phase = AgentPhase.OBSERVING,
                            activeTool = null
                        )
                    }

                    val resultText = when (executionResult) {
                        is ToolResult.Success -> executionResult.output
                        is ToolResult.Failure -> "ERROR: ${executionResult.reason}"
                        is ToolResult.Denied -> "DENIED: User declined execution of ${parsedCall.name}."
                    }

                    // Inject tool result into conversation using Gemma 4 format.
                    val toolResultFormatted = ToolCallParser.formatToolResult(
                        toolName = parsedCall.name,
                        result = resultText
                    )
                    conversationHistory.add(
                        Message(role = MessageRole.TOOL_RESULT, content = toolResultFormatted)
                    )

                    Log.d(TAG, "Tool result: ${resultText.take(200)}")

                    // ── REFINING phase → loop back ──────────────────
                    _agentState.update {
                        it.copy(phase = AgentPhase.REFINING)
                    }

                    // Continue the loop — the model will see the tool result
                    // in the next iteration and either call another tool or
                    // produce a final answer.
                    continue
                }

                // No tool call — this is the final response.
                val cleanResponse = ToolCallParser.stripToolTokens(responseText).trim()
                val modelMessage = Message(role = MessageRole.MODEL, content = cleanResponse)
                conversationHistory.add(modelMessage)

                _agentState.update {
                    it.copy(
                        phase = AgentPhase.IDLE,
                        planSummary = null,
                        activeTool = null
                    )
                }

                onComplete(modelMessage)
                return
            }

            // Exhausted iterations — return whatever we have.
            Log.w(TAG, "Max iterations ($MAX_ITERATIONS) reached")
            val fallbackMessage = Message(
                role = MessageRole.MODEL,
                content = "I've reached the iteration limit ($MAX_ITERATIONS). " +
                        "Here is my best response based on the information gathered."
            )
            conversationHistory.add(fallbackMessage)
            _agentState.update { it.copy(phase = AgentPhase.IDLE) }
            onComplete(fallbackMessage)

        } catch (e: Exception) {
            Log.e(TAG, "Unrecoverable error in agentic loop", e)
            _agentState.update {
                it.copy(
                    phase = AgentPhase.ERROR,
                    errorMessage = e.localizedMessage ?: "Unknown error"
                )
            }
            onError(e.localizedMessage ?: "Unknown error")
        }
    }

    /** Resets the agent to IDLE, clearing any error state. */
    fun reset() {
        _agentState.update {
            AgentState(phase = AgentPhase.IDLE)
        }
    }

    /** Clears conversation history (e.g. on new session). */
    fun clearHistory() {
        conversationHistory.clear()
    }

    // ── Internal helpers ────────────────────────────────────────────

    private fun buildSystemPrompt(): String = buildString {
        append(SecurityPolicy.SYSTEM_PREAMBLE)
        appendLine()
        appendLine()
        append(toolRegistry.buildCatalogue())
    }

    /**
     * Builds the formatted prompt from context messages using Gemma 4's
     * native turn tokens: `<|turn>role ... <turn|>`.
     *
     * Tool-call and tool-result messages are grouped into a single
     * `<|turn>model` turn (left open so the model continues generating
     * after seeing the tool result), matching the Gemma 4 wire format:
     *
     * ```
     * <|turn>model
     * <|tool_call>call:func{...}<tool_call|><|tool_response>response:func{...}<tool_response|>
     * ```
     */
    private fun buildPromptFromContext(messages: List<Message>): String = buildString {
        var inModelTurn = false

        messages.forEach { msg ->
            when (msg.role) {
                MessageRole.SYSTEM -> {
                    if (inModelTurn) { appendLine("<turn|>"); inModelTurn = false }
                    appendLine("<|turn>system")
                    appendLine(msg.content)
                    appendLine("<turn|>")
                }
                MessageRole.USER -> {
                    if (inModelTurn) { appendLine("<turn|>"); inModelTurn = false }
                    appendLine("<|turn>user")
                    appendLine(msg.content)
                    appendLine("<turn|>")
                }
                MessageRole.MODEL -> {
                    if (inModelTurn) { appendLine("<turn|>"); inModelTurn = false }
                    appendLine("<|turn>model")
                    appendLine(msg.content)
                    appendLine("<turn|>")
                }
                MessageRole.TOOL_CALL -> {
                    // Tool call starts or continues a model turn.
                    if (!inModelTurn) {
                        appendLine("<|turn>model")
                        inModelTurn = true
                    }
                    appendLine(msg.content)
                }
                MessageRole.TOOL_RESULT -> {
                    // Tool result continues the same model turn as the
                    // tool call, following the Gemma 4 convention.
                    if (!inModelTurn) {
                        appendLine("<|turn>model")
                        inModelTurn = true
                    }
                    appendLine(msg.content)
                }
            }
        }

        // Leave a model turn open for the engine to generate into.
        if (!inModelTurn) {
            appendLine("<|turn>model")
        }
        // Do NOT close with <turn|> — the model continues from here.
    }

    // ── Repetition detection ────────────────────────────────────────

    /**
     * Detects when the model is stuck in a repetitive loop by checking
     * if a short pattern repeats multiple times at the end of [text].
     */
    private fun isRepeating(text: String): Boolean {
        if (text.length < REPETITION_CHECK_THRESHOLD) return false
        val tail = text.takeLast(200)
        // Try pattern lengths from 2 to 40 characters
        for (patLen in 2..minOf(40, tail.length / 4)) {
            val pattern = tail.substring(tail.length - patLen)
            val minRepeats = 4
            val expected = pattern.repeat(minRepeats)
            if (tail.length >= expected.length && tail.endsWith(expected)) {
                return true
            }
        }
        return false
    }

    /**
     * Trims the repeated suffix from [text], keeping only the first
     * occurrence of the repeating pattern.
     */
    private fun trimRepetition(text: String): String {
        if (text.length < REPETITION_CHECK_THRESHOLD) return text
        val tail = text.takeLast(200)
        for (patLen in 2..minOf(40, tail.length / 4)) {
            val pattern = tail.substring(tail.length - patLen)
            val expected = pattern.repeat(4)
            if (tail.length >= expected.length && tail.endsWith(expected)) {
                // Find where the repetition starts and trim
                val firstOccurrence = text.indexOf(pattern)
                if (firstOccurrence >= 0) {
                    return text.substring(0, firstOccurrence + patLen)
                }
            }
        }
        return text
    }
}
