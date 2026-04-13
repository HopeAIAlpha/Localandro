package com.localandro.gemma4e2b.agent

import android.util.Log
import com.localandro.gemma4e2b.domain.model.Message
import com.localandro.gemma4e2b.domain.model.MessageRole
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import com.localandro.gemma4e2b.memory.LongTermMemory
import com.localandro.gemma4e2b.memory.SlidingWindowContext
import com.localandro.gemma4e2b.security.SecurityPolicy
import com.localandro.gemma4e2b.tools.ToolRegistry
import com.localandro.gemma4e2b.tools.ToolResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
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

        /** Maximum Plan→Execute→Observe cycles per user request. */
        const val MAX_ITERATIONS = 5
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
                val prompt = buildPromptFromContext(contextMessages)

                Log.d(TAG, "Iteration $iterations — sending prompt (${prompt.length} chars)")

                // Stream model response.
                val fullResponse = StringBuilder()
                var hasToolCall = false

                inferenceRepository.streamResponse(prompt)
                    .flowOn(Dispatchers.IO)
                    .catch { e ->
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
                        // Only emit user-visible text to the UI — never leak
                        // tool-call delimiters or JSON payloads.
                        val visibleText = ToolCallParser.getVisibleText(fullResponse.toString())
                        onToken(visibleText)
                    }

                if (_agentState.value.phase == AgentPhase.ERROR) return

                val responseText = fullResponse.toString()

                // Check for tool call in the response.
                val toolCallResult = ToolCallParser.extractToolCall(responseText)

                if (toolCallResult != null) {
                    hasToolCall = true
                    val (parsedCall, remaining) = toolCallResult

                    Log.d(TAG, "Tool call detected: ${parsedCall.name}")

                    // Add the model's planning message (text before tool call).
                    val planText = ToolCallParser.getVisibleText(responseText).trim()
                    if (planText.isNotEmpty()) {
                        _agentState.update {
                            it.copy(planSummary = planText)
                        }
                    }

                    // Record tool call in history.
                    conversationHistory.add(
                        Message(role = MessageRole.TOOL_CALL, content = parsedCall.raw)
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

                    // Inject tool result into conversation.
                    val toolResultFormatted = ToolCallParser.formatToolResult(resultText)
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
                content = "He alcanzado el límite de iteraciones ($MAX_ITERATIONS). " +
                        "Aquí está mi mejor respuesta basada en la información recopilada."
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

    private fun buildPromptFromContext(messages: List<Message>): String = buildString {
        messages.forEach { msg ->
            when (msg.role) {
                MessageRole.SYSTEM -> {
                    appendLine("<start_of_turn>system")
                    appendLine(msg.content)
                    appendLine("<end_of_turn>")
                }
                MessageRole.USER -> {
                    appendLine("<start_of_turn>user")
                    appendLine(msg.content)
                    appendLine("<end_of_turn>")
                }
                MessageRole.MODEL -> {
                    appendLine("<start_of_turn>model")
                    appendLine(msg.content)
                    appendLine("<end_of_turn>")
                }
                MessageRole.TOOL_CALL -> {
                    appendLine("<start_of_turn>model")
                    appendLine("<|tool_call|>")
                    appendLine(msg.content)
                    appendLine("<|end_tool_call|>")
                    appendLine("<end_of_turn>")
                }
                MessageRole.TOOL_RESULT -> {
                    appendLine("<start_of_turn>tool")
                    appendLine(msg.content)
                    appendLine("<end_of_turn>")
                }
            }
        }
        // Prompt the model to generate its turn.
        appendLine("<start_of_turn>model")
    }
}
