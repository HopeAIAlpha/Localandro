package com.localandro.gemma4e2b.memory

import com.localandro.gemma4e2b.domain.model.Message
import com.localandro.gemma4e2b.domain.model.MessageRole

/**
 * Sliding-window context manager that keeps the most relevant messages
 * within a fixed token budget while maintaining **chronological order**.
 *
 * Priority order (never evicted first → evicted first):
 * 1. System prompt – always retained.
 * 2. Most recent messages (all types) – retained newest-first until budget.
 * 3. Older messages – evicted from the oldest end.
 *
 * All message types (USER, MODEL, TOOL_CALL, TOOL_RESULT) are kept in
 * their natural chronological order so that the model sees a coherent
 * conversation flow. Tool results are NOT separated to the end — they
 * appear immediately after their corresponding tool call.
 *
 * Token estimation uses a simple character-based heuristic
 * (≈4 characters per token for English text, adjusted for multilingual).
 */
class SlidingWindowContext(
    private val maxTokens: Int = DEFAULT_MAX_TOKENS,
    private val charsPerToken: Int = DEFAULT_CHARS_PER_TOKEN
) {

    companion object {
        /**
         * Default context window budget for the prompt.
         * Set to 3072 (not the full 4096 model context) to leave ~1024
         * tokens for the model's generation within the E2B's 4096 limit.
         */
        const val DEFAULT_MAX_TOKENS = 3072

        /** Conservative estimate: ~4 chars/token for mixed-language content. */
        const val DEFAULT_CHARS_PER_TOKEN = 4
    }

    /**
     * Builds a token-budgeted context window from the full conversation
     * history, ensuring the system prompt is always included and messages
     * are in chronological order.
     *
     * @param systemPrompt The immutable system prompt (highest priority).
     * @param messages     The full conversation history.
     * @param memoryContext Additional context from [LongTermMemory] (optional).
     * @return A list of messages that fit within the token budget, in
     *         chronological order.
     */
    fun buildContext(
        systemPrompt: String,
        messages: List<Message>,
        memoryContext: String = ""
    ): List<Message> {
        val maxChars = maxTokens * charsPerToken

        // 1. Reserve space for the system prompt and memory context.
        val systemContent = if (memoryContext.isNotEmpty()) {
            "$systemPrompt\n\n$memoryContext"
        } else {
            systemPrompt
        }
        val systemMessage = Message(
            id = "system-prompt",
            role = MessageRole.SYSTEM,
            content = systemContent
        )
        var usedChars = systemContent.length

        // 2. Working backwards from newest, include as many messages as
        //    fit while maintaining chronological order. This keeps all
        //    message types (USER, MODEL, TOOL_CALL, TOOL_RESULT) in
        //    their natural interleaved order.
        val selected = mutableListOf<Message>()

        for (msg in messages.reversed()) {
            val msgChars = msg.content.length
            if (usedChars + msgChars > maxChars) break
            selected.add(0, msg) // prepend to maintain chronological order
            usedChars += msgChars
        }

        // 3. Assemble final context.
        return buildList {
            add(systemMessage)
            addAll(selected)
        }
    }

    /** Estimates the token count of a string. */
    fun estimateTokens(text: String): Int = text.length / charsPerToken
}
