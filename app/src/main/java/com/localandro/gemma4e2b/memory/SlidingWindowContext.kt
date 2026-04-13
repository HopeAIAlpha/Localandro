package com.localandro.gemma4e2b.memory

import com.localandro.gemma4e2b.domain.model.Message
import com.localandro.gemma4e2b.domain.model.MessageRole

/**
 * Sliding-window context manager that keeps the most relevant messages
 * within a fixed token budget.
 *
 * Priority order (never evicted first → evicted first):
 * 1. System prompt – always retained.
 * 2. Last tool results – always retained (needed for the current turn).
 * 3. Recent user and model messages – retained newest-first until budget.
 * 4. Older messages – evicted from the oldest end.
 *
 * Token estimation uses a simple character-based heuristic
 * (≈4 characters per token for English text, adjusted for multilingual).
 */
class SlidingWindowContext(
    private val maxTokens: Int = DEFAULT_MAX_TOKENS,
    private val charsPerToken: Int = DEFAULT_CHARS_PER_TOKEN
) {

    companion object {
        /** Default context window size matching Gemma 4 E2B's capacity. */
        const val DEFAULT_MAX_TOKENS = 4096

        /** Conservative estimate: ~4 chars/token for mixed-language content. */
        const val DEFAULT_CHARS_PER_TOKEN = 4
    }

    /**
     * Builds a token-budgeted context window from the full conversation
     * history, ensuring the system prompt and latest tool results are
     * always included.
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

        // 2. Separate latest tool results (always include).
        val recentToolResults = messages.filter {
            it.role == MessageRole.TOOL_RESULT || it.role == MessageRole.TOOL_CALL
        }.takeLast(4) // Keep at most 4 most recent tool interactions.

        recentToolResults.forEach { usedChars += it.content.length }

        // 3. Fill remaining budget with conversation messages (newest first).
        val conversationMessages = messages.filter {
            it.role == MessageRole.USER || it.role == MessageRole.MODEL
        }
        val selectedConversation = mutableListOf<Message>()

        for (msg in conversationMessages.reversed()) {
            val msgChars = msg.content.length
            if (usedChars + msgChars > maxChars) break
            selectedConversation.add(0, msg) // prepend to maintain order
            usedChars += msgChars
        }

        // 4. Assemble final context in chronological order.
        return buildList {
            add(systemMessage)
            addAll(selectedConversation)
            addAll(recentToolResults)
        }
    }

    /** Estimates the token count of a string. */
    fun estimateTokens(text: String): Int = text.length / charsPerToken
}
