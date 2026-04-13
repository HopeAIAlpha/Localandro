package com.localandro.gemma4e2b.agent

import org.json.JSONException
import org.json.JSONObject

/**
 * Parses Gemma 4 tool-call tokens emitted inside model output.
 *
 * Expected wire format (Gemma 4 function-calling protocol):
 * ```
 * <|tool_call|>
 * {"name": "tool_name", "arguments": {"key": "value"}}
 * <|end_tool_call|>
 * ```
 *
 * The parser is **permissive**: it also matches common model hallucinations
 * such as `<tool_call>`, `<|tool_call>`, `<tool_call|>`, and `</tool_call>`.
 *
 * After tool execution the result is injected as:
 * ```
 * <|tool_result|>
 * {"result": "..."}
 * <|end_tool_result|>
 * ```
 */
object ToolCallParser {

    // Canonical delimiters used for output formatting.
    private const val TOOL_CALL_OPEN = "<|tool_call|>"
    private const val TOOL_CALL_CLOSE = "<|end_tool_call|>"
    private const val TOOL_RESULT_OPEN = "<|tool_result|>"
    private const val TOOL_RESULT_CLOSE = "<|end_tool_result|>"

    /**
     * Permissive regex matching common hallucinated variants of the
     * tool-call open delimiter:
     * `<|tool_call|>`, `<tool_call>`, `<|tool_call>`, `<tool_call|>`, `< tool_call >`
     */
    private val TOOL_CALL_OPEN_REGEX =
        Regex("""<\|?\s*tool_call\s*\|?>""")

    /**
     * Permissive regex matching common hallucinated variants of the
     * tool-call close delimiter:
     * `<|end_tool_call|>`, `</tool_call>`, `<|end_tool_call>`, `<end_tool_call|>`,
     * `<end_tool_call>`, `< /tool_call >`, `<| end_tool_call |>`
     */
    private val TOOL_CALL_CLOSE_REGEX =
        Regex("""<\|?\s*/?(?:end_)?tool_call\s*\|?>""")

    /** Permissive regex for tool-result open delimiter variants. */
    private val TOOL_RESULT_OPEN_REGEX =
        Regex("""<\|?\s*tool_result\s*\|?>""")

    /** Permissive regex for tool-result close delimiter variants. */
    private val TOOL_RESULT_CLOSE_REGEX =
        Regex("""<\|?\s*/?(?:end_)?tool_result\s*\|?>""")

    /**
     * Combined regex that matches ANY tool-related delimiter variant.
     * Used by [stripToolTokens] and [getVisibleText] to filter UI output.
     */
    private val ANY_TOOL_DELIMITER_REGEX =
        Regex("""<\|?\s*/?(?:end_)?(?:tool_call|tool_result)\s*\|?>""")

    /**
     * Detects a partial (potentially incomplete) tool-call opener building up
     * at the end of streamed text. Used to suppress premature UI emission.
     * Matches prefixes like `<`, `<|`, `<|tool`, `<tool_c`, etc.
     */
    private val PARTIAL_TOOL_CALL_OPEN_REGEX =
        Regex("""<\|?\s*t(?:o(?:o(?:l(?:_(?:c(?:a(?:l(?:l\s*\|?>?)?)?)?)?)?)?)?)?$""")

    /**
     * A parsed tool invocation extracted from model output.
     *
     * @param name      Tool identifier that must match a key in the ToolRegistry.
     * @param arguments Arbitrary key-value arguments for the tool.
     * @param raw       Original raw text between the delimiters (for debugging).
     */
    data class ParsedToolCall(
        val name: String,
        val arguments: Map<String, String>,
        val raw: String
    )

    /**
     * Attempts to extract a [ParsedToolCall] from the model's accumulated output.
     * Uses permissive regex to tolerate hallucinated delimiter variants.
     *
     * @return The parsed call and the remaining text **after** the closing delimiter,
     *         or `null` if no complete tool-call block is present yet.
     */
    fun extractToolCall(modelOutput: String): Pair<ParsedToolCall, String>? {
        val openMatch = TOOL_CALL_OPEN_REGEX.find(modelOutput) ?: return null
        val searchStart = openMatch.range.last + 1

        // Find the close delimiter AFTER the open. We need a close that is NOT
        // the same match as the open (the close regex is a superset of the open).
        val closeMatch = findCloseDelimiter(modelOutput, searchStart) ?: return null

        val jsonBlock = modelOutput
            .substring(searchStart, closeMatch.range.first)
            .trim()

        val remaining = modelOutput.substring(closeMatch.range.last + 1)

        return try {
            val json = JSONObject(jsonBlock)
            val name = json.getString("name")
            val argsJson = json.optJSONObject("arguments") ?: JSONObject()
            val args = mutableMapOf<String, String>()
            argsJson.keys().forEach { key -> args[key] = argsJson.optString(key, "") }
            ParsedToolCall(name = name, arguments = args, raw = jsonBlock) to remaining
        } catch (_: JSONException) {
            null
        }
    }

    /**
     * Finds the closing delimiter after [startIndex]. The close regex is a
     * superset of the open regex, so we look for either `</tool_call>` style
     * or `<|end_tool_call|>` style, but also accept any match of the close
     * regex that is positioned after the JSON content.
     */
    private fun findCloseDelimiter(text: String, startIndex: Int): MatchResult? {
        val sub = text.substring(startIndex)
        // First try dedicated end/close patterns
        val endMatch = Regex("""<\|?\s*/?(?:end_)?tool_call\s*\|?>""").find(sub)
        return endMatch?.let {
            // Adjust range to be relative to original string
            Regex("""<\|?\s*/?(?:end_)?tool_call\s*\|?>""").find(text, startIndex)
        }
    }

    /**
     * Wraps a tool result into the injection format that the model expects
     * when generating its next turn.
     */
    fun formatToolResult(result: String): String =
        "$TOOL_RESULT_OPEN\n$result\n$TOOL_RESULT_CLOSE"

    /**
     * Returns `true` when the model output contains a (possibly incomplete)
     * tool-call opener — using permissive matching.
     */
    fun containsToolCallStart(modelOutput: String): Boolean =
        TOOL_CALL_OPEN_REGEX.containsMatchIn(modelOutput)

    /**
     * Returns `true` when the accumulated text ends with what looks like the
     * beginning of a tool-call delimiter being streamed token-by-token.
     * This is used to suppress partial delimiters from the UI stream.
     */
    fun endsWithPartialToolToken(text: String): Boolean {
        // Check the last 20 chars for a partial match
        val tail = if (text.length > 20) text.substring(text.length - 20) else text
        return PARTIAL_TOOL_CALL_OPEN_REGEX.containsMatchIn(tail)
    }

    /**
     * Strips all tool-call / tool-result delimiters (including hallucinated
     * variants) from [text] so that the user-facing message is clean.
     */
    fun stripToolTokens(text: String): String =
        ANY_TOOL_DELIMITER_REGEX.replace(text, "").trim()

    /**
     * Extracts only the text that should be visible to the user from the
     * model's streaming output. Removes:
     * 1. Everything from the first tool-call opener onwards.
     * 2. Any hallucinated delimiter tokens.
     *
     * @param accumulatedOutput The full accumulated model output so far.
     * @return The clean, user-visible portion of the output.
     */
    fun getVisibleText(accumulatedOutput: String): String {
        // If a tool-call block has started, only show text before it.
        val openMatch = TOOL_CALL_OPEN_REGEX.find(accumulatedOutput)
        val textBeforeToolCall = if (openMatch != null) {
            accumulatedOutput.substring(0, openMatch.range.first)
        } else {
            accumulatedOutput
        }
        // Strip any stray delimiter tokens that may have leaked.
        return stripToolTokens(textBeforeToolCall)
    }
}
