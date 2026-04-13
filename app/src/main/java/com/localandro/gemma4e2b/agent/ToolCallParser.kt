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
 * After tool execution the result is injected as:
 * ```
 * <|tool_result|>
 * {"result": "..."}
 * <|end_tool_result|>
 * ```
 */
object ToolCallParser {

    private const val TOOL_CALL_OPEN = "<|tool_call|>"
    private const val TOOL_CALL_CLOSE = "<|end_tool_call|>"
    private const val TOOL_RESULT_OPEN = "<|tool_result|>"
    private const val TOOL_RESULT_CLOSE = "<|end_tool_result|>"

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
     *
     * @return The parsed call and the remaining text **after** the closing delimiter,
     *         or `null` if no complete tool-call block is present yet.
     */
    fun extractToolCall(modelOutput: String): Pair<ParsedToolCall, String>? {
        val openIdx = modelOutput.indexOf(TOOL_CALL_OPEN)
        if (openIdx == -1) return null

        val closeIdx = modelOutput.indexOf(TOOL_CALL_CLOSE, openIdx + TOOL_CALL_OPEN.length)
        if (closeIdx == -1) return null // block not yet complete

        val jsonBlock = modelOutput
            .substring(openIdx + TOOL_CALL_OPEN.length, closeIdx)
            .trim()

        val remaining = modelOutput.substring(closeIdx + TOOL_CALL_CLOSE.length)

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
     * Wraps a tool result into the injection format that the model expects
     * when generating its next turn.
     */
    fun formatToolResult(result: String): String =
        "$TOOL_RESULT_OPEN\n$result\n$TOOL_RESULT_CLOSE"

    /** Returns `true` when the model output contains a (possibly incomplete) tool-call opener. */
    fun containsToolCallStart(modelOutput: String): Boolean =
        modelOutput.contains(TOOL_CALL_OPEN)

    /**
     * Strips all tool-call / tool-result delimiters from [text] so that the
     * user-facing message is clean.
     */
    fun stripToolTokens(text: String): String = text
        .replace(TOOL_CALL_OPEN, "")
        .replace(TOOL_CALL_CLOSE, "")
        .replace(TOOL_RESULT_OPEN, "")
        .replace(TOOL_RESULT_CLOSE, "")
        .trim()
}
