package com.localandro.gemma4e2b.agent

import android.util.Log
import org.json.JSONException
import org.json.JSONObject

/**
 * Parses Gemma 4 tool-call tokens emitted inside model output.
 *
 * **Official Gemma 4 wire format** (from Google AI docs):
 *
 * Tool definition (system prompt):
 * ```
 * <|tool>declaration:func_name{description:<|"|>...<|"|>,parameters:{...}}<tool|>
 * ```
 *
 * Model emits a tool call:
 * ```
 * <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>
 * ```
 *
 * Application injects a tool result:
 * ```
 * <|tool_response>response:func_name{key:<|"|>value<|"|>}<tool_response|>
 * ```
 *
 * String values are delimited with `<|"|>` (not regular quotes).
 *
 * The parser is **permissive**: it also tolerates common hallucinations
 * such as `<tool_call>`, `<|tool_call|>`, `</tool_call>`, and mixed
 * forms, falling back to JSON extraction when the native KV format
 * is not present.
 */
object ToolCallParser {

    private const val TAG = "ToolCallParser"

    // ── Canonical delimiters (Gemma 4 native) ───────────────────────

    private const val TOOL_CALL_OPEN   = "<|tool_call>"
    private const val TOOL_CALL_CLOSE  = "<tool_call|>"
    private const val TOOL_RESP_OPEN   = "<|tool_response>"
    private const val TOOL_RESP_CLOSE  = "<tool_response|>"
    private const val TOOL_DEF_OPEN    = "<|tool>"
    private const val TOOL_DEF_CLOSE   = "<tool|>"
    /** Gemma 4 string value delimiter. */
    private const val STR_DELIM        = "<|\"|>"

    // ── Permissive regex patterns ───────────────────────────────────
    // These catch both the official tokens AND common hallucinations.

    /**
     * Matches any variant of a tool-call open token:
     * `<|tool_call>`, `<|tool_call|>`, `<tool_call>`, `<tool_call|>`
     */
    private val TOOL_CALL_OPEN_REGEX =
        Regex("""<\|?\s*tool_call\s*\|?>""")

    /**
     * Matches any variant of a tool-call close token:
     * `<tool_call|>`, `<|end_tool_call|>`, `<|end_tool_call>`,
     * `</tool_call>`, `<end_tool_call>`
     */
    private val TOOL_CALL_CLOSE_REGEX =
        Regex("""</?(?:\|?\s*(?:end_)?tool_call\s*\|?|tool_call\|)>""")

    /** Matches any variant of tool_response open. */
    private val TOOL_RESP_OPEN_REGEX =
        Regex("""<\|?\s*tool_(?:response|result)\s*\|?>""")

    /** Matches any variant of tool_response close. */
    private val TOOL_RESP_CLOSE_REGEX =
        Regex("""</?(?:\|?\s*(?:end_)?tool_(?:response|result)\s*\|?|tool_(?:response|result)\|)>""")

    /** Matches ANY tool delimiter variant (for stripping). */
    private val ANY_TOOL_DELIMITER_REGEX =
        Regex("""</?(?:\|?\s*(?:end_)?(?:tool_call|tool_response|tool_result|tool)\s*\|?|(?:tool_call|tool_response|tool_result|tool)\|)>""")

    // ── Data classes ────────────────────────────────────────────────

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

    // ── Public API ──────────────────────────────────────────────────

    /**
     * Attempts to extract a [ParsedToolCall] from the model's accumulated
     * output. Tries the native Gemma 4 `call:name{...}` format first,
     * then falls back to JSON extraction for hallucinated outputs.
     *
     * @return The parsed call and the remaining text **after** the closing
     *         delimiter, or `null` if no complete tool-call block is present.
     */
    fun extractToolCall(modelOutput: String): Pair<ParsedToolCall, String>? {
        val openMatch = TOOL_CALL_OPEN_REGEX.find(modelOutput) ?: return null
        val afterOpen = openMatch.range.last + 1

        // Find the close delimiter AFTER the open token.
        val closeMatch = findCloseAfter(modelOutput, afterOpen) ?: return null

        val payload = modelOutput
            .substring(afterOpen, closeMatch.range.first)
            .trim()

        val remaining = modelOutput.substring(closeMatch.range.last + 1)

        // Try native Gemma 4 format: call:func_name{arg:<|"|>val<|"|>, ...}
        val nativeResult = parseNativePayload(payload)
        if (nativeResult != null) {
            return nativeResult.copy(raw = payload) to remaining
        }

        // Fallback: try JSON format (hallucinated or prompted)
        val jsonResult = parseJsonPayload(payload)
        if (jsonResult != null) {
            return jsonResult.copy(raw = payload) to remaining
        }

        Log.w(TAG, "Could not parse tool-call payload: ${payload.take(200)}")
        return null
    }

    /**
     * Formats a tool result in the official Gemma 4 `<|tool_response>` format.
     *
     * @param toolName Name of the tool that was executed.
     * @param result   The tool's output string.
     */
    fun formatToolResult(toolName: String, result: String): String {
        // Collapse newlines into spaces so the tool result stays on a
        // single line, matching the Gemma 4 wire format the model expects.
        val singleLine = result.replace("\n", " ").replace("\"", "\\\"")
        return "${TOOL_RESP_OPEN}response:$toolName{result:${STR_DELIM}$singleLine${STR_DELIM}}${TOOL_RESP_CLOSE}"
    }

    /**
     * Returns `true` when the model output contains a (possibly incomplete)
     * tool-call opener — using permissive matching.
     */
    fun containsToolCallStart(modelOutput: String): Boolean =
        TOOL_CALL_OPEN_REGEX.containsMatchIn(modelOutput)

    /**
     * Strips all tool-related delimiter tokens (including hallucinated
     * variants) from [text] so the user-facing message is clean.
     */
    fun stripToolTokens(text: String): String =
        ANY_TOOL_DELIMITER_REGEX.replace(text, "").trim()

    /**
     * Extracts only the text that should be visible to the user.
     * Everything from the first tool-call opener onwards is removed.
     */
    fun getVisibleText(accumulatedOutput: String): String {
        val openMatch = TOOL_CALL_OPEN_REGEX.find(accumulatedOutput)
        val textBefore = if (openMatch != null) {
            accumulatedOutput.substring(0, openMatch.range.first)
        } else {
            accumulatedOutput
        }
        return stripToolTokens(textBefore)
    }

    /**
     * Builds the Gemma 4 tool definition block for inclusion in the
     * system prompt.
     *
     * @param name         Tool name.
     * @param description  Human-readable description.
     * @param parameters   Parameter name → description map.
     */
    fun buildToolDefinition(
        name: String,
        description: String,
        parameters: Map<String, String>
    ): String = buildString {
        append("${TOOL_DEF_OPEN}declaration:$name{")
        append("description:$STR_DELIM$description$STR_DELIM,")
        append("parameters:{properties:{")
        parameters.entries.forEachIndexed { index, (key, desc) ->
            if (index > 0) append(",")
            append("$key:{description:$STR_DELIM$desc$STR_DELIM,type:${STR_DELIM}STRING$STR_DELIM}")
        }
        append("},type:${STR_DELIM}OBJECT$STR_DELIM}}")
        append(TOOL_DEF_CLOSE)
    }

    // ── Internal helpers ────────────────────────────────────────────

    /**
     * Finds a close-delimiter match after [startIndex] in [text].
     * Must find a match distinct from the open token.
     */
    private fun findCloseAfter(text: String, startIndex: Int): MatchResult? {
        val sub = text.substring(startIndex)
        return TOOL_CALL_CLOSE_REGEX.find(sub)?.let {
            TOOL_CALL_CLOSE_REGEX.find(text, startIndex)
        }
    }

    /**
     * Parses the native Gemma 4 `call:func_name{key:<|"|>val<|"|>,...}` format.
     */
    private fun parseNativePayload(payload: String): ParsedToolCall? {
        // Pattern: call:function_name{...}
        val callMatch = Regex("""call:(\w+)\{(.*)\}""", RegexOption.DOT_MATCHES_ALL)
            .find(payload) ?: return null

        val funcName = callMatch.groupValues[1]
        val argsBlock = callMatch.groupValues[2]

        val args = parseGemma4Args(argsBlock)
        return ParsedToolCall(name = funcName, arguments = args, raw = payload)
    }

    /**
     * Parses Gemma 4 key-value arguments from inside `{...}`.
     * Handles `<|"|>` delimited string values and bare values.
     */
    private fun parseGemma4Args(block: String): Map<String, String> {
        val args = mutableMapOf<String, String>()
        // Match: key:<|"|>value<|"|>  or  key:barevalue
        val pattern = Regex("""(\w+):<\|"\|>(.*?)<\|"\|>""")
        pattern.findAll(block).forEach { match ->
            args[match.groupValues[1]] = match.groupValues[2]
        }
        // Also try bare values like key:123
        if (args.isEmpty()) {
            val barePattern = Regex("""(\w+):([^,}]+)""")
            barePattern.findAll(block).forEach { match ->
                val key = match.groupValues[1]
                val value = match.groupValues[2].trim()
                if (key !in args) {
                    args[key] = value
                }
            }
        }
        return args
    }

    /**
     * Fallback: tries to parse the payload as JSON.
     * Handles `{"name": "...", "arguments": {...}}` format from hallucinations.
     */
    private fun parseJsonPayload(payload: String): ParsedToolCall? {
        // Try to find JSON within the payload
        val jsonStr = extractJsonFromString(payload) ?: return null
        return try {
            val json = JSONObject(jsonStr)
            val name = json.getString("name")
            val argsJson = json.optJSONObject("arguments") ?: JSONObject()
            val args = mutableMapOf<String, String>()
            argsJson.keys().forEach { key -> args[key] = argsJson.optString(key, "") }
            ParsedToolCall(name = name, arguments = args, raw = payload)
        } catch (_: JSONException) {
            null
        }
    }

    /**
     * Extracts a JSON object substring from text (finds first `{` and
     * its matching `}`).
     */
    private fun extractJsonFromString(text: String): String? {
        val start = text.indexOf('{')
        if (start == -1) return null

        var depth = 0
        for (i in start until text.length) {
            when (text[i]) {
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) return text.substring(start, i + 1)
                }
            }
        }
        return null
    }
}
