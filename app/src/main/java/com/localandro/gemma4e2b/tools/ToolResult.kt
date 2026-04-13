package com.localandro.gemma4e2b.tools

/**
 * Outcome of a [Tool.execute] invocation.
 */
sealed class ToolResult {

    /**
     * The tool completed successfully.
     *
     * @param output Textual output to be injected into the model context
     *               via the `<|tool_result|>` protocol.
     */
    data class Success(val output: String) : ToolResult()

    /**
     * The tool failed.
     *
     * @param reason Human-readable error that is shown both to the model
     *               (so it can self-correct) and to the user.
     */
    data class Failure(val reason: String) : ToolResult()

    /**
     * Execution was blocked by the security policy because the user
     * declined the confirmation dialog.
     */
    data object Denied : ToolResult()
}
