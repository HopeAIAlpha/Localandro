package com.localandro.gemma4e2b.tools

/**
 * Contract for every executable tool that the on-device agent can invoke.
 *
 * Implementations must be stateless and safe to call from coroutine context
 * ([kotlinx.coroutines.Dispatchers.IO] for I/O-bound, [Dispatchers.Default]
 * for CPU-bound).
 */
interface Tool {

    /** Unique identifier used in the `<|tool_call|>` JSON `"name"` field. */
    val name: String

    /** Short human-readable description injected into the system prompt. */
    val description: String

    /** JSON Schema-style parameter descriptions for the system prompt. */
    val parameterSchema: Map<String, String>

    /**
     * Whether this tool can modify device state or data (destructive).
     *
     * When `true` the [SecurityPolicy] requires explicit user confirmation
     * before execution via [ToolConfirmationDialog].
     */
    val requiresConfirmation: Boolean

    /**
     * Executes the tool with the given [arguments].
     *
     * @return A [ToolResult] containing either the output or an error.
     */
    suspend fun execute(arguments: Map<String, String>): ToolResult
}
