package com.localandro.gemma4e2b.tools

import android.util.Log

/**
 * Central registry of all available tools.
 *
 * The registry is injected into the [ActionOrchestrator] and used to:
 * 1. Resolve a tool by name when a `<|tool_call|>` is parsed.
 * 2. Generate the tool catalogue section of the system prompt so the model
 *    knows which tools are available and how to invoke them.
 */
class ToolRegistry {

    companion object {
        private const val TAG = "ToolRegistry"
    }

    private val tools = mutableMapOf<String, Tool>()

    /** Registers a [tool]. Overwrites any existing tool with the same [Tool.name]. */
    fun register(tool: Tool) {
        tools[tool.name] = tool
        Log.d(TAG, "Registered tool: ${tool.name}")
    }

    /** Unregisters the tool with the given [name], if present. */
    fun unregister(name: String) {
        tools.remove(name)
    }

    /** Looks up a tool by [name]. Returns `null` if not registered. */
    fun resolve(name: String): Tool? = tools[name]

    /** Returns an unmodifiable snapshot of all registered tools. */
    fun all(): List<Tool> = tools.values.toList()

    /**
     * Builds the tool catalogue that is appended to the system prompt.
     *
     * Format (one block per tool):
     * ```
     * Available tools:
     * - tool_name: description
     *   Parameters: { "param": "description", ... }
     * ```
     */
    fun buildCatalogue(): String = buildString {
        appendLine("Available tools:")
        tools.values.sortedBy { it.name }.forEach { tool ->
            appendLine("- ${tool.name}: ${tool.description}")
            if (tool.parameterSchema.isNotEmpty()) {
                val params = tool.parameterSchema.entries.joinToString(", ") {
                    "\"${it.key}\": \"${it.value}\""
                }
                appendLine("  Parameters: { $params }")
            }
        }
    }
}
