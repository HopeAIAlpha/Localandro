package com.localandro.gemma4e2b.tools

import android.content.Context
import android.util.Log
import java.io.File

/**
 * File system exploration tools that operate within the app's sandboxed
 * storage ([Context.getFilesDir] and [Context.getExternalFilesDir]).
 *
 * These tools never access arbitrary paths outside the app sandbox,
 * adhering to Android Scoped Storage restrictions.
 */
object FileExplorer {

    private const val TAG = "FileExplorer"

    /**
     * Lists files and directories under the app's sandbox.
     *
     * Arguments:
     * - `path`: Relative path inside the sandbox (default: root).
     * - `filter`: Optional extension filter (e.g. `pdf`, `md`, `json`).
     */
    class ListFilesTool(private val context: Context) : Tool {
        override val name = "list_files"
        override val description = "Lists files in the app sandbox directory."
        override val parameterSchema = mapOf(
            "path" to "Relative path inside app sandbox (default: root)",
            "filter" to "Optional file extension filter (e.g. pdf, md, json)"
        )
        override val requiresConfirmation = false

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            val relativePath = arguments["path"] ?: ""
            val filter = arguments["filter"]?.lowercase()

            val baseDir = context.filesDir
            val targetDir = if (relativePath.isEmpty()) baseDir
            else File(baseDir, relativePath)

            if (!targetDir.canonicalPath.startsWith(baseDir.canonicalPath)) {
                return ToolResult.Failure("Access denied: path is outside the app sandbox")
            }

            if (!targetDir.exists()) {
                return ToolResult.Failure("Directory not found: $relativePath")
            }

            val files = targetDir.listFiles()?.let { entries ->
                if (filter != null) {
                    entries.filter { it.extension.lowercase() == filter }
                } else {
                    entries.toList()
                }
            } ?: emptyList()

            val listing = files.joinToString("\n") { file ->
                val type = if (file.isDirectory) "[DIR]" else "[FILE]"
                val size = if (file.isFile) " (${file.length()} bytes)" else ""
                "$type ${file.name}$size"
            }

            return ToolResult.Success(
                if (listing.isEmpty()) "Empty directory: $relativePath"
                else listing
            )
        }
    }

    /**
     * Reads the text content of a file inside the app sandbox.
     *
     * Supports: `.txt`, `.md`, `.json`, `.csv`, `.log`.
     * Binary files (e.g. PDF) are not read directly; use [SearchFilesTool] instead.
     *
     * Arguments:
     * - `path`: Relative path to the file inside the sandbox.
     * - `max_chars`: Maximum characters to return (default: 4000).
     */
    class ReadFileTool(private val context: Context) : Tool {
        override val name = "read_file"
        override val description = "Reads text content of a file in the app sandbox."
        override val parameterSchema = mapOf(
            "path" to "Relative path to the file inside the app sandbox",
            "max_chars" to "Maximum characters to return (default: 4000)"
        )
        override val requiresConfirmation = false

        private val allowedExtensions = setOf("txt", "md", "json", "csv", "log", "xml", "yaml", "yml")

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            val relativePath = arguments["path"]
                ?: return ToolResult.Failure("Missing required parameter: path")
            val maxChars = arguments["max_chars"]?.toIntOrNull() ?: 4000

            val baseDir = context.filesDir
            val file = File(baseDir, relativePath)

            if (!file.canonicalPath.startsWith(baseDir.canonicalPath)) {
                return ToolResult.Failure("Access denied: path is outside the app sandbox")
            }

            if (!file.exists()) {
                return ToolResult.Failure("File not found: $relativePath")
            }

            if (file.extension.lowercase() !in allowedExtensions) {
                return ToolResult.Failure(
                    "Unsupported file type: .${file.extension}. " +
                    "Supported: ${allowedExtensions.joinToString(", ") { ".$it" }}"
                )
            }

            return try {
                val content = file.readText(Charsets.UTF_8)
                val truncated = if (content.length > maxChars) {
                    content.take(maxChars) + "\n[…truncated at $maxChars chars]"
                } else content
                ToolResult.Success(truncated)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to read file: $relativePath", e)
                ToolResult.Failure("Failed to read file: ${e.localizedMessage}")
            }
        }
    }

    /**
     * Searches for files whose name contains a query string.
     *
     * Arguments:
     * - `query`: Substring to search in file names.
     * - `extension`: Optional extension filter.
     */
    class SearchFilesTool(private val context: Context) : Tool {
        override val name = "search_files"
        override val description = "Searches for files by name in the app sandbox."
        override val parameterSchema = mapOf(
            "query" to "Substring to search in file names",
            "extension" to "Optional file extension filter"
        )
        override val requiresConfirmation = false

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            val query = arguments["query"]
                ?: return ToolResult.Failure("Missing required parameter: query")
            val extension = arguments["extension"]?.lowercase()

            val results = mutableListOf<String>()
            val baseDir = context.filesDir

            baseDir.walkTopDown().maxDepth(5).forEach { file ->
                if (file.isFile && file.name.contains(query, ignoreCase = true)) {
                    if (extension == null || file.extension.lowercase() == extension) {
                        val relPath = file.relativeTo(baseDir).path
                        results.add("$relPath (${file.length()} bytes)")
                    }
                }
            }

            return ToolResult.Success(
                if (results.isEmpty()) "No files found matching '$query'"
                else results.joinToString("\n")
            )
        }
    }

    /**
     * Convenience function to register all file explorer tools at once.
     */
    fun registerAll(registry: ToolRegistry, context: Context) {
        registry.register(ListFilesTool(context))
        registry.register(ReadFileTool(context))
        registry.register(SearchFilesTool(context))
    }
}
