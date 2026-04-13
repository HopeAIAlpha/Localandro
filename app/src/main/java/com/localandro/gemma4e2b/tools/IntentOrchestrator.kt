package com.localandro.gemma4e2b.tools

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.CalendarContract
import android.util.Log

/**
 * Native Android intent-based tools that allow the agent to interact with
 * the device's installed applications.
 *
 * Each inner class is a standalone [Tool] that can be registered independently
 * in the [ToolRegistry].
 */
object IntentOrchestrator {

    private const val TAG = "IntentOrchestrator"

    // ── Open App ────────────────────────────────────────────────────

    /**
     * Opens an installed application by its package name.
     *
     * Arguments:
     * - `package_name`: Fully-qualified package name (e.g. `com.android.settings`).
     */
    class OpenAppTool(private val context: Context) : Tool {
        override val name = "open_app"
        override val description = "Opens an installed app by its package name."
        override val parameterSchema = mapOf("package_name" to "Fully-qualified Android package name")
        override val requiresConfirmation = false

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            val packageName = arguments["package_name"]
                ?: return ToolResult.Failure("Missing required parameter: package_name")

            return try {
                val launchIntent = context.packageManager.getLaunchIntentForPackage(packageName)
                    ?: return ToolResult.Failure("App not found: $packageName")
                launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(launchIntent)
                ToolResult.Success("Opened app: $packageName")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open app: $packageName", e)
                ToolResult.Failure("Failed to open app: ${e.localizedMessage}")
            }
        }
    }

    // ── Send SMS ────────────────────────────────────────────────────

    /**
     * Composes an SMS via the default messaging app (does NOT send automatically).
     *
     * Arguments:
     * - `phone_number`: Destination phone number.
     * - `message`: Text body of the SMS.
     */
    class SendSmsTool(private val context: Context) : Tool {
        override val name = "send_sms"
        override val description = "Opens the SMS app with a pre-filled message for user to confirm and send."
        override val parameterSchema = mapOf(
            "phone_number" to "Destination phone number",
            "message" to "Text body of the SMS"
        )
        override val requiresConfirmation = true

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            val phone = arguments["phone_number"]
                ?: return ToolResult.Failure("Missing required parameter: phone_number")
            val body = arguments["message"] ?: ""

            return try {
                val intent = Intent(Intent.ACTION_SENDTO).apply {
                    data = Uri.parse("smsto:$phone")
                    putExtra("sms_body", body)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(intent)
                ToolResult.Success("SMS composer opened for $phone")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to compose SMS", e)
                ToolResult.Failure("Failed to compose SMS: ${e.localizedMessage}")
            }
        }
    }

    // ── Create Calendar Event ───────────────────────────────────────

    /**
     * Creates a calendar event via the system calendar intent.
     *
     * Arguments:
     * - `title`: Event title.
     * - `description`: Optional event description.
     * - `begin_time`: Start time in epoch millis.
     * - `end_time`: End time in epoch millis.
     */
    class CreateCalendarEventTool(private val context: Context) : Tool {
        override val name = "create_calendar_event"
        override val description = "Opens the calendar app with a pre-filled event for user confirmation."
        override val parameterSchema = mapOf(
            "title" to "Event title",
            "description" to "Event description (optional)",
            "begin_time" to "Start time in epoch milliseconds",
            "end_time" to "End time in epoch milliseconds"
        )
        override val requiresConfirmation = true

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            val title = arguments["title"]
                ?: return ToolResult.Failure("Missing required parameter: title")
            val beginTime = arguments["begin_time"]?.toLongOrNull()
                ?: return ToolResult.Failure("Missing or invalid parameter: begin_time")
            val endTime = arguments["end_time"]?.toLongOrNull()
                ?: return ToolResult.Failure("Missing or invalid parameter: end_time")

            return try {
                val intent = Intent(Intent.ACTION_INSERT).apply {
                    data = CalendarContract.Events.CONTENT_URI
                    putExtra(CalendarContract.Events.TITLE, title)
                    putExtra(CalendarContract.Events.DESCRIPTION, arguments["description"] ?: "")
                    putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, beginTime)
                    putExtra(CalendarContract.EXTRA_EVENT_END_TIME, endTime)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(intent)
                ToolResult.Success("Calendar event composer opened: $title")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create calendar event", e)
                ToolResult.Failure("Failed to create calendar event: ${e.localizedMessage}")
            }
        }
    }

    /**
     * Convenience function to register all intent tools at once.
     */
    fun registerAll(registry: ToolRegistry, context: Context) {
        registry.register(OpenAppTool(context))
        registry.register(SendSmsTool(context))
        registry.register(CreateCalendarEventTool(context))
    }
}
