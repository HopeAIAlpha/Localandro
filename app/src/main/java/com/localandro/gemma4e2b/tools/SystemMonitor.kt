package com.localandro.gemma4e2b.tools

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.util.Log

/**
 * System monitoring tools that allow the agent to read device health
 * metrics and auto-regulate its behaviour (e.g. reduce inference load
 * when battery is low or thermal state is critical).
 */
object SystemMonitor {

    private const val TAG = "SystemMonitor"

    // ── Battery Status ──────────────────────────────────────────────

    /**
     * Reads current battery level, charging state, and temperature.
     *
     * No arguments required.
     */
    class BatteryStatusTool(private val context: Context) : Tool {
        override val name = "battery_status"
        override val description = "Returns current battery level, charging state, and temperature."
        override val parameterSchema = emptyMap<String, String>()
        override val requiresConfirmation = false

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            return try {
                val batteryStatus: Intent? = IntentFilter(Intent.ACTION_BATTERY_CHANGED).let {
                    context.registerReceiver(null, it)
                }

                if (batteryStatus == null) {
                    return ToolResult.Failure("Unable to read battery status")
                }

                val level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
                val percentage = if (scale > 0) (level * 100) / scale else -1

                val status = batteryStatus.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
                val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                        status == BatteryManager.BATTERY_STATUS_FULL

                val plugged = batteryStatus.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1)
                val chargingSource = when (plugged) {
                    BatteryManager.BATTERY_PLUGGED_USB -> "USB"
                    BatteryManager.BATTERY_PLUGGED_AC -> "AC"
                    BatteryManager.BATTERY_PLUGGED_WIRELESS -> "Wireless"
                    else -> "Not charging"
                }

                val tempRaw = batteryStatus.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)
                val tempCelsius = tempRaw / 10.0

                ToolResult.Success(
                    "Battery: $percentage%\n" +
                    "Charging: $isCharging ($chargingSource)\n" +
                    "Temperature: ${tempCelsius}°C"
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to read battery status", e)
                ToolResult.Failure("Failed to read battery status: ${e.localizedMessage}")
            }
        }
    }

    // ── Thermal Status ──────────────────────────────────────────────

    /**
     * Reads the device's thermal status (API 29+).
     *
     * No arguments required.
     */
    class ThermalStatusTool(private val context: Context) : Tool {
        override val name = "thermal_status"
        override val description = "Returns the device's current thermal throttling status."
        override val parameterSchema = emptyMap<String, String>()
        override val requiresConfirmation = false

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            return try {
                val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager

                val thermalStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    when (powerManager.currentThermalStatus) {
                        PowerManager.THERMAL_STATUS_NONE -> "None (normal)"
                        PowerManager.THERMAL_STATUS_LIGHT -> "Light throttling"
                        PowerManager.THERMAL_STATUS_MODERATE -> "Moderate throttling"
                        PowerManager.THERMAL_STATUS_SEVERE -> "Severe throttling"
                        PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
                        PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
                        PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown imminent"
                        else -> "Unknown"
                    }
                } else {
                    "Unavailable (requires API 29+)"
                }

                ToolResult.Success("Thermal status: $thermalStatus")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to read thermal status", e)
                ToolResult.Failure("Failed to read thermal status: ${e.localizedMessage}")
            }
        }
    }

    // ── Device Info ─────────────────────────────────────────────────

    /**
     * Returns basic device information (model, Android version, available memory).
     *
     * No arguments required.
     */
    class DeviceInfoTool(private val context: Context) : Tool {
        override val name = "device_info"
        override val description = "Returns device model, Android version, and available memory."
        override val parameterSchema = emptyMap<String, String>()
        override val requiresConfirmation = false

        override suspend fun execute(arguments: Map<String, String>): ToolResult {
            return try {
                val runtime = Runtime.getRuntime()
                val freeMemMb = runtime.freeMemory() / (1024 * 1024)
                val totalMemMb = runtime.totalMemory() / (1024 * 1024)
                val maxMemMb = runtime.maxMemory() / (1024 * 1024)

                ToolResult.Success(
                    "Device: ${Build.MANUFACTURER} ${Build.MODEL}\n" +
                    "Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})\n" +
                    "JVM Memory: ${freeMemMb}MB free / ${totalMemMb}MB allocated / ${maxMemMb}MB max"
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to read device info", e)
                ToolResult.Failure("Failed to read device info: ${e.localizedMessage}")
            }
        }
    }

    /**
     * Convenience function to register all system monitor tools at once.
     */
    fun registerAll(registry: ToolRegistry, context: Context) {
        registry.register(BatteryStatusTool(context))
        registry.register(ThermalStatusTool(context))
        registry.register(DeviceInfoTool(context))
    }
}
