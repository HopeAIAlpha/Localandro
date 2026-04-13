package com.localandro.gemma4e2b.security

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf

/**
 * Data class holding the state for a pending tool confirmation request.
 *
 * @param toolName    Name of the tool requesting execution.
 * @param description Human-readable description of what the tool will do.
 * @param onResult    Callback invoked with `true` (approved) or `false` (denied).
 */
data class ToolConfirmationRequest(
    val toolName: String,
    val description: String,
    val onResult: (Boolean) -> Unit
)

/**
 * Composable dialog that requests explicit user confirmation before a
 * destructive or sensitive tool is executed.
 *
 * The dialog is shown when [request] is non-null and automatically
 * dismissed after the user responds.
 *
 * @param request Mutable state holding the current confirmation request,
 *                or `null` when no confirmation is pending.
 */
@Composable
fun ToolConfirmationDialog(request: MutableState<ToolConfirmationRequest?>) {
    val current = request.value ?: return

    AlertDialog(
        onDismissRequest = {
            current.onResult(false)
            request.value = null
        },
        title = {
            Text(
                text = "Confirmar acción",
                style = MaterialTheme.typography.titleMedium
            )
        },
        text = {
            Text(
                text = current.description,
                style = MaterialTheme.typography.bodyMedium
            )
        },
        confirmButton = {
            TextButton(onClick = {
                current.onResult(true)
                request.value = null
            }) {
                Text("Permitir")
            }
        },
        dismissButton = {
            TextButton(onClick = {
                current.onResult(false)
                request.value = null
            }) {
                Text("Denegar")
            }
        }
    )
}

/**
 * Creates a [SecurityPolicy.confirmationHandler] that bridges the coroutine
 * world with the Compose UI by posting a [ToolConfirmationRequest] to a
 * [MutableState] and suspending until the user responds.
 */
fun createConfirmationHandler(
    requestState: MutableState<ToolConfirmationRequest?>
): suspend (String, String) -> Boolean {
    return { toolName, description ->
        kotlinx.coroutines.suspendCancellableCoroutine { continuation ->
            requestState.value = ToolConfirmationRequest(
                toolName = toolName,
                description = description,
                onResult = { approved ->
                    if (continuation.isActive) {
                        continuation.resume(approved) {}
                    }
                }
            )
            continuation.invokeOnCancellation {
                requestState.value = null
            }
        }
    }
}
