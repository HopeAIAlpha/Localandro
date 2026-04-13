package com.localandro.gemma4e2b.ui.setup

import android.view.WindowManager
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.localandro.gemma4e2b.download.DownloadState
import com.localandro.gemma4e2b.download.ModelDownloadManager

/**
 * First-run setup screen that downloads the Gemma 4 E2B model from
 * Hugging Face with a real progress bar (0 %–100 %).
 *
 * Uses [WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON] to prevent the
 * display from turning off while the large model file is downloading.
 *
 * @param onDownloadComplete Callback invoked with the path of the
 *   downloaded model file once the download finishes successfully.
 */
@Composable
fun SetupScreen(onDownloadComplete: (String) -> Unit) {
    val context = LocalContext.current
    val downloadManager = remember { ModelDownloadManager(context) }

    var downloadProgress by remember { mutableStateOf(0f) }
    var statusText by remember { mutableStateOf("Preparando descarga del modelo…") }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var isDownloading by remember { mutableStateOf(true) }
    var retryTrigger by remember { mutableIntStateOf(0) }

    val animatedProgress by animateFloatAsState(
        targetValue = downloadProgress,
        animationSpec = tween(durationMillis = 300),
        label = "download_progress"
    )

    // Keep the screen awake while downloading.
    val activity = context as? android.app.Activity
    DisposableEffect(Unit) {
        activity?.window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            activity?.window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    // Start the download when the composable enters composition or on retry.
    LaunchedEffect(retryTrigger) {
        downloadManager.download().collect { state ->
            when (state) {
                is DownloadState.Starting -> {
                    statusText = "Conectando con Hugging Face…"
                    isDownloading = true
                }
                is DownloadState.Progress -> {
                    val percent = state.percent
                    downloadProgress = if (percent >= 0) percent / 100f else 0f
                    val mbDownloaded = state.bytesDownloaded / (1024.0 * 1024.0)
                    val mbTotal = state.totalBytes / (1024.0 * 1024.0)
                    statusText = if (percent >= 0) {
                        "Descargando modelo… ${percent}%\n" +
                                String.format("%.1f MB / %.1f MB", mbDownloaded, mbTotal)
                    } else {
                        "Descargando modelo…\n" +
                                String.format("%.1f MB descargados", mbDownloaded)
                    }
                }
                is DownloadState.Completed -> {
                    downloadProgress = 1f
                    statusText = "¡Descarga completa!"
                    isDownloading = false
                    onDownloadComplete(state.filePath)
                }
                is DownloadState.Error -> {
                    errorMessage = state.message
                    isDownloading = false
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Gemma 4 E2B",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "Configuración inicial",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(48.dp))

        if (errorMessage != null) {
            Text(
                text = "Error: $errorMessage",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(16.dp))

            TextButton(onClick = {
                errorMessage = null
                isDownloading = true
                downloadProgress = 0f
                retryTrigger++
            }) {
                Text("Reintentar")
            }
        } else {
            if (isDownloading && downloadProgress <= 0f) {
                CircularProgressIndicator(
                    modifier = Modifier.size(64.dp),
                    strokeWidth = 4.dp
                )
            } else {
                LinearProgressIndicator(
                    progress = { animatedProgress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp),
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = statusText,
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.fillMaxWidth()
            )
        }

        Spacer(modifier = Modifier.height(48.dp))

        Text(
            text = "El modelo se descarga una sola vez.\nNo cierres la aplicación.",
            style = MaterialTheme.typography.bodySmall,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.fillMaxWidth()
        )
    }
}
