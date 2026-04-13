package com.localandro.gemma4e2b

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import com.localandro.gemma4e2b.download.ModelDownloadManager
import com.localandro.gemma4e2b.inference.LiteRTLMInferenceRepository
import com.localandro.gemma4e2b.ui.chat.ChatScreen
import com.localandro.gemma4e2b.ui.setup.SetupScreen
import com.localandro.gemma4e2b.ui.theme.LocalandroTheme

/**
 * Entry point of the Localandro application.
 *
 * Implements a simple state machine at startup:
 *   1. Check whether the model file already exists in [filesDir].
 *   2. If **not** → navigate to [SetupScreen] to download it.
 *   3. If **yes** → navigate to [ChatScreen] to start inference.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val downloadManager = ModelDownloadManager(applicationContext)
        val initialModelPath = if (downloadManager.isModelDownloaded()) {
            downloadManager.getModelFile().absolutePath
        } else {
            null
        }

        // Single repository instance shared across configuration changes.
        val inferenceRepository: InferenceRepository = LiteRTLMInferenceRepository(applicationContext)

        setContent {
            LocalandroTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    AppEntryPoint(
                        initialModelPath = initialModelPath,
                        inferenceRepository = inferenceRepository
                    )
                }
            }
        }
    }
}

/**
 * Root composable implementing the first-run state machine.
 *
 * @param initialModelPath absolute path if the model is present, `null` otherwise.
 * @param inferenceRepository the on-device LLM engine abstraction.
 */
@Composable
private fun AppEntryPoint(
    initialModelPath: String?,
    inferenceRepository: InferenceRepository
) {
    var modelPath by remember { mutableStateOf(initialModelPath) }

    val currentModelPath = modelPath
    if (currentModelPath != null) {
        // Model exists → start inference UI
        ChatScreen(
            modelPath = currentModelPath,
            inferenceRepository = inferenceRepository
        )
    } else {
        // Model missing → download it first
        SetupScreen(
            onDownloadComplete = { path ->
                modelPath = path
            }
        )
    }
}
