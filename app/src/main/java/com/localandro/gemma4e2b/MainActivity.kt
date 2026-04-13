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
import com.localandro.gemma4e2b.agent.ActionOrchestrator
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import com.localandro.gemma4e2b.download.ModelDownloadManager
import com.localandro.gemma4e2b.inference.RemoteInferenceRepository
import com.localandro.gemma4e2b.memory.LongTermMemory
import com.localandro.gemma4e2b.memory.SlidingWindowContext
import com.localandro.gemma4e2b.security.SecurityPolicy
import com.localandro.gemma4e2b.tools.FileExplorer
import com.localandro.gemma4e2b.tools.IntentOrchestrator
import com.localandro.gemma4e2b.tools.SystemMonitor
import com.localandro.gemma4e2b.tools.ToolRegistry
import com.localandro.gemma4e2b.ui.chat.ChatScreen
import com.localandro.gemma4e2b.ui.setup.SetupScreen
import com.localandro.gemma4e2b.ui.theme.LocalandroTheme

/**
 * Entry point of the Localandro application.
 *
 * Implements a simple state machine at startup:
 *   1. Check whether the model file already exists in [filesDir].
 *   2. If **not** → navigate to [SetupScreen] to download it.
 *   3. If **yes** → navigate to [ChatScreen] with the full agentic stack.
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
        // Uses IPC to delegate inference to a separate process (:inference_engine)
        // so that native JNI crashes don't kill the main UI process.
        val inferenceRepository: InferenceRepository = RemoteInferenceRepository(applicationContext)

        // ── Agentic stack setup ─────────────────────────────────────
        val toolRegistry = ToolRegistry()
        IntentOrchestrator.registerAll(toolRegistry, applicationContext)
        FileExplorer.registerAll(toolRegistry, applicationContext)
        SystemMonitor.registerAll(toolRegistry, applicationContext)

        val securityPolicy = SecurityPolicy()
        val longTermMemory = LongTermMemory(applicationContext)
        val slidingWindowContext = SlidingWindowContext()

        val orchestrator = ActionOrchestrator(
            inferenceRepository = inferenceRepository,
            toolRegistry = toolRegistry,
            securityPolicy = securityPolicy,
            longTermMemory = longTermMemory,
            slidingWindowContext = slidingWindowContext
        )

        setContent {
            LocalandroTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    AppEntryPoint(
                        initialModelPath = initialModelPath,
                        inferenceRepository = inferenceRepository,
                        orchestrator = orchestrator
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
 * @param orchestrator the agentic action orchestrator.
 */
@Composable
private fun AppEntryPoint(
    initialModelPath: String?,
    inferenceRepository: InferenceRepository,
    orchestrator: ActionOrchestrator
) {
    var modelPath by remember { mutableStateOf(initialModelPath) }

    val currentModelPath = modelPath
    if (currentModelPath != null) {
        // Model exists → start inference UI with agentic orchestration
        ChatScreen(
            modelPath = currentModelPath,
            inferenceRepository = inferenceRepository,
            orchestrator = orchestrator
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
