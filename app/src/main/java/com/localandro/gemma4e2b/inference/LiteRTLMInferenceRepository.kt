package com.localandro.gemma4e2b.inference

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.localandro.gemma4e2b.domain.repository.InferenceConfig
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

/**
 * Implementation of [InferenceRepository] backed by the LiteRT-LM
 * native inference engine.
 *
 * Uses [Engine] for model loading with GPU offload. A **fresh**
 * [Conversation][com.google.ai.edge.litertlm.Conversation] is
 * created for every [streamResponse] call so that the caller has
 * full control over what goes into context and no stale state
 * accumulates inside the native C++ layer.
 *
 * **Thread safety:** A [Mutex] serialises all calls to the native C++
 * engine to prevent concurrent JNI access that would corrupt native
 * memory pointers (SIGSEGV in `liblitertlm_jni.so`).
 *
 * @param context Application context required by the native engine.
 */
class LiteRTLMInferenceRepository(
    private val context: Context
) : InferenceRepository {

    companion object {
        private const val TAG = "LiteRTLMInference"
    }

    @Volatile
    private var engine: Engine? = null

    private var config: InferenceConfig = InferenceConfig()

    /**
     * Mutex that serialises access to the native LiteRT-LM engine.
     * The C++ layer is **not** thread-safe: concurrent calls corrupt
     * internal state and cause SIGSEGV. This mutex ensures only one
     * inference request is in-flight at any time.
     */
    private val inferenceMutex = Mutex()

    // ── Lifecycle ────────────────────────────────────────────────────

    override suspend fun initialize(
        modelPath: String,
        config: InferenceConfig
    ): Result<Unit> = runCatching {
        inferenceMutex.withLock {
            val file = File(modelPath)
            require(file.exists()) { "Model file not found: $modelPath" }

            this.config = config

            // Build engine config targeting GPU backend (Adreno 710).
            val engineConfig = EngineConfig(
                modelPath = modelPath,
                backend = Backend.GPU(),
                cacheDir = context.cacheDir.path
            )

            Log.i(TAG, "Creating LiteRT-LM Engine from: $modelPath")
            val eng = Engine(engineConfig)
            eng.initialize()
            engine = eng

            // Verify the engine works by creating and immediately
            // closing a throw-away conversation.
            val warmup = eng.createConversation(buildConversationConfig())
            warmup.close()

            Log.i(TAG, "Engine initialized successfully (GPU offload active)")
        }
    }

    override fun isInitialized(): Boolean = engine != null

    override suspend fun release() {
        inferenceMutex.withLock {
            Log.i(TAG, "Releasing engine resources")
            try {
                engine?.close()
            } catch (e: Exception) {
                Log.w(TAG, "Error closing engine: ${e.message}")
            }
            engine = null
        }
    }

    // ── Inference ────────────────────────────────────────────────────

    /**
     * Creates a **fresh** [Conversation] for each call so that no stale
     * internal state accumulates across agentic-loop iterations.
     *
     * A persistent (reused) Conversation would cause context to grow
     * unboundedly: each iteration sends the full prompt, and the native
     * engine also retains all previous turns, effectively doubling the
     * context on every call until the native buffer overflows → SIGSEGV.
     */
    override fun streamResponse(prompt: String): Flow<String> {
        val eng = engine
            ?: throw IllegalStateException("Engine not initialized – call initialize() first")

        return kotlinx.coroutines.flow.flow {
            inferenceMutex.withLock {
                val conv = eng.createConversation(buildConversationConfig())
                try {
                    conv.sendMessageAsync(prompt)
                        .map { message -> message.toString() }
                        .catch { e ->
                            Log.e(TAG, "Streaming error: ${e.message}", e)
                            throw e
                        }
                        .collect { emit(it) }
                } finally {
                    try {
                        conv.close()
                    } catch (e: Exception) {
                        Log.w(TAG, "Error closing conversation: ${e.message}")
                    }
                }
            }
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────

    private fun buildConversationConfig(): ConversationConfig =
        ConversationConfig(
            samplerConfig = SamplerConfig(
                temperature = config.temperature.toDouble(),
                topK = config.topK,
                topP = config.topP.toDouble()
            )
        )
}
