package com.localandro.gemma4e2b.inference

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
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
 * Uses [Engine] for model loading with GPU offload, and
 * [Conversation] for stateful, streaming token generation
 * targeting the Gemma 4 E2B `.litertlm` model.
 *
 * **Thread safety:** A [Mutex] serialises all calls to the native C++
 * engine ([Conversation.sendMessageAsync]) to prevent concurrent JNI
 * access that would corrupt native memory pointers (SIGSEGV).
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

    @Volatile
    private var conversation: Conversation? = null

    private var config: InferenceConfig = InferenceConfig()

    /**
     * Mutex that serialises access to the native LiteRT-LM engine.
     * The C++ layer is **not** thread-safe: concurrent calls to
     * `sendMessageAsync` corrupt internal state and cause SIGSEGV
     * in `liblitertlm_jni.so`. This mutex ensures only one
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

            // Create a conversation with sampling parameters.
            val conversationConfig = ConversationConfig(
                samplerConfig = SamplerConfig(
                    temperature = config.temperature.toDouble(),
                    topK = config.topK,
                    topP = config.topP.toDouble()
                )
            )

            Log.i(TAG, "Creating Conversation (temp=${config.temperature}, topK=${config.topK})")
            conversation = eng.createConversation(conversationConfig)

            Log.i(TAG, "Engine initialized successfully (GPU offload active)")
        }
    }

    override fun isInitialized(): Boolean = engine != null && conversation != null

    override suspend fun release() {
        inferenceMutex.withLock {
            Log.i(TAG, "Releasing engine resources")
            try {
                conversation?.close()
            } catch (e: Exception) {
                Log.w(TAG, "Error closing conversation: ${e.message}")
            }
            try {
                engine?.close()
            } catch (e: Exception) {
                Log.w(TAG, "Error closing engine: ${e.message}")
            }
            conversation = null
            engine = null
        }
    }

    // ── Inference ────────────────────────────────────────────────────

    override fun streamResponse(prompt: String): Flow<String> {
        val conv = conversation
            ?: throw IllegalStateException("Engine not initialized – call initialize() first")

        // Acquire the mutex before the native call begins and hold it for
        // the entire streaming duration. This guarantees that only one
        // inference request touches the JNI layer at a time.
        return kotlinx.coroutines.flow.flow {
            inferenceMutex.withLock {
                conv.sendMessageAsync(prompt)
                    .map { message -> message.toString() }
                    .catch { e ->
                        Log.e(TAG, "Streaming error: ${e.message}", e)
                        throw e
                    }
                    .collect { emit(it) }
            }
        }
    }
}
