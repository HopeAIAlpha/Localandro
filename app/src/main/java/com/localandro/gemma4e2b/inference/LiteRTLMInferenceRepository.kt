package com.localandro.gemma4e2b.inference

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession.LlmInferenceSessionOptions
import com.localandro.gemma4e2b.domain.repository.InferenceConfig
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import java.io.File

/**
 * Implementation of [InferenceRepository] backed by the MediaPipe
 * LLM Inference engine (LiteRT-LM compatible).
 *
 * Uses [LlmInference] for model loading with GPU offload, and
 * [LlmInferenceSession] for stateful, streaming token generation
 * targeting the Gemma 4 E2B `.task` model.
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
    private var llmInference: LlmInference? = null

    @Volatile
    private var llmSession: LlmInferenceSession? = null

    private var config: InferenceConfig = InferenceConfig()

    // ── Lifecycle ────────────────────────────────────────────────────

    override suspend fun initialize(
        modelPath: String,
        config: InferenceConfig
    ): Result<Unit> = runCatching {
        val file = File(modelPath)
        require(file.exists()) { "Model file not found: $modelPath" }

        this.config = config

        // Build engine options targeting GPU backend (Adreno 710).
        val inferenceOptions = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxTokens(config.maxTokens)
            .setMaxTopK(config.topK)
            .build()

        Log.i(TAG, "Creating LlmInference engine from: $modelPath")
        llmInference = LlmInference.createFromOptions(context, inferenceOptions)

        // Create a session with sampling parameters.
        val sessionOptions = LlmInferenceSessionOptions.builder()
            .setTemperature(config.temperature)
            .setTopK(config.topK)
            .setTopP(config.topP)
            .build()

        Log.i(TAG, "Creating LlmInferenceSession (temp=${config.temperature}, topK=${config.topK})")
        llmSession = LlmInferenceSession.createFromOptions(llmInference!!, sessionOptions)

        Log.i(TAG, "Engine initialized successfully (GPU offload active)")
    }

    override fun isInitialized(): Boolean = llmInference != null && llmSession != null

    override suspend fun release() {
        Log.i(TAG, "Releasing engine resources")
        try {
            llmSession?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing session: ${e.message}")
        }
        try {
            llmInference?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing inference: ${e.message}")
        }
        llmSession = null
        llmInference = null
    }

    // ── Inference ────────────────────────────────────────────────────

    override fun streamResponse(prompt: String): Flow<String> = callbackFlow {
        val session = llmSession
            ?: throw IllegalStateException("Engine not initialized – call initialize() first")

        session.addQueryChunk(prompt)

        session.generateResponseAsync { partialResult, done ->
            if (partialResult.isNotEmpty()) {
                trySend(partialResult)
            }
            if (done) {
                close()
            }
        }

        awaitClose {
            // If the collector is cancelled, the future will complete on its own;
            // we don't force-cancel to avoid native crashes.
        }
    }
}
