package com.localandro.gemma4e2b.inference

import com.localandro.gemma4e2b.domain.repository.InferenceConfig
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File

/**
 * Implementation of [InferenceRepository] backed by the LiteRT-LM engine.
 *
 * **Current status:** stub implementation.  The full LiteRT-LM native library
 * (`com.google.ai.edge.litert:litert-lm`) has not yet been added to the
 * project dependencies.  Once added, replace the stub bodies below with
 * calls to `LlmInference.createFromOptions(…)` and
 * `LlmInference.generateResponseAsync(…)`.
 *
 * The stub simulates:
 * - A warm-up delay during [initialize] (GPU offload time).
 * - Token-by-token streaming in [streamResponse] so the UI pipeline can be
 *   validated end-to-end before the native engine is integrated.
 */
class LiteRTLMInferenceRepository : InferenceRepository {

    @Volatile
    private var initialized = false

    @Volatile
    private var modelPath: String? = null

    private var config: InferenceConfig = InferenceConfig()

    // ── Lifecycle ────────────────────────────────────────────────────

    override suspend fun initialize(
        modelPath: String,
        config: InferenceConfig
    ): Result<Unit> = runCatching {
        val file = File(modelPath)
        require(file.exists()) { "Model file not found: $modelPath" }

        this.config = config
        this.modelPath = modelPath

        // TODO: Replace with real LiteRT-LM initialization:
        //   val options = LlmInference.LlmInferenceOptions.builder()
        //       .setModelPath(modelPath)
        //       .setMaxTokens(config.maxTokens)
        //       .setPreferredBackend(if (config.useGpu) Backend.GPU else Backend.CPU)
        //       .build()
        //   llmInference = LlmInference.createFromOptions(context, options)

        // Simulate warm-up / GPU shader compilation (~2 s).
        delay(2_000)

        initialized = true
    }

    override fun isInitialized(): Boolean = initialized

    override suspend fun release() {
        // TODO: llmInference?.close()
        initialized = false
        modelPath = null
    }

    // ── Inference ────────────────────────────────────────────────────

    override fun streamResponse(prompt: String): Flow<String> = flow {
        check(initialized) { "Engine not initialized – call initialize() first" }

        // TODO: Replace with real LiteRT-LM streaming inference:
        //   llmInference.generateResponseAsync(prompt).collect { partial ->
        //       emit(partial)
        //   }

        // Stub: emit a helpful placeholder response token-by-token.
        val response = buildStubResponse(prompt)
        for (token in tokenize(response)) {
            delay(30) // simulate ~30 tokens/s generation speed
            emit(token)
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────

    /**
     * Produces a stub response so the streaming UI pipeline can be verified.
     */
    private fun buildStubResponse(prompt: String): String {
        return "¡Hola! Soy Gemma 4 E2B ejecutándose localmente en tu dispositivo. " +
                "El motor de inferencia LiteRT-LM aún no está conectado — " +
                "esta es una respuesta de prueba para validar el pipeline de streaming. " +
                "Tu mensaje fue: \"${prompt.take(120)}\""
    }

    /**
     * Splits text into word-level tokens to simulate streaming output.
     */
    private fun tokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val words = text.split(" ")
        for ((index, word) in words.withIndex()) {
            tokens.add(if (index == 0) word else " $word")
        }
        return tokens
    }
}
