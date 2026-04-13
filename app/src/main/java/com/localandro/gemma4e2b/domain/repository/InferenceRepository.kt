package com.localandro.gemma4e2b.domain.repository

import kotlinx.coroutines.flow.Flow

/**
 * Contract for the on-device LLM inference engine.
 *
 * Implementations are responsible for loading the model, running inference,
 * and streaming generated tokens back to the caller.
 */
interface InferenceRepository {

    /**
     * Loads the model from [modelPath] using the given [config].
     *
     * This operation may take several seconds (warm-up / GPU offload).
     * Must be called exactly once before [streamResponse].
     */
    suspend fun initialize(
        modelPath: String,
        config: InferenceConfig = InferenceConfig()
    ): Result<Unit>

    /**
     * Runs inference on [prompt] and returns a [Flow] that emits tokens
     * one at a time as they are generated.
     *
     * @throws IllegalStateException if [initialize] has not been called.
     */
    fun streamResponse(prompt: String): Flow<String>

    /** Returns `true` when the engine has been successfully initialized. */
    fun isInitialized(): Boolean

    /** Releases all native resources held by the engine. */
    suspend fun release()
}

/**
 * Configuration parameters for the inference engine.
 *
 * Default values target a balance between quality and speed on mid-range
 * Android devices (e.g. Snapdragon 7s Gen 2 / Adreno 710).
 */
data class InferenceConfig(
    val maxTokens: Int = 512,
    val contextSize: Int = 4096,
    val temperature: Float = 0.6f,
    val topK: Int = 40,
    val topP: Float = 0.9f,
    val useGpu: Boolean = true
)
