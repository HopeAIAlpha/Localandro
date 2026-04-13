package com.localandro.gemma4e2b.inference

/**
 * Shared IPC message constants for communication between the main process
 * and the `:inference_engine` service process via Android [android.os.Messenger].
 *
 * ## Protocol overview
 *
 * | Direction        | What                 | Payload (Bundle keys)           |
 * |------------------|----------------------|---------------------------------|
 * | Client → Service | MSG_INITIALIZE       | MODEL_PATH, config fields       |
 * | Service → Client | MSG_INITIALIZE_RESULT| SUCCESS, ERROR (on failure)     |
 * | Client → Service | MSG_STREAM_REQUEST   | PROMPT                          |
 * | Service → Client | MSG_STREAM_TOKEN     | TOKEN                           |
 * | Service → Client | MSG_STREAM_COMPLETE  | *(empty)*                       |
 * | Service → Client | MSG_STREAM_ERROR     | ERROR                           |
 * | Client → Service | MSG_RELEASE          | *(empty)*                       |
 * | Client → Service | MSG_IS_INITIALIZED   | *(empty)*                       |
 * | Service → Client | MSG_IS_INITIALIZED_RESULT | RESULT (boolean)           |
 */
object InferenceIpcProtocol {

    // ── Client → Service messages ─────────────────────────────────────
    /** Request to initialise the LiteRT-LM engine with a model path + config. */
    const val MSG_INITIALIZE = 1
    /** Request to stream a response for the given prompt. */
    const val MSG_STREAM_REQUEST = 2
    /** Request to release all native resources held by the engine. */
    const val MSG_RELEASE = 3
    /** Query whether the engine has been initialised. */
    const val MSG_IS_INITIALIZED = 4

    // ── Service → Client messages ─────────────────────────────────────
    /** Result of an [MSG_INITIALIZE] request. */
    const val MSG_INITIALIZE_RESULT = 101
    /** A single streamed token during inference. */
    const val MSG_STREAM_TOKEN = 102
    /** Signals that streaming has completed successfully. */
    const val MSG_STREAM_COMPLETE = 103
    /** Signals that an error occurred during streaming. */
    const val MSG_STREAM_ERROR = 104
    /** Result of an [MSG_IS_INITIALIZED] query. */
    const val MSG_IS_INITIALIZED_RESULT = 105

    // ── Bundle keys ───────────────────────────────────────────────────
    const val KEY_MODEL_PATH = "model_path"
    const val KEY_PROMPT = "prompt"
    const val KEY_TOKEN = "token"
    const val KEY_ERROR = "error"
    const val KEY_SUCCESS = "success"
    const val KEY_RESULT = "result"

    // InferenceConfig fields
    const val KEY_MAX_TOKENS = "max_tokens"
    const val KEY_CONTEXT_SIZE = "context_size"
    const val KEY_TEMPERATURE = "temperature"
    const val KEY_TOP_K = "top_k"
    const val KEY_TOP_P = "top_p"
    const val KEY_USE_GPU = "use_gpu"
}
