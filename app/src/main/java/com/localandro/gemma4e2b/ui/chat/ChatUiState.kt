package com.localandro.gemma4e2b.ui.chat

import com.localandro.gemma4e2b.domain.model.Message

/**
 * Immutable snapshot of the chat screen's UI state.
 *
 * Exposed by [ChatViewModel] as a `StateFlow` and collected by
 * [ChatScreen] via `collectAsStateWithLifecycle()`.
 */
data class ChatUiState(
    /** Full conversation history (user + model messages). */
    val messages: List<Message> = emptyList(),

    /** Current phase of the inference engine. */
    val inferenceState: InferenceState = InferenceState.INITIALIZING,

    /** Partial response being streamed from the model (shown in real time). */
    val streamBuffer: String = "",

    /** Human-readable error description, if any. */
    val errorMessage: String? = null
)

/** Represents the lifecycle phases of the inference engine and generation. */
enum class InferenceState {
    /** Engine is loading the model / performing GPU warm-up. */
    INITIALIZING,
    /** Engine is ready; user may send a message. */
    IDLE,
    /** Model is actively generating tokens. */
    GENERATING,
    /** An error occurred (see [ChatUiState.errorMessage]). */
    ERROR
}
