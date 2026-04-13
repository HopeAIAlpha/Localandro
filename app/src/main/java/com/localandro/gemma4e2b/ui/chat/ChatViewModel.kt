package com.localandro.gemma4e2b.ui.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.localandro.gemma4e2b.domain.model.Message
import com.localandro.gemma4e2b.domain.model.MessageRole
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Presentation-layer ViewModel for the chat screen.
 *
 * Responsibilities:
 * 1. Initialize the [InferenceRepository] on creation (GPU warm-up).
 * 2. Accept user messages and stream model responses token-by-token.
 * 3. Expose a [ChatUiState] for the Compose UI to observe.
 *
 * @param inferenceRepository The on-device LLM engine abstraction.
 * @param modelPath Absolute path to the downloaded model file.
 */
class ChatViewModel(
    private val inferenceRepository: InferenceRepository,
    private val modelPath: String
) : ViewModel() {

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    init {
        initializeEngine()
    }

    // ── Engine initialization ───────────────────────────────────────

    private fun initializeEngine() {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.update { it.copy(inferenceState = InferenceState.INITIALIZING) }

            inferenceRepository.initialize(modelPath)
                .onSuccess {
                    _uiState.update { it.copy(inferenceState = InferenceState.IDLE) }
                }
                .onFailure { error ->
                    _uiState.update {
                        it.copy(
                            inferenceState = InferenceState.ERROR,
                            errorMessage = error.localizedMessage
                                ?: "Error al inicializar el motor de IA"
                        )
                    }
                }
        }
    }

    // ── Public actions ──────────────────────────────────────────────

    /**
     * Sends a user message and begins streaming the model's response.
     * No-op if the engine is not idle.
     */
    fun sendMessage(text: String) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return
        if (_uiState.value.inferenceState != InferenceState.IDLE) return

        val userMessage = Message(role = MessageRole.USER, content = trimmed)

        _uiState.update {
            it.copy(
                messages = it.messages + userMessage,
                inferenceState = InferenceState.GENERATING,
                streamBuffer = ""
            )
        }

        viewModelScope.launch {
            val fullResponse = StringBuilder()

            inferenceRepository.streamResponse(trimmed)
                .flowOn(Dispatchers.IO)
                .catch { error ->
                    _uiState.update {
                        it.copy(
                            inferenceState = InferenceState.ERROR,
                            errorMessage = error.localizedMessage
                                ?: "Error durante la inferencia"
                        )
                    }
                }
                .onCompletion {
                    // Only finalize if no error occurred.
                    if (_uiState.value.inferenceState == InferenceState.GENERATING) {
                        val modelMessage = Message(
                            role = MessageRole.MODEL,
                            content = fullResponse.toString()
                        )
                        _uiState.update {
                            it.copy(
                                messages = it.messages + modelMessage,
                                inferenceState = InferenceState.IDLE,
                                streamBuffer = ""
                            )
                        }
                    }
                }
                .collect { token ->
                    fullResponse.append(token)
                    _uiState.update {
                        it.copy(streamBuffer = fullResponse.toString())
                    }
                }
        }
    }

    /** Retry engine initialization after an error. */
    fun retryInitialization() {
        _uiState.update {
            it.copy(
                inferenceState = InferenceState.INITIALIZING,
                errorMessage = null
            )
        }
        initializeEngine()
    }

    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch(Dispatchers.IO) {
            inferenceRepository.release()
        }
    }

    // ── Factory ─────────────────────────────────────────────────────

    /**
     * [ViewModelProvider.Factory] that injects the [InferenceRepository]
     * and [modelPath] into [ChatViewModel] without requiring Hilt.
     */
    class Factory(
        private val inferenceRepository: InferenceRepository,
        private val modelPath: String
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            require(modelClass.isAssignableFrom(ChatViewModel::class.java)) {
                "Unknown ViewModel class: ${modelClass.name}"
            }
            return ChatViewModel(inferenceRepository, modelPath) as T
        }
    }
}
