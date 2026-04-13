package com.localandro.gemma4e2b.ui.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.localandro.gemma4e2b.agent.ActionOrchestrator
import com.localandro.gemma4e2b.agent.AgentPhase
import com.localandro.gemma4e2b.domain.model.Message
import com.localandro.gemma4e2b.domain.model.MessageRole
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Presentation-layer ViewModel for the chat screen.
 *
 * Responsibilities:
 * 1. Initialize the [InferenceRepository] on creation (GPU warm-up).
 * 2. Accept user messages and route them through the [ActionOrchestrator]
 *    for the Plan→Execute→Observe→Refine agentic loop.
 * 3. Expose a [ChatUiState] for the Compose UI to observe.
 *
 * @param inferenceRepository The on-device LLM engine abstraction.
 * @param modelPath Absolute path to the downloaded model file.
 * @param orchestrator The agentic action orchestrator.
 */
class ChatViewModel(
    private val inferenceRepository: InferenceRepository,
    private val modelPath: String,
    private val orchestrator: ActionOrchestrator
) : ViewModel() {

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    /** Exposes the agent FSM state for UI observation. */
    val agentState: StateFlow<com.localandro.gemma4e2b.agent.AgentState> = orchestrator.agentState

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
     * Sends a user message through the agentic orchestration loop.
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

        viewModelScope.launch(Dispatchers.IO) {
            val streamBuffer = StringBuilder()

            orchestrator.processUserMessage(
                userText = trimmed,
                onToken = { token ->
                    streamBuffer.append(token)
                    _uiState.update {
                        it.copy(streamBuffer = streamBuffer.toString())
                    }
                },
                onToolCall = { toolName ->
                    _uiState.update {
                        it.copy(
                            streamBuffer = streamBuffer.toString() +
                                    "\n🔧 Ejecutando: $toolName…"
                        )
                    }
                    // Reset stream buffer for next iteration.
                    streamBuffer.clear()
                },
                onComplete = { modelMessage ->
                    _uiState.update {
                        it.copy(
                            messages = it.messages + modelMessage,
                            inferenceState = InferenceState.IDLE,
                            streamBuffer = ""
                        )
                    }
                },
                onError = { errorMsg ->
                    _uiState.update {
                        it.copy(
                            inferenceState = InferenceState.ERROR,
                            errorMessage = errorMsg
                        )
                    }
                }
            )
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
        orchestrator.reset()
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
     * [ViewModelProvider.Factory] that injects dependencies into
     * [ChatViewModel] without requiring Hilt.
     */
    class Factory(
        private val inferenceRepository: InferenceRepository,
        private val modelPath: String,
        private val orchestrator: ActionOrchestrator
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            require(modelClass.isAssignableFrom(ChatViewModel::class.java)) {
                "Unknown ViewModel class: ${modelClass.name}"
            }
            return ChatViewModel(inferenceRepository, modelPath, orchestrator) as T
        }
    }
}
