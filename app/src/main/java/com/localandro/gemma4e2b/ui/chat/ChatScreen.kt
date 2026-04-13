package com.localandro.gemma4e2b.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.runtime.collectAsState
import androidx.lifecycle.viewmodel.compose.viewModel
import com.localandro.gemma4e2b.domain.model.Message
import com.localandro.gemma4e2b.domain.model.MessageRole
import com.localandro.gemma4e2b.domain.repository.InferenceRepository

/**
 * Full chat screen with real-time token streaming from Gemma 4 E2B.
 *
 * Shows a loading overlay while the LiteRT-LM engine warms up (GPU offload),
 * then presents a conversational UI where the user can type prompts and
 * see the model's response appear token by token.
 *
 * @param modelPath Absolute path to the downloaded model file inside filesDir.
 * @param inferenceRepository The inference engine to use.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    modelPath: String,
    inferenceRepository: InferenceRepository
) {
    val chatViewModel: ChatViewModel = viewModel(
        factory = ChatViewModel.Factory(inferenceRepository, modelPath)
    )
    val uiState by chatViewModel.uiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Gemma 4 E2B") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { innerPadding ->

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .imePadding()
        ) {
            when (uiState.inferenceState) {
                InferenceState.INITIALIZING -> {
                    InitializingOverlay()
                }

                InferenceState.ERROR -> {
                    ErrorOverlay(
                        message = uiState.errorMessage ?: "Error desconocido",
                        onRetry = { chatViewModel.retryInitialization() }
                    )
                }

                InferenceState.IDLE, InferenceState.GENERATING -> {
                    ChatContent(
                        uiState = uiState,
                        onSendMessage = { chatViewModel.sendMessage(it) },
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

// ── Initialization overlay ──────────────────────────────────────────

@Composable
private fun InitializingOverlay() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(64.dp),
            strokeWidth = 4.dp
        )

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Cargando motor de IA (Offload a GPU)…",
            style = MaterialTheme.typography.titleMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurface
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "Inicializando LiteRT-LM y compilando shaders.\nEsto puede tardar unos segundos.",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

// ── Error overlay ───────────────────────────────────────────────────

@Composable
private fun ErrorOverlay(message: String, onRetry: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Error de inicialización",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.error
        )

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(24.dp))

        TextButton(onClick = onRetry) {
            Text("Reintentar")
        }
    }
}

// ── Chat content (messages + input) ─────────────────────────────────

@Composable
private fun ChatContent(
    uiState: ChatUiState,
    onSendMessage: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val listState = rememberLazyListState()
    val isGenerating = uiState.inferenceState == InferenceState.GENERATING

    // Auto-scroll to bottom when new messages arrive or stream buffer updates.
    val totalItems = uiState.messages.size + (if (isGenerating) 1 else 0)
    LaunchedEffect(totalItems, uiState.streamBuffer) {
        if (totalItems > 0) {
            listState.animateScrollToItem(totalItems - 1)
        }
    }

    Column(modifier = modifier) {

        // ── Message list ────────────────────────────────────────────
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(uiState.messages, key = { it.id }) { message ->
                MessageBubble(message)
            }

            // Show streaming partial response as it arrives.
            if (isGenerating && uiState.streamBuffer.isNotEmpty()) {
                item(key = "streaming") {
                    StreamingBubble(text = uiState.streamBuffer)
                }
            }

            // Typing indicator when waiting for first token.
            if (isGenerating && uiState.streamBuffer.isEmpty()) {
                item(key = "typing") {
                    TypingIndicator()
                }
            }
        }

        // ── Input bar ───────────────────────────────────────────────
        ChatInputBar(
            enabled = !isGenerating,
            onSend = onSendMessage
        )
    }
}

// ── Message bubble ──────────────────────────────────────────────────

@Composable
private fun MessageBubble(message: Message) {
    val isUser = message.role == MessageRole.USER

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .clip(
                    RoundedCornerShape(
                        topStart = 16.dp,
                        topEnd = 16.dp,
                        bottomStart = if (isUser) 16.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 16.dp
                    )
                )
                .background(
                    if (isUser) MaterialTheme.colorScheme.primaryContainer
                    else MaterialTheme.colorScheme.secondaryContainer
                )
                .padding(horizontal = 14.dp, vertical = 10.dp)
        ) {
            Text(
                text = message.content,
                style = MaterialTheme.typography.bodyLarge,
                color = if (isUser) MaterialTheme.colorScheme.onPrimaryContainer
                else MaterialTheme.colorScheme.onSecondaryContainer
            )
        }
    }
}

// ── Streaming bubble (partial response) ─────────────────────────────

@Composable
private fun StreamingBubble(text: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .clip(RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp, bottomEnd = 16.dp, bottomStart = 4.dp))
                .background(MaterialTheme.colorScheme.secondaryContainer)
                .padding(horizontal = 14.dp, vertical = 10.dp)
        ) {
            Text(
                text = text,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
        }
    }
}

// ── Typing indicator ────────────────────────────────────────────────

@Composable
private fun TypingIndicator() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(16.dp))
                .background(MaterialTheme.colorScheme.secondaryContainer)
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            Text(
                text = "Generando…",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
        }
    }
}

// ── Input bar ───────────────────────────────────────────────────────

@Composable
private fun ChatInputBar(enabled: Boolean, onSend: (String) -> Unit) {
    var inputText by rememberSaveable { mutableStateOf("") }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        OutlinedTextField(
            value = inputText,
            onValueChange = { inputText = it },
            modifier = Modifier.weight(1f),
            placeholder = { Text("Escribe un mensaje…") },
            enabled = enabled,
            maxLines = 4,
            shape = RoundedCornerShape(24.dp)
        )

        IconButton(
            onClick = {
                if (inputText.isNotBlank()) {
                    onSend(inputText)
                    inputText = ""
                }
            },
            enabled = enabled && inputText.isNotBlank()
        ) {
            Text(
                text = "➤",
                style = MaterialTheme.typography.titleLarge,
                color = if (enabled && inputText.isNotBlank())
                    MaterialTheme.colorScheme.primary
                else
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
            )
        }
    }
}
