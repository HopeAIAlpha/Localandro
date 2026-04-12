# Documento de Diseño y Blueprint Técnico: Gemma 4 E2B Android App

## Especificaciones de Referencia del Hardware

| Componente | Valor |
|---|---|
| Dispositivo | Redmi Note 13 Pro |
| SoC | Qualcomm Snapdragon 7s Gen 2 (SM7435, Samsung 4nm) |
| CPU | 4× Cortex-A78 @ 2.4 GHz + 4× Cortex-A55 @ 1.95 GHz |
| GPU | Adreno 710 (Vulkan 1.1, OpenCL 2.0) |
| NPU | Hexagon 710 |
| RAM física | 8 GB LPDDR5 @ 3200 MHz |
| RAM virtual (HyperOS) | 8 GB swap en UFS 2.2 |
| Almacenamiento disponible | 82.8 GB |
| Android | 16 (compilación BP2A.250605.031.A3) |
| Modelo IA | Gemma 4 E2B (~2.3B params efectivos, INT4, ~2.6 GB) |
| Framework de inferencia | LiteRT-LM (sucesor de MediaPipe LLM Inference, deprecado) |
| TTFT objetivo | < 1000 ms tras warm-up |
| Throughput objetivo | 12-25 tokens/s |
| Contexto operativo | 4096 tokens (máximo configurable: 8192) |

---

## 1. Arquitectura de la Aplicación (MVVM + Clean Architecture)

### 1.1 Diagrama de Capas

```
┌──────────────────────────────────────────────────────────────────┐
│                        UI Layer (Compose)                        │
│  ChatScreen · ToolConfirmationDialog · ThermalStatusBanner       │
│  Observa: viewModel.uiState.collectAsStateWithLifecycle()        │
├──────────────────────────────────────────────────────────────────┤
│                    Presentation Layer                             │
│  ChatViewModel · ThermalViewModel                                │
│  Expone: StateFlow<ChatUiState> · StateFlow<ThermalState>        │
│  Consume: InferenceUseCase · ToolOrchestrationUseCase            │
├──────────────────────────────────────────────────────────────────┤
│                      Domain Layer                                │
│  Entities: Message · ToolDefinition · ToolCall · ToolResult      │
│  UseCases: InferenceUseCase · ToolOrchestrationUseCase           │
│            FileAccessUseCase · ThermalPolicyUseCase               │
│  Interfaces: InferenceRepository · FileRepository                │
│              ThermalMonitor · ToolRegistry                        │
├──────────────────────────────────────────────────────────────────┤
│                       Data Layer                                 │
│  LiteRTLMInferenceRepository (implementa InferenceRepository)    │
│  ScopedFileRepository (implementa FileRepository)                │
│  AndroidThermalMonitor (implementa ThermalMonitor)               │
│  ToolRegistryImpl (implementa ToolRegistry)                      │
├──────────────────────────────────────────────────────────────────┤
│                   Framework / External                            │
│  LiteRT-LM Engine · Android SAF/DocumentFile                     │
│  PowerManager · ContentResolver · SensorManager                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Entidades del Domain Layer

```kotlin
// domain/model/Message.kt
data class Message(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val toolCall: ToolCall? = null,
    val toolResult: ToolResult? = null
)

enum class MessageRole { USER, MODEL, SYSTEM, TOOL_RESULT }

// domain/model/ToolDefinition.kt
data class ToolDefinition(
    val name: String,
    val description: String,
    val parameters: JsonSchema
)

data class JsonSchema(
    val type: String = "object",
    val properties: Map<String, PropertySchema>,
    val required: List<String> = emptyList()
)

data class PropertySchema(
    val type: String,
    val description: String,
    val enum: List<String>? = null
)

// domain/model/ToolCall.kt
data class ToolCall(
    val name: String,
    val arguments: Map<String, Any>
)

// domain/model/ToolResult.kt
data class ToolResult(
    val name: String,
    val output: Map<String, Any>,
    val success: Boolean = true,
    val errorMessage: String? = null
)
```

### 1.3 Interfaces del Domain Layer

```kotlin
// domain/repository/InferenceRepository.kt
interface InferenceRepository {
    suspend fun initialize(modelPath: String, config: InferenceConfig): Result<Unit>
    fun streamResponse(prompt: String): Flow<String>
    fun isInitialized(): Boolean
    suspend fun release()
}

data class InferenceConfig(
    val maxTokens: Int = 512,
    val contextSize: Int = 4096,
    val temperature: Float = 0.6f,
    val topK: Int = 40,
    val topP: Float = 0.9f,
    val useGpu: Boolean = true
)

// domain/repository/FileRepository.kt
interface FileRepository {
    suspend fun readFileContent(uri: Uri): Result<String>
    suspend fun listDirectory(uri: Uri): Result<List<FileEntry>>
    suspend fun getFileMetadata(uri: Uri): Result<FileMetadata>
    suspend fun writeFile(uri: Uri, content: String): Result<Unit>
}

data class FileEntry(
    val name: String,
    val uri: Uri,
    val mimeType: String,
    val sizeBytes: Long,
    val isDirectory: Boolean
)

data class FileMetadata(
    val name: String,
    val sizeBytes: Long,
    val mimeType: String,
    val lastModified: Long
)

// domain/monitor/ThermalMonitor.kt
interface ThermalMonitor {
    val thermalState: StateFlow<ThermalState>
    fun startMonitoring()
    fun stopMonitoring()
}
```

### 1.4 ChatViewModel — Gestión de Estados de Inferencia Asíncrona

```kotlin
// presentation/viewmodel/ChatViewModel.kt
@HiltViewModel
class ChatViewModel @Inject constructor(
    private val inferenceUseCase: InferenceUseCase,
    private val toolOrchestrationUseCase: ToolOrchestrationUseCase,
    private val thermalPolicyUseCase: ThermalPolicyUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    // Colecta cambios térmicos para ajustar parámetros de inferencia dinámicamente
    init {
        viewModelScope.launch {
            thermalPolicyUseCase.activePolicyFlow.collect { policy ->
                _uiState.update { it.copy(activePolicy = policy) }
            }
        }
    }

    fun sendMessage(userInput: String) {
        if (_uiState.value.inferenceState == InferenceState.GENERATING) return

        val userMessage = Message(role = MessageRole.USER, content = userInput)
        _uiState.update {
            it.copy(
                messages = it.messages + userMessage,
                inferenceState = InferenceState.GENERATING,
                streamBuffer = ""
            )
        }

        viewModelScope.launch {
            val policy = _uiState.value.activePolicy
            val prompt = buildPromptWithToolDefinitions(
                messages = _uiState.value.messages,
                tools = toolOrchestrationUseCase.getRegisteredTools()
            )

            val fullResponse = StringBuilder()

            inferenceUseCase.streamInference(prompt, policy.toInferenceConfig())
                .onEach { token ->
                    fullResponse.append(token)
                    _uiState.update { it.copy(streamBuffer = fullResponse.toString()) }
                }
                .catch { error ->
                    _uiState.update {
                        it.copy(
                            inferenceState = InferenceState.ERROR,
                            errorMessage = error.message
                        )
                    }
                }
                .onCompletion {
                    val responseText = fullResponse.toString()
                    handleModelResponse(responseText)
                }
                .collect()
        }
    }

    private suspend fun handleModelResponse(responseText: String) {
        val parsedToolCall = toolOrchestrationUseCase.parseToolCall(responseText)

        if (parsedToolCall != null) {
            // El modelo solicitó ejecutar una herramienta
            _uiState.update {
                it.copy(
                    inferenceState = InferenceState.AWAITING_TOOL_CONFIRMATION,
                    pendingToolCall = parsedToolCall
                )
            }
        } else {
            // Respuesta directa sin tool calling
            val modelMessage = Message(role = MessageRole.MODEL, content = responseText)
            _uiState.update {
                it.copy(
                    messages = it.messages + modelMessage,
                    inferenceState = InferenceState.IDLE,
                    streamBuffer = ""
                )
            }
        }
    }

    fun confirmToolExecution() {
        val toolCall = _uiState.value.pendingToolCall ?: return
        viewModelScope.launch {
            _uiState.update { it.copy(inferenceState = InferenceState.EXECUTING_TOOL) }

            val result = toolOrchestrationUseCase.executeTool(toolCall)

            // Inyectar resultado en el contexto y continuar generación
            val toolResultMessage = Message(
                role = MessageRole.TOOL_RESULT,
                content = result.toPromptFormat(),
                toolResult = result
            )
            _uiState.update {
                it.copy(
                    messages = it.messages + toolResultMessage,
                    pendingToolCall = null
                )
            }
            // Re-enviar al modelo con el resultado de la herramienta
            continueAfterToolResult()
        }
    }

    fun rejectToolExecution() {
        _uiState.update {
            it.copy(
                inferenceState = InferenceState.IDLE,
                pendingToolCall = null
            )
        }
    }
}

// presentation/state/ChatUiState.kt
data class ChatUiState(
    val messages: List<Message> = emptyList(),
    val inferenceState: InferenceState = InferenceState.IDLE,
    val streamBuffer: String = "",
    val pendingToolCall: ToolCall? = null,
    val activePolicy: ThermalPolicy = ThermalPolicy.NORMAL,
    val errorMessage: String? = null
)

enum class InferenceState {
    IDLE,
    GENERATING,
    AWAITING_TOOL_CONFIRMATION,
    EXECUTING_TOOL,
    ERROR
}
```

**Flujo de estados del ViewModel:**

```
IDLE → [usuario envía mensaje] → GENERATING
GENERATING → [modelo responde sin tool_call] → IDLE
GENERATING → [modelo emite <|tool_call>] → AWAITING_TOOL_CONFIRMATION
AWAITING_TOOL_CONFIRMATION → [usuario confirma] → EXECUTING_TOOL
AWAITING_TOOL_CONFIRMATION → [usuario rechaza] → IDLE
EXECUTING_TOOL → [resultado inyectado] → GENERATING → IDLE
GENERATING → [error de inferencia] → ERROR
ERROR → [usuario reintenta] → IDLE
```

---

## 2. Implementación del Framework de Herramientas (Tool Calling)

### 2.1 Protocolo de Tokens Especiales de Gemma 4

Gemma 4 E2B utiliza 3 pares de tokens delimitadores:

| Token apertura | Token cierre | Emisor | Propósito |
|---|---|---|---|
| `<\|tool>` | `<tool\|>` | Sistema/App | Declarar herramienta disponible (JSON Schema) |
| `<\|tool_call>` | `<tool_call\|>` | Modelo | Solicitar ejecución de herramienta con argumentos |
| `<\|tool_result>` | `<tool_result\|>` | Sistema/App | Retornar resultado de ejecución al modelo |

### 2.2 Construcción del Prompt con Herramientas

```kotlin
// domain/usecase/ToolOrchestrationUseCase.kt
class ToolOrchestrationUseCase @Inject constructor(
    private val toolRegistry: ToolRegistry,
    private val fileRepository: FileRepository
) {

    fun getRegisteredTools(): List<ToolDefinition> = toolRegistry.getAllTools()

    fun buildSystemPromptWithTools(): String {
        val toolDeclarations = toolRegistry.getAllTools().joinToString("\n") { tool ->
            """
<|tool>
{
  "name": "${tool.name}",
  "description": "${tool.description}",
  "parameters": ${tool.parameters.toJson()}
}
<tool|>""".trimIndent()
        }

        return """<start_of_turn>system
Eres un asistente local ejecutándose en un dispositivo Android. Tienes acceso a las siguientes herramientas para interactuar con el sistema de archivos y sensores del dispositivo. Responde en español. Si necesitas información de un archivo o sensor, usa la herramienta correspondiente. No ejecutes herramientas destructivas sin que el usuario lo solicite explícitamente.

$toolDeclarations
<end_of_turn>"""
    }

    fun parseToolCall(modelOutput: String): ToolCall? {
        val regex = Regex(
            """<\|tool_call>\s*(\{.*?\})\s*<\|tool_call\|>""",
            RegexOption.DOT_MATCHES_ALL
        )
        val match = regex.find(modelOutput) ?: return null
        return try {
            val json = JSONObject(match.groupValues[1])
            ToolCall(
                name = json.getString("name"),
                arguments = json.getJSONObject("arguments").toMap()
            )
        } catch (e: JSONException) {
            null
        }
    }

    suspend fun executeTool(toolCall: ToolCall): ToolResult {
        val handler = toolRegistry.getHandler(toolCall.name)
            ?: return ToolResult(
                name = toolCall.name,
                output = mapOf("error" to "Herramienta no registrada: ${toolCall.name}"),
                success = false
            )

        return try {
            handler.execute(toolCall.arguments)
        } catch (e: SecurityException) {
            ToolResult(
                name = toolCall.name,
                output = mapOf("error" to "Permiso denegado: ${e.message}"),
                success = false,
                errorMessage = e.message
            )
        } catch (e: Exception) {
            ToolResult(
                name = toolCall.name,
                output = mapOf("error" to "Error de ejecución: ${e.message}"),
                success = false,
                errorMessage = e.message
            )
        }
    }

    fun formatToolResult(result: ToolResult): String {
        val outputJson = JSONObject(result.output).toString(2)
        return """<|tool_result>
{
  "name": "${result.name}",
  "output": $outputJson
}
<tool_result|>"""
    }
}
```

### 2.3 ToolRegistry — Interfaz Escalable

```kotlin
// domain/registry/ToolRegistry.kt
interface ToolRegistry {
    fun register(tool: ToolDefinition, handler: ToolHandler)
    fun unregister(toolName: String)
    fun getHandler(toolName: String): ToolHandler?
    fun getAllTools(): List<ToolDefinition>
    fun getToolsForCategory(category: ToolCategory): List<ToolDefinition>
}

interface ToolHandler {
    val category: ToolCategory
    val isDestructive: Boolean
    suspend fun execute(arguments: Map<String, Any>): ToolResult
}

enum class ToolCategory {
    FILE_SYSTEM,
    DEVICE_SENSORS,
    SYSTEM_INFO,
    NETWORK
}

// data/registry/ToolRegistryImpl.kt
@Singleton
class ToolRegistryImpl @Inject constructor() : ToolRegistry {

    private val tools = ConcurrentHashMap<String, Pair<ToolDefinition, ToolHandler>>()

    override fun register(tool: ToolDefinition, handler: ToolHandler) {
        tools[tool.name] = tool to handler
    }

    override fun unregister(toolName: String) {
        tools.remove(toolName)
    }

    override fun getHandler(toolName: String): ToolHandler? = tools[toolName]?.second

    override fun getAllTools(): List<ToolDefinition> = tools.values.map { it.first }

    override fun getToolsForCategory(category: ToolCategory): List<ToolDefinition> =
        tools.values.filter { it.second.category == category }.map { it.first }
}
```

### 2.4 Herramientas Pre-Registradas

```kotlin
// data/tools/FileReadHandler.kt
class FileReadHandler @Inject constructor(
    private val fileRepository: FileRepository
) : ToolHandler {

    override val category = ToolCategory.FILE_SYSTEM
    override val isDestructive = false

    override suspend fun execute(arguments: Map<String, Any>): ToolResult {
        val uriString = arguments["uri"] as? String
            ?: return ToolResult("read_file", mapOf("error" to "Parámetro 'uri' requerido"), false)
        val uri = Uri.parse(uriString)
        val maxChars = (arguments["max_chars"] as? Number)?.toInt() ?: 8000

        return fileRepository.readFileContent(uri).fold(
            onSuccess = { content ->
                val truncated = if (content.length > maxChars) {
                    content.take(maxChars) + "\n[...truncado a $maxChars caracteres]"
                } else content
                ToolResult("read_file", mapOf("content" to truncated, "chars" to truncated.length))
            },
            onFailure = { e ->
                ToolResult("read_file", mapOf("error" to (e.message ?: "Error desconocido")), false)
            }
        )
    }

    companion object {
        val DEFINITION = ToolDefinition(
            name = "read_file",
            description = "Lee el contenido de texto de un archivo del dispositivo. Retorna el texto truncado a max_chars.",
            parameters = JsonSchema(
                properties = mapOf(
                    "uri" to PropertySchema("string", "URI del archivo (content:// o file://)"),
                    "max_chars" to PropertySchema("integer", "Límite de caracteres a retornar. Default: 8000")
                ),
                required = listOf("uri")
            )
        )
    }
}

// data/tools/ListDirectoryHandler.kt
class ListDirectoryHandler @Inject constructor(
    private val fileRepository: FileRepository
) : ToolHandler {

    override val category = ToolCategory.FILE_SYSTEM
    override val isDestructive = false

    override suspend fun execute(arguments: Map<String, Any>): ToolResult {
        val uriString = arguments["uri"] as? String
            ?: return ToolResult("list_directory", mapOf("error" to "Parámetro 'uri' requerido"), false)
        val uri = Uri.parse(uriString)

        return fileRepository.listDirectory(uri).fold(
            onSuccess = { entries ->
                val listing = entries.map { entry ->
                    mapOf(
                        "name" to entry.name,
                        "type" to if (entry.isDirectory) "directory" else "file",
                        "size_bytes" to entry.sizeBytes,
                        "mime_type" to entry.mimeType
                    )
                }
                ToolResult("list_directory", mapOf("entries" to listing, "count" to entries.size))
            },
            onFailure = { e ->
                ToolResult("list_directory", mapOf("error" to (e.message ?: "Error desconocido")), false)
            }
        )
    }

    companion object {
        val DEFINITION = ToolDefinition(
            name = "list_directory",
            description = "Lista archivos y subdirectorios dentro de un directorio del dispositivo.",
            parameters = JsonSchema(
                properties = mapOf(
                    "uri" to PropertySchema("string", "URI del directorio (content:// scheme)")
                ),
                required = listOf("uri")
            )
        )
    }
}

// data/tools/DeviceSensorHandler.kt
class DeviceSensorHandler @Inject constructor(
    @ApplicationContext private val context: Context
) : ToolHandler {

    override val category = ToolCategory.DEVICE_SENSORS
    override val isDestructive = false

    override suspend fun execute(arguments: Map<String, Any>): ToolResult {
        val sensorType = arguments["sensor"] as? String
            ?: return ToolResult("read_sensor", mapOf("error" to "Parámetro 'sensor' requerido"), false)

        return when (sensorType) {
            "battery" -> readBatteryInfo()
            "storage" -> readStorageInfo()
            "memory" -> readMemoryInfo()
            else -> ToolResult("read_sensor", mapOf("error" to "Sensor no soportado: $sensorType"), false)
        }
    }

    private fun readBatteryInfo(): ToolResult {
        val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val isCharging = batteryManager.isCharging
        val temperature = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)

        return ToolResult("read_sensor", mapOf(
            "sensor" to "battery",
            "level_percent" to level,
            "is_charging" to isCharging
        ))
    }

    private fun readStorageInfo(): ToolResult {
        val stat = StatFs(Environment.getDataDirectory().path)
        val availableBytes = stat.availableBytes
        val totalBytes = stat.totalBytes

        return ToolResult("read_sensor", mapOf(
            "sensor" to "storage",
            "available_gb" to String.format("%.1f", availableBytes / 1_073_741_824.0),
            "total_gb" to String.format("%.1f", totalBytes / 1_073_741_824.0)
        ))
    }

    private fun readMemoryInfo(): ToolResult {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        return ToolResult("read_sensor", mapOf(
            "sensor" to "memory",
            "available_mb" to (memInfo.availMem / 1_048_576),
            "total_mb" to (memInfo.totalMem / 1_048_576),
            "is_low_memory" to memInfo.lowMemory,
            "threshold_mb" to (memInfo.threshold / 1_048_576)
        ))
    }

    companion object {
        val DEFINITION = ToolDefinition(
            name = "read_sensor",
            description = "Lee información de sensores y estado del dispositivo: batería, almacenamiento, memoria RAM.",
            parameters = JsonSchema(
                properties = mapOf(
                    "sensor" to PropertySchema(
                        "string",
                        "Tipo de sensor a leer",
                        enum = listOf("battery", "storage", "memory")
                    )
                ),
                required = listOf("sensor")
            )
        )
    }
}

// data/tools/WriteFileHandler.kt
class WriteFileHandler @Inject constructor(
    private val fileRepository: FileRepository
) : ToolHandler {

    override val category = ToolCategory.FILE_SYSTEM
    override val isDestructive = true  // Requiere confirmación del usuario

    override suspend fun execute(arguments: Map<String, Any>): ToolResult {
        val uriString = arguments["uri"] as? String
            ?: return ToolResult("write_file", mapOf("error" to "Parámetro 'uri' requerido"), false)
        val content = arguments["content"] as? String
            ?: return ToolResult("write_file", mapOf("error" to "Parámetro 'content' requerido"), false)

        val uri = Uri.parse(uriString)
        return fileRepository.writeFile(uri, content).fold(
            onSuccess = {
                ToolResult("write_file", mapOf(
                    "status" to "written",
                    "uri" to uriString,
                    "bytes_written" to content.toByteArray(Charsets.UTF_8).size
                ))
            },
            onFailure = { e ->
                ToolResult("write_file", mapOf("error" to (e.message ?: "Error de escritura")), false)
            }
        )
    }

    companion object {
        val DEFINITION = ToolDefinition(
            name = "write_file",
            description = "Escribe contenido de texto en un archivo. ACCIÓN DESTRUCTIVA: requiere confirmación del usuario.",
            parameters = JsonSchema(
                properties = mapOf(
                    "uri" to PropertySchema("string", "URI destino del archivo"),
                    "content" to PropertySchema("string", "Contenido de texto a escribir")
                ),
                required = listOf("uri", "content")
            )
        )
    }
}
```

### 2.5 Inicialización del Registro en el Módulo DI

```kotlin
// di/ToolModule.kt
@Module
@InstallIn(SingletonComponent::class)
object ToolModule {

    @Provides
    @Singleton
    fun provideToolRegistry(
        fileReadHandler: FileReadHandler,
        listDirectoryHandler: ListDirectoryHandler,
        deviceSensorHandler: DeviceSensorHandler,
        writeFileHandler: WriteFileHandler
    ): ToolRegistry {
        val registry = ToolRegistryImpl()
        registry.register(FileReadHandler.DEFINITION, fileReadHandler)
        registry.register(ListDirectoryHandler.DEFINITION, listDirectoryHandler)
        registry.register(DeviceSensorHandler.DEFINITION, deviceSensorHandler)
        registry.register(WriteFileHandler.DEFINITION, writeFileHandler)
        return registry
    }
}
```

### 2.6 Principio "Sistema sobre Usuario" — Validación de Herramientas Destructivas

El `ChatViewModel` implementa una barrera de confirmación antes de ejecutar cualquier herramienta marcada como `isDestructive = true`:

```
Flujo de validación:
1. Modelo emite <|tool_call> → parseToolCall() extrae ToolCall
2. toolRegistry.getHandler(toolCall.name).isDestructive → true/false
3. Si isDestructive == true:
   a. Estado → AWAITING_TOOL_CONFIRMATION
   b. UI renderiza diálogo de confirmación con:
      - Nombre de herramienta
      - Argumentos (URI, contenido truncado)
      - Botón "Ejecutar" / Botón "Cancelar"
   c. Solo si usuario confirma → executeTool()
4. Si isDestructive == false:
   a. Ejecutar directamente (sin diálogo)
   b. Inyectar <|tool_result> al contexto
```

---

## 3. Sistema de Gestión de Archivos y Permisos

### 3.1 Estrategia de Acceso: Android 16 Scoped Storage

Android 16 restringe el acceso a archivos externos. La aplicación opera bajo las siguientes reglas:

| Ubicación | API de acceso | Permisos requeridos |
|---|---|---|
| Directorio privado de la app (`/data/data/pkg/`) | `File` API directa | Ninguno |
| Directorio externo privado (`Android/data/pkg/`) | `File` API directa | Ninguno |
| Modelo LLM (descargado) | Almacenado en directorio privado | Ninguno |
| Documentos del usuario (PDFs, TXT, etc.) | SAF via `Intent.ACTION_OPEN_DOCUMENT` | Selección explícita del usuario |
| Directorios del usuario | SAF via `Intent.ACTION_OPEN_DOCUMENT_TREE` | Selección explícita del usuario + persistencia |
| Media (fotos, audio, video) | `MediaStore` API | `READ_MEDIA_IMAGES/AUDIO/VIDEO` |

### 3.2 Implementación del FileRepository con SAF

```kotlin
// data/repository/ScopedFileRepository.kt
class ScopedFileRepository @Inject constructor(
    @ApplicationContext private val context: Context
) : FileRepository {

    override suspend fun readFileContent(uri: Uri): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).readText()
            } ?: throw IOException("No se pudo abrir el stream para: $uri")
        }
    }

    override suspend fun listDirectory(uri: Uri): Result<List<FileEntry>> = withContext(Dispatchers.IO) {
        runCatching {
            val documentFile = DocumentFile.fromTreeUri(context, uri)
                ?: throw IOException("URI de directorio no válida: $uri")

            documentFile.listFiles().map { file ->
                FileEntry(
                    name = file.name ?: "sin_nombre",
                    uri = file.uri,
                    mimeType = file.type ?: "application/octet-stream",
                    sizeBytes = file.length(),
                    isDirectory = file.isDirectory
                )
            }
        }
    }

    override suspend fun getFileMetadata(uri: Uri): Result<FileMetadata> = withContext(Dispatchers.IO) {
        runCatching {
            val documentFile = DocumentFile.fromSingleUri(context, uri)
                ?: throw IOException("URI no válida: $uri")

            FileMetadata(
                name = documentFile.name ?: "sin_nombre",
                sizeBytes = documentFile.length(),
                mimeType = documentFile.type ?: "application/octet-stream",
                lastModified = documentFile.lastModified()
            )
        }
    }

    override suspend fun writeFile(uri: Uri, content: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            context.contentResolver.openOutputStream(uri, "wt")?.use { stream ->
                stream.write(content.toByteArray(Charsets.UTF_8))
            } ?: throw IOException("No se pudo abrir stream de escritura para: $uri")
        }
    }
}
```

### 3.3 Sandbox de Acción — Lectura de Documentos para Contexto

La Sandbox de Acción permite al modelo leer documentos seleccionados por el usuario para enriquecer el contexto de sus respuestas sin exponer el sistema de archivos completo.

```kotlin
// presentation/viewmodel/ChatViewModel.kt (extensión)

// El usuario selecciona un documento vía SAF picker desde la UI
fun onDocumentSelected(uri: Uri) {
    // Persistir permiso para acceso futuro dentro de la sesión
    context.contentResolver.takePersistableUriPermission(
        uri,
        Intent.FLAG_GRANT_READ_URI_PERMISSION
    )

    viewModelScope.launch {
        _uiState.update { it.copy(inferenceState = InferenceState.GENERATING) }

        val metadata = fileRepository.getFileMetadata(uri).getOrNull()
        val content = fileRepository.readFileContent(uri).getOrNull()

        if (content != null && metadata != null) {
            // Truncar a presupuesto de tokens seguro
            // 1 token ≈ 4 caracteres en español → 4096 tokens ≈ 16384 caracteres
            // Reservar 2048 tokens para sistema + herramientas + respuesta
            // Presupuesto para documento: 2048 tokens ≈ 8192 caracteres
            val maxDocChars = 8192
            val truncatedContent = if (content.length > maxDocChars) {
                content.take(maxDocChars) + "\n[...documento truncado a $maxDocChars caracteres]"
            } else content

            val documentContext = Message(
                role = MessageRole.SYSTEM,
                content = """Documento adjunto por el usuario:
Nombre: ${metadata.name}
Tipo: ${metadata.mimeType}
Tamaño: ${metadata.sizeBytes} bytes

Contenido:
$truncatedContent"""
            )
            _uiState.update {
                it.copy(
                    messages = it.messages + documentContext,
                    inferenceState = InferenceState.IDLE,
                    attachedDocumentName = metadata.name
                )
            }
        }
    }
}
```

**Presupuesto de tokens para la Sandbox:**

| Segmento | Tokens asignados | Caracteres (~4 chars/token) |
|---|---|---|
| System prompt + herramientas | ~800 | ~3200 |
| Historial de conversación | ~1024 | ~4096 |
| Documento adjunto (sandbox) | ~2048 | ~8192 |
| Generación de respuesta | ~224 | ~896 |
| **Total contexto** | **4096** | **~16384** |

### 3.4 Lanzamiento del SAF Picker desde Compose

```kotlin
// presentation/ui/components/AttachDocumentButton.kt
@Composable
fun AttachDocumentButton(onDocumentSelected: (Uri) -> Unit) {
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { onDocumentSelected(it) }
    }

    IconButton(onClick = {
        launcher.launch(arrayOf(
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/pdf",
            "application/json"
        ))
    }) {
        Icon(
            imageVector = Icons.Default.AttachFile,
            contentDescription = "Adjuntar documento"
        )
    }
}
```

---

## 4. Estrategia de Control Térmico y Memoria

### 4.1 Máquina de Estado Térmica

```kotlin
// domain/model/ThermalState.kt
sealed class ThermalState(val level: Int) {
    data object None : ThermalState(0)
    data object Light : ThermalState(1)
    data object Moderate : ThermalState(2)
    data object Severe : ThermalState(3)
    data object Critical : ThermalState(4)
    data object Emergency : ThermalState(5)
    data object Shutdown : ThermalState(6)

    companion object {
        fun fromPowerManager(status: Int): ThermalState = when (status) {
            PowerManager.THERMAL_STATUS_NONE -> None
            PowerManager.THERMAL_STATUS_LIGHT -> Light
            PowerManager.THERMAL_STATUS_MODERATE -> Moderate
            PowerManager.THERMAL_STATUS_SEVERE -> Severe
            PowerManager.THERMAL_STATUS_CRITICAL -> Critical
            PowerManager.THERMAL_STATUS_EMERGENCY -> Emergency
            PowerManager.THERMAL_STATUS_SHUTDOWN -> Shutdown
            else -> None
        }
    }
}

// domain/model/ThermalPolicy.kt
data class ThermalPolicy(
    val maxTokens: Int,
    val contextSize: Int,
    val cooldownBetweenRequestsMs: Long,
    val gpuEnabled: Boolean,
    val label: String
) {
    fun toInferenceConfig() = InferenceConfig(
        maxTokens = maxTokens,
        contextSize = contextSize,
        useGpu = gpuEnabled
    )

    companion object {
        val NORMAL = ThermalPolicy(
            maxTokens = 512,
            contextSize = 4096,
            cooldownBetweenRequestsMs = 0,
            gpuEnabled = true,
            label = "Normal"
        )
        val DEGRADED = ThermalPolicy(
            maxTokens = 384,
            contextSize = 3072,
            cooldownBetweenRequestsMs = 1000,
            gpuEnabled = true,
            label = "Degradado"
        )
        val THROTTLED = ThermalPolicy(
            maxTokens = 256,
            contextSize = 2048,
            cooldownBetweenRequestsMs = 2000,
            gpuEnabled = false,  // Desactivar GPU para reducir carga térmica
            label = "Limitado"
        )
        val PAUSED = ThermalPolicy(
            maxTokens = 0,
            contextSize = 0,
            cooldownBetweenRequestsMs = Long.MAX_VALUE,
            gpuEnabled = false,
            label = "Pausado"
        )
    }
}
```

### 4.2 Implementación del Monitor Térmico

```kotlin
// data/monitor/AndroidThermalMonitor.kt
@Singleton
class AndroidThermalMonitor @Inject constructor(
    @ApplicationContext private val context: Context
) : ThermalMonitor {

    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val _thermalState = MutableStateFlow<ThermalState>(ThermalState.None)
    override val thermalState: StateFlow<ThermalState> = _thermalState.asStateFlow()

    private val listener = PowerManager.OnThermalStatusChangedListener { status ->
        val newState = ThermalState.fromPowerManager(status)
        val previousState = _thermalState.value

        if (newState != previousState) {
            Log.d("ThermalMonitor",
                "Transición térmica: ${previousState::class.simpleName} → ${newState::class.simpleName}")
            _thermalState.value = newState
        }
    }

    override fun startMonitoring() {
        powerManager.addThermalStatusListener(
            ContextCompat.getMainExecutor(context),
            listener
        )
        // Leer estado inicial
        _thermalState.value = ThermalState.fromPowerManager(powerManager.currentThermalStatus)
    }

    override fun stopMonitoring() {
        powerManager.removeThermalStatusListener(listener)
    }
}
```

### 4.3 Use Case de Política Térmica

```kotlin
// domain/usecase/ThermalPolicyUseCase.kt
class ThermalPolicyUseCase @Inject constructor(
    private val thermalMonitor: ThermalMonitor
) {
    val activePolicyFlow: Flow<ThermalPolicy> = thermalMonitor.thermalState.map { state ->
        when (state) {
            is ThermalState.None,
            is ThermalState.Light -> ThermalPolicy.NORMAL

            is ThermalState.Moderate -> ThermalPolicy.DEGRADED

            is ThermalState.Severe -> ThermalPolicy.THROTTLED

            is ThermalState.Critical,
            is ThermalState.Emergency,
            is ThermalState.Shutdown -> ThermalPolicy.PAUSED
        }
    }.distinctUntilChanged()
}
```

### 4.4 Tabla de Reglas de Reducción Dinámica

| Estado térmico | `maxTokens` | `contextSize` | GPU | Cooldown entre requests | Acción en UI |
|---|---|---|---|---|---|
| `NONE` / `LIGHT` | 512 | 4096 | ON | 0 ms | Ninguna |
| `MODERATE` | 384 | 3072 | ON | 1000 ms | Banner amarillo: "Dispositivo calentándose" |
| `SEVERE` | 256 | 2048 | OFF | 2000 ms | Banner naranja: "Rendimiento reducido" |
| `CRITICAL` / `EMERGENCY` | 0 | 0 | OFF | ∞ | Banner rojo: "Inferencia pausada por temperatura" |
| `SHUTDOWN` | 0 | 0 | OFF | ∞ | Liberar modelo de memoria |

### 4.5 Presupuesto de Memoria en Tiempo de Ejecución

| Componente | Consumo (MB) |
|---|---|
| Android 16 + HyperOS 3.0 + servicios de sistema | ~3500 |
| Pesos del modelo Gemma 4 E2B INT4 | ~2000-2600 |
| KV Cache (ctx=4096, INT4, 18 layers, 2048 dim) | ~400-600 |
| Runtime LiteRT-LM (Engine + buffers) | ~200-400 |
| Aplicación Android (Compose, ViewModels, DI) | ~100-200 |
| **Total en uso** | **~6200-7300** |
| **Headroom en 8 GB físicos** | **~700-1800** |

Regla de seguridad: si `ActivityManager.MemoryInfo.availMem` < 500 MB, forzar `ThermalPolicy.THROTTLED` independientemente del estado térmico:

```kotlin
// Extensión al ThermalPolicyUseCase
val effectivePolicyFlow: Flow<ThermalPolicy> = combine(
    activePolicyFlow,
    memoryMonitorFlow()  // Flow que emite cada 10s el availMem
) { thermalPolicy, availMemMb ->
    if (availMemMb < 500 && thermalPolicy != ThermalPolicy.PAUSED) {
        ThermalPolicy.THROTTLED.copy(label = "Memoria baja (${availMemMb}MB)")
    } else {
        thermalPolicy
    }
}
```

---

## 5. Guía de Desarrollo de Código (Snippets Críticos)

### 5.1 Inicialización de LiteRT-LM con GPU Offload (Adreno 710)

```kotlin
// data/repository/LiteRTLMInferenceRepository.kt
@Singleton
class LiteRTLMInferenceRepository @Inject constructor(
    @ApplicationContext private val context: Context
) : InferenceRepository {

    private var engine: Engine? = null
    private var conversation: Conversation? = null

    override suspend fun initialize(
        modelPath: String,
        config: InferenceConfig
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val engineConfig = EngineConfig(
                modelPath = modelPath,
                // Seleccionar backend GPU para Adreno 710 (Snapdragon 7s Gen 2)
                // LiteRT-LM auto-selecciona el backend óptimo entre CPU/GPU/NPU
                // En Adreno 710: OpenCL 2.0 backend es el preferido
                backend = if (config.useGpu) Backend.GPU else Backend.CPU,
                maxContextLength = config.contextSize,
                numThreads = 4  // Usar solo los 4 cores Cortex-A78 de rendimiento
            )

            engine = Engine(engineConfig).also { eng ->
                eng.initialize()
                conversation = eng.createConversation()
            }

            Log.d("LiteRTLM", buildString {
                append("Motor inicializado: ")
                append("backend=${if (config.useGpu) "GPU" else "CPU"}, ")
                append("ctx=${config.contextSize}, ")
                append("threads=4")
            })
        }
    }

    override fun streamResponse(prompt: String): Flow<String> = channelFlow {
        val conv = conversation
            ?: throw IllegalStateException("Motor no inicializado. Llamar initialize() primero.")

        conv.sendMessageAsync(prompt).collect { token ->
            send(token)
        }
    }.flowOn(Dispatchers.IO)

    override fun isInitialized(): Boolean = engine != null

    override suspend fun release() = withContext(Dispatchers.IO) {
        conversation?.close()
        engine?.close()
        conversation = null
        engine = null
        Log.d("LiteRTLM", "Motor liberado")
    }
}
```

### 5.2 Ciclo de Vida del Motor en Application/Activity

```kotlin
// presentation/GemmaApplication.kt
@HiltAndroidApp
class GemmaApplication : Application() {

    @Inject lateinit var inferenceRepository: InferenceRepository
    @Inject lateinit var thermalMonitor: ThermalMonitor

    override fun onCreate() {
        super.onCreate()
        thermalMonitor.startMonitoring()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
            // Sistema necesita memoria urgentemente — liberar modelo
            CoroutineScope(Dispatchers.IO).launch {
                inferenceRepository.release()
                Log.w("GemmaApp", "Modelo liberado por presión de memoria (level=$level)")
            }
        }
    }
}
```

### 5.3 Parser de Intenciones — Detección de Tool Calls en Streaming

El parser debe detectar tokens de herramientas durante el streaming (no al final), para poder interrumpir la generación y solicitar confirmación.

```kotlin
// domain/parser/StreamingToolCallParser.kt
class StreamingToolCallParser {

    private val buffer = StringBuilder()
    private var insideToolCall = false
    private var toolCallBuffer = StringBuilder()

    sealed class ParseEvent {
        data class TextToken(val text: String) : ParseEvent()
        data class ToolCallDetected(val toolCall: ToolCall) : ParseEvent()
        data class PartialToolCall(val accumulated: String) : ParseEvent()
    }

    /**
     * Procesa cada token emitido por el modelo durante streaming.
     * Retorna un ParseEvent indicando si el token es texto normal,
     * parte de un tool_call en progreso, o un tool_call completo.
     */
    fun processToken(token: String): ParseEvent {
        buffer.append(token)

        // Detectar inicio de tool_call
        if (!insideToolCall && buffer.endsWith("<|tool_call>")) {
            insideToolCall = true
            toolCallBuffer.clear()
            // Remover el marcador del buffer de texto visible
            val textBeforeMarker = buffer.toString()
                .removeSuffix("<|tool_call>")
            buffer.clear()
            buffer.append(textBeforeMarker)
            return ParseEvent.PartialToolCall("")
        }

        if (insideToolCall) {
            toolCallBuffer.append(token)

            // Detectar cierre de tool_call
            if (toolCallBuffer.toString().contains("<|tool_call|>")) {
                insideToolCall = false
                val jsonContent = toolCallBuffer.toString()
                    .substringBefore("<|tool_call|>")
                    .trim()
                toolCallBuffer.clear()

                return try {
                    val json = JSONObject(jsonContent)
                    val toolCall = ToolCall(
                        name = json.getString("name"),
                        arguments = json.getJSONObject("arguments").toMap()
                    )
                    ParseEvent.ToolCallDetected(toolCall)
                } catch (e: JSONException) {
                    // JSON malformado — tratar como texto
                    ParseEvent.TextToken(jsonContent)
                }
            }

            return ParseEvent.PartialToolCall(toolCallBuffer.toString())
        }

        return ParseEvent.TextToken(token)
    }

    fun reset() {
        buffer.clear()
        toolCallBuffer.clear()
        insideToolCall = false
    }
}
```

### 5.4 Integración del Parser en el Flujo de Streaming del ViewModel

```kotlin
// Dentro de ChatViewModel.sendMessage()
viewModelScope.launch {
    val parser = StreamingToolCallParser()
    val visibleText = StringBuilder()

    inferenceUseCase.streamInference(prompt, policy.toInferenceConfig())
        .onEach { token ->
            when (val event = parser.processToken(token)) {
                is StreamingToolCallParser.ParseEvent.TextToken -> {
                    visibleText.append(event.text)
                    _uiState.update { it.copy(streamBuffer = visibleText.toString()) }
                }
                is StreamingToolCallParser.ParseEvent.PartialToolCall -> {
                    // Mostrar indicador de "procesando herramienta..."
                    _uiState.update { it.copy(inferenceState = InferenceState.GENERATING) }
                }
                is StreamingToolCallParser.ParseEvent.ToolCallDetected -> {
                    // Interrumpir streaming, solicitar confirmación si destructiva
                    val handler = toolRegistry.getHandler(event.toolCall.name)
                    if (handler?.isDestructive == true) {
                        _uiState.update {
                            it.copy(
                                inferenceState = InferenceState.AWAITING_TOOL_CONFIRMATION,
                                pendingToolCall = event.toolCall,
                                streamBuffer = visibleText.toString()
                            )
                        }
                    } else {
                        // Herramienta no destructiva: ejecutar directamente
                        _uiState.update {
                            it.copy(inferenceState = InferenceState.EXECUTING_TOOL)
                        }
                        val result = toolOrchestrationUseCase.executeTool(event.toolCall)
                        continueWithToolResult(result)
                    }
                }
            }
        }
        .catch { error ->
            _uiState.update {
                it.copy(inferenceState = InferenceState.ERROR, errorMessage = error.message)
            }
        }
        .collect()
}
```

### 5.5 Estructura de Directorio del Proyecto

```
app/
├── src/main/
│   ├── java/com/aurora/gemma/
│   │   ├── di/
│   │   │   ├── AppModule.kt
│   │   │   ├── InferenceModule.kt
│   │   │   └── ToolModule.kt
│   │   ├── domain/
│   │   │   ├── model/
│   │   │   │   ├── Message.kt
│   │   │   │   ├── ToolCall.kt
│   │   │   │   ├── ToolDefinition.kt
│   │   │   │   ├── ToolResult.kt
│   │   │   │   ├── ThermalState.kt
│   │   │   │   └── ThermalPolicy.kt
│   │   │   ├── repository/
│   │   │   │   ├── InferenceRepository.kt
│   │   │   │   └── FileRepository.kt
│   │   │   ├── monitor/
│   │   │   │   └── ThermalMonitor.kt
│   │   │   ├── registry/
│   │   │   │   ├── ToolRegistry.kt
│   │   │   │   └── ToolHandler.kt
│   │   │   ├── parser/
│   │   │   │   └── StreamingToolCallParser.kt
│   │   │   └── usecase/
│   │   │       ├── InferenceUseCase.kt
│   │   │       ├── ToolOrchestrationUseCase.kt
│   │   │       ├── FileAccessUseCase.kt
│   │   │       └── ThermalPolicyUseCase.kt
│   │   ├── data/
│   │   │   ├── repository/
│   │   │   │   ├── LiteRTLMInferenceRepository.kt
│   │   │   │   └── ScopedFileRepository.kt
│   │   │   ├── monitor/
│   │   │   │   └── AndroidThermalMonitor.kt
│   │   │   ├── registry/
│   │   │   │   └── ToolRegistryImpl.kt
│   │   │   └── tools/
│   │   │       ├── FileReadHandler.kt
│   │   │       ├── ListDirectoryHandler.kt
│   │   │       ├── WriteFileHandler.kt
│   │   │       └── DeviceSensorHandler.kt
│   │   └── presentation/
│   │       ├── GemmaApplication.kt
│   │       ├── MainActivity.kt
│   │       ├── state/
│   │       │   └── ChatUiState.kt
│   │       ├── viewmodel/
│   │       │   ├── ChatViewModel.kt
│   │       │   └── ThermalViewModel.kt
│   │       └── ui/
│   │           ├── screens/
│   │           │   └── ChatScreen.kt
│   │           ├── components/
│   │           │   ├── MessageBubble.kt
│   │           │   ├── StreamingText.kt
│   │           │   ├── ToolConfirmationDialog.kt
│   │           │   ├── ThermalStatusBanner.kt
│   │           │   └── AttachDocumentButton.kt
│   │           └── theme/
│   │               └── GemmaTheme.kt
│   └── res/
│       └── ...
├── build.gradle.kts
└── proguard-rules.pro
```

### 5.6 Dependencias Gradle

```kotlin
// app/build.gradle.kts
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.dagger.hilt.android")
    id("org.jetbrains.kotlin.plugin.compose")
    kotlin("kapt")
}

android {
    namespace = "com.aurora.gemma"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.aurora.gemma"
        minSdk = 29  // API 29 para PowerManager.THERMAL_STATUS
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"
    }

    buildFeatures {
        compose = true
    }
}

dependencies {
    // LiteRT-LM (reemplazo de MediaPipe LLM Inference)
    implementation("com.google.ai.edge.litertlm:litertlm-android:latest.release")

    // Jetpack Compose BOM
    val composeBom = platform("androidx.compose:compose-bom:2026.04.00")
    implementation(composeBom)
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")

    // Hilt DI
    implementation("com.google.dagger:hilt-android:2.52")
    kapt("com.google.dagger:hilt-android-compiler:2.52")
    implementation("androidx.hilt:hilt-navigation-compose:1.2.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // DocumentFile (SAF)
    implementation("androidx.documentfile:documentfile:1.0.1")

    // JSON parsing
    implementation("org.json:json:20240303")
}
```

---

## Referencias Técnicas

| Fuente | URL |
|---|---|
| LiteRT-LM Overview | https://ai.google.dev/edge/litert-lm/overview |
| LiteRT-LM GitHub | https://github.com/google-ai-edge/LiteRT-LM |
| LiteRT-LM Kotlin Getting Started | https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md |
| Gemma 4 Function Calling Docs | https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4 |
| Gemma 4 Tool Calling Playbook | https://www.gemma4.app/playbooks/tool-calling |
| Gemma 4 E2B — Hugging Face | https://huggingface.co/google/gemma-4-E2B |
| Android Scoped Storage Guide | https://source.android.com/docs/core/storage/scoped |
| Android Data Storage Overview | https://developer.android.com/training/data-storage |
| PowerManager Thermal API | https://developer.android.com/reference/kotlin/android/os/PowerManager.OnThermalStatusChangedListener |
| Android Thermal Mitigation | https://source.android.com/docs/core/power/thermal-mitigation |
| MediaPipe LLM (deprecated) → LiteRT-LM | https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference |
| Compose Architecture Guide | https://developer.android.com/develop/ui/compose/architecture |
