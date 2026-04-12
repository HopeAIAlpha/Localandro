# Despliegue Local de Gemma 4 E2B en Redmi Note 13 Pro

## 1. Resumen de Viabilidad Técnica y Arquitectónica

### 1.1 Modelo Seleccionado: Gemma 4 E2B

| Parámetro | Valor |
|---|---|
| Parámetros efectivos | ~2.3B (activos por token) |
| Parámetros totales (con embeddings) | ~5.1B |
| Ventana de contexto | 128K tokens |
| Modalidades de entrada | Texto, imagen, video (hasta 60s a 1 FPS), audio nativo (ASR) |
| Licencia | Apache 2.0 |
| Fecha de publicación | 31 de marzo de 2026 |
| Soporte nativo de function calling | Sí — protocolo de tokens especiales `<\|tool\|>`, `<\|tool_call\|>`, `<\|tool_result\|>` |
| Arquitectura de atención | Híbrida: sliding-window local + full-context global alterno |
| Técnica de eficiencia | Per-Layer Embeddings (PLE) |

**Justificación de selección sobre E4B:** El variante E4B (~4.5B efectivos, ~8B totales) requiere ~3.6 GB de VRAM en FP16 y ~5-6 GB en INT4. En el Redmi Note 13 Pro, con 8 GB de RAM física (+ 8 GB de RAM virtual/swap por MIUI), el presupuesto de memoria disponible para el runtime del modelo tras descontar el sistema operativo (Android 16 + HyperOS consume ~3-4 GB) deja ~4-5 GB reales para inferencia. E2B en INT4 requiere ~2-3 GB, lo que garantiza headroom operativo y reduce la presión térmica. E4B es viable pero marginal; E2B es la elección determinista para estabilidad sostenida.

### 1.2 Perfil del Hardware Objetivo

| Componente | Especificación |
|---|---|
| SoC | Qualcomm Snapdragon 7s Gen 2 (SM7435) |
| Proceso de fabricación | Samsung 4nm |
| CPU | 4× Cortex-A78 @ 2.4 GHz + 4× Cortex-A55 @ 1.95 GHz |
| GPU | Adreno 710 (Vulkan 1.1, OpenCL 2.0, OpenGL ES 3.2) |
| NPU | Hexagon 710 (AI Engine de Qualcomm) |
| RAM física | 8 GB LPDDR5 @ 3200 MHz |
| RAM virtual (MIUI) | 8 GB adicionales (swap en almacenamiento) |
| Almacenamiento disponible | 82.8 GB (UFS 2.2) |
| Android | 16 (compilación BP2A.250605.031.A3) |
| Parche de seguridad | 2026-02-01 |

### 1.3 Evaluación de Restricciones Térmicas

El Snapdragon 7s Gen 2 fabricado en el nodo Samsung 4nm presenta degradación térmica de 10-25% tras 10-15 minutos de carga sostenida. El Redmi Note 13 Pro no incluye cámara de vapor ni sistema de refrigeración activa. Esto implica:

- **Inferencia en ráfagas cortas (< 5 min):** rendimiento pleno (~16-25 tokens/s para E2B en INT4).
- **Inferencia sostenida (> 15 min):** degradación esperada a ~12-18 tokens/s por throttling térmico.
- **Mitigación recomendada:** limitar `max_tokens` por generación a 512-1024, implementar pausas entre solicitudes consecutivas de >2 segundos, y reducir la ventana de contexto activa a 4096-8192 tokens en lugar de los 128K teóricos.

### 1.4 Rendimiento TTT (Time To Token) Estimado

| Métrica | Valor estimado (E2B INT4, Redmi Note 13 Pro) |
|---|---|
| Time to First Token (TTFT) | 800-2500 ms (carga en frío: 10-30s primera vez) |
| Tokens por segundo (sostenido) | 12-20 tok/s |
| Tokens por segundo (ráfaga) | 16-25 tok/s |
| Latencia percibida con streaming | < 1s para primer token tras warm-up |

---

## 2. Prerrequisitos y Dependencias de Software

### 2.1 Ruta A: Google AI Edge Gallery (Recomendada — Sin Compilación)

| Dependencia | Versión / Fuente | Propósito |
|---|---|---|
| Google AI Edge Gallery APK | Última release desde [github.com/google-ai-edge/gallery/releases](https://github.com/google-ai-edge/gallery/releases) | Aplicación host para inferencia on-device |
| Modelo Gemma 4 E2B (task file) | `gemma-4-e2b-it-int4.task` (~2.6 GB) descargado desde la app o [Kaggle](https://www.kaggle.com/models/google/gemma-4) | Archivo de modelo cuantizado INT4 empaquetado para MediaPipe |
| Android | ≥ 14 (el dispositivo tiene Android 16 ✓) | Compatibilidad con AI Edge Gallery |
| Almacenamiento libre | ≥ 5 GB (APK + modelo + cache) | Espacio para el modelo y datos temporales |
| Permisos | Almacenamiento, instalación de fuentes desconocidas | Sideloading del APK si no se usa Play Store |

### 2.2 Ruta B: MediaPipe LLM Inference API (Integración en App Android Propia)

| Dependencia | Versión / Fuente | Propósito |
|---|---|---|
| Android Studio | Hedgehog 2024.1+ o posterior | IDE de desarrollo |
| Android SDK | API Level 34+ (target: 35/36) | SDK de compilación |
| Gradle dependency | `com.google.mediapipe:tasks-genai:0.10.27` | Librería MediaPipe GenAI |
| Modelo Gemma 4 E2B | `gemma-4-e2b-it-int4.task` | Modelo cuantizado para MediaPipe |
| Kotlin | 1.9+ | Lenguaje de la aplicación host |
| ADB | Incluido en Android SDK Platform-Tools | Transferencia del modelo al dispositivo |

### 2.3 Ruta C: llama.cpp Cross-Compilado (Control Total de Inferencia)

| Dependencia | Versión / Fuente | Propósito |
|---|---|---|
| llama.cpp | `master` branch de [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Motor de inferencia |
| Android NDK | r26+ | Toolchain de cross-compilación ARM64 |
| CMake | 3.24+ | Sistema de build |
| Modelo GGUF | `gemma-4-e2b-it-Q4_K_M.gguf` (~2-4 GB) de [Hugging Face](https://huggingface.co/google/gemma-4-E2B) | Modelo cuantizado en formato GGUF |
| Termux (opcional) | F-Droid o GitHub release | Shell en el dispositivo para ejecución directa |
| ADB | Android SDK Platform-Tools | Transferencia de binarios y modelo |

---

## 3. Guía Paso a Paso

### Ruta A: Despliegue vía Google AI Edge Gallery

#### Paso 1 — Descargar e instalar el APK

```
1. En el Redmi Note 13 Pro, abrir Ajustes > Aplicaciones > Acceso especial a aplicaciones > Instalar apps desconocidas.
2. Habilitar el permiso para el navegador o gestor de archivos que usarás.
3. Navegar a: https://github.com/google-ai-edge/gallery/releases
4. Descargar el archivo .apk de la última release estable.
5. Abrir el archivo descargado e instalar.
6. Conceder permisos de almacenamiento al abrir la aplicación.
```

#### Paso 2 — Descargar el modelo Gemma 4 E2B

```
1. Abrir Google AI Edge Gallery.
2. Seleccionar "AI Chat" o "Get Models".
3. Localizar "Gemma 4 E2B" (variante INT4, ~2.6 GB).
4. Iniciar la descarga. El modelo se almacena localmente en /storage/emulated/0/Android/data/com.google.ai.edge.gallery/.
5. Verificar que el almacenamiento disponible post-descarga sea ≥ 2 GB libre.
```

#### Paso 3 — Configurar parámetros de inferencia

Dentro de la app, en la sección "Prompt Lab" o configuración del chat:

| Parámetro | Valor recomendado | Justificación |
|---|---|---|
| `temperature` | `0.6` | Balance entre coherencia y variabilidad; valores > 0.8 incrementan latencia por mayor entropía |
| `top_k` | `40` | Restricción del vocabulario de muestreo para reducir cómputo |
| `max_tokens` | `512` | Limitar longitud de generación para mantener TTT estable y evitar acumulación térmica |
| Contexto activo | `4096` tokens | Reducir de 128K teórico a 4K para mantener KV cache dentro del presupuesto de RAM |

#### Paso 4 — Ejecutar benchmark integrado

```
1. En AI Edge Gallery, navegar a "Model Management & Benchmark".
2. Seleccionar el modelo Gemma 4 E2B descargado.
3. Ejecutar el benchmark.
4. Registrar: tokens/segundo, TTFT, uso de memoria.
5. Valores esperados: 16-25 tok/s (ráfaga), 12-20 tok/s (sostenido).
```

#### Paso 5 — Validar function calling (tools)

```
1. En el chat de AI Edge Gallery, configurar un system prompt con declaración de herramientas.
2. Verificar que el modelo responde con el formato <|tool_call> ... <tool_call|>.
3. Inyectar resultados con formato <|tool_result> ... <tool_result|>.
```

---

### Ruta B: Integración en Aplicación Android con MediaPipe

#### Paso 1 — Crear proyecto Android

```
1. Abrir Android Studio.
2. New Project > Empty Compose Activity.
3. Min SDK: API 26 (Android 8.0).
4. Target SDK: API 35+.
5. Lenguaje: Kotlin.
```

#### Paso 2 — Agregar dependencia de MediaPipe GenAI

En `build.gradle.kts` (módulo app):

```kotlin
dependencies {
    implementation("com.google.mediapipe:tasks-genai:0.10.27")
}
```

Sincronizar Gradle.

#### Paso 3 — Transferir modelo al dispositivo

```bash
adb shell mkdir -p /data/local/tmp/llm/
adb push gemma-4-e2b-it-int4.task /data/local/tmp/llm/
```

Alternativa para producción: usar Play Asset Delivery o descarga on-demand desde un servidor propio.

#### Paso 4 — Inicializar LlmInference

```kotlin
import com.google.mediapipe.tasks.genai.llminference.LlmInference

val options = LlmInference.LlmInferenceOptions.builder()
    .setModelPath("/data/local/tmp/llm/gemma-4-e2b-it-int4.task")
    .setMaxTokens(512)           // Limitar generación por respuesta
    .setTemperature(0.6f)        // Coherencia sobre creatividad
    .build()

val llmInference = LlmInference.createFromOptions(applicationContext, options)
```

#### Paso 5 — Implementar generación con streaming

```kotlin
// Generación síncrona (bloquea thread — usar en coroutine)
val response = llmInference.generateResponse("¿Cuál es la capital de Francia?")

// Generación asíncrona con streaming (recomendada para UI)
llmInference.generateResponseAsync("Explica la fotosíntesis") { partialResult, done ->
    runOnUiThread {
        updateChatUI(partialResult)
        if (done) markGenerationComplete()
    }
}
```

#### Paso 6 — Implementar orquestación de herramientas (function calling)

```kotlin
val systemPrompt = """
<start_of_turn>system
Eres un asistente con acceso a las siguientes herramientas:
<|tool>
{
  "name": "buscar_paciente",
  "description": "Busca información de un paciente por su ID",
  "parameters": {
    "type": "object",
    "properties": {
      "patient_id": {"type": "string", "description": "ID único del paciente"}
    },
    "required": ["patient_id"]
  }
}
<tool|>
<|tool>
{
  "name": "registrar_nota",
  "description": "Registra una nota clínica para un paciente",
  "parameters": {
    "type": "object",
    "properties": {
      "patient_id": {"type": "string"},
      "content": {"type": "string"},
      "category": {"type": "string", "enum": ["progress_note", "assessment", "plan"]}
    },
    "required": ["patient_id", "content"]
  }
}
<tool|>
<end_of_turn>
""".trimIndent()

// Construir prompt completo
val fullPrompt = systemPrompt + "\n<start_of_turn>user\nBusca al paciente P-1234\n<end_of_turn>"
val modelOutput = llmInference.generateResponse(fullPrompt)

// Parser: detectar <|tool_call> en la respuesta
val toolCallRegex = Regex("<\\|tool_call>\\s*(\\{.*?\\})\\s*<\\|tool_call\\|>", RegexOption.DOT_MATCHES_ALL)
val match = toolCallRegex.find(modelOutput)
if (match != null) {
    val toolCallJson = JSONObject(match.groupValues[1])
    val toolName = toolCallJson.getString("name")
    val args = toolCallJson.getJSONObject("arguments")
    // Ejecutar la herramienta localmente
    val result = executeLocalTool(toolName, args)
    // Re-inyectar resultado
    val continuation = fullPrompt + "\n" + modelOutput +
        "\n<|tool_result>\n$result\n<tool_result|>\n<start_of_turn>model\n"
    val finalResponse = llmInference.generateResponse(continuation)
}
```

#### Paso 7 — Gestión térmica programática

```kotlin
// Monitorear temperatura del dispositivo
val thermalService = getSystemService(Context.POWER_SERVICE) as PowerManager

// Android 16 soporta thermal status listener
thermalService.addThermalStatusListener { status ->
    when (status) {
        PowerManager.THERMAL_STATUS_MODERATE -> {
            // Reducir max_tokens a 256, aumentar pausa entre requests a 3s
            adjustInferenceParams(maxTokens = 256, cooldownMs = 3000)
        }
        PowerManager.THERMAL_STATUS_SEVERE -> {
            // Pausar inferencia, notificar al usuario
            pauseInference()
            showThermalWarning()
        }
    }
}
```

---

### Ruta C: llama.cpp Cross-Compilado con Aceleración GPU

#### Paso 1 — Clonar llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

#### Paso 2 — Cross-compilar para ARM64 con OpenCL (recomendado para Adreno 710)

OpenCL es más estable que Vulkan en el GPU Adreno 710 del Snapdragon 7s Gen 2:

```bash
export ANDROID_NDK=/path/to/android-ndk-r26c

cmake -B build-android \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON

cmake --build build-android --config Release -j$(nproc)
```

Alternativa con Vulkan (menos estable en Adreno 710):

```bash
cmake -B build-android-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_VULKAN=ON

cmake --build build-android-vulkan --config Release -j$(nproc)
```

#### Paso 3 — Obtener modelo GGUF cuantizado

```bash
# Opción 1: Descargar directamente desde Hugging Face
# Buscar: google/gemma-4-E2B-GGUF o cuantizaciones de la comunidad
# Archivo recomendado: gemma-4-e2b-it-Q4_K_M.gguf (~2-3 GB)

# Opción 2: Cuantizar desde pesos originales con llama.cpp
python convert_hf_to_gguf.py /path/to/gemma-4-e2b/ --outtype q4_k_m
```

#### Paso 4 — Transferir binario y modelo al dispositivo

```bash
adb push build-android/bin/llama-cli /data/local/tmp/
adb push gemma-4-e2b-it-Q4_K_M.gguf /data/local/tmp/
adb shell chmod +x /data/local/tmp/llama-cli
```

#### Paso 5 — Ejecutar inferencia en el dispositivo

```bash
adb shell
cd /data/local/tmp

./llama-cli \
  -m gemma-4-e2b-it-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --batch-size 256 \
  --threads 4 \
  --temp 0.6 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  -p "<start_of_turn>user\n¿Qué es la inteligencia artificial?\n<end_of_turn>\n<start_of_turn>model\n"
```

**Explicación de parámetros:**

| Flag | Valor | Justificación |
|---|---|---|
| `--n-gpu-layers 99` | Offload completo al GPU Adreno 710 vía OpenCL/Vulkan | Maximizar uso de GPU para descargar CPU |
| `--ctx-size 4096` | 4096 tokens | KV cache cabe en ~500 MB con INT4; 128K requeriría >8 GB |
| `--batch-size 256` | 256 tokens por batch | Balance entre throughput y memoria; 512 puede causar OOM |
| `--threads 4` | 4 threads CPU | Usar solo los 4 cores A78 (2.4 GHz); evitar los A55 de eficiencia |
| `--temp 0.6` | Temperatura 0.6 | Coherencia sin repetición excesiva |
| `--top-k 40` | Top-K 40 | Reducción de vocabulario activo para acelerar sampling |
| `--top-p 0.9` | Nucleus sampling 0.9 | Complemento a top-k para calidad |
| `--repeat-penalty 1.1` | Penalización de repetición 1.1 | Evitar bucles de texto |

#### Paso 6 — Modo servidor (para integración con app Android)

```bash
./llama-server \
  -m gemma-4-e2b-it-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --batch-size 256 \
  --threads 4 \
  --host 127.0.0.1 \
  --port 8080
```

La app Android se conecta a `http://127.0.0.1:8080/v1/chat/completions` con el formato OpenAI-compatible:

```json
{
  "model": "gemma-4-e2b",
  "messages": [
    {"role": "system", "content": "Eres un asistente clínico."},
    {"role": "user", "content": "Busca al paciente P-1234"}
  ],
  "max_tokens": 512,
  "temperature": 0.6,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "buscar_paciente",
        "description": "Busca información de un paciente por ID",
        "parameters": {
          "type": "object",
          "properties": {
            "patient_id": {"type": "string"}
          },
          "required": ["patient_id"]
        }
      }
    }
  ]
}
```

---

## 4. Parámetros de Configuración Técnica para TTT Óptimo

### 4.1 Tabla de Configuración Consolidada

| Parámetro | Valor | Aplica a | Efecto en TTT |
|---|---|---|---|
| Cuantización | INT4 (Q4_K_M) | Todas las rutas | Reduce footprint de ~10 GB (FP16) a ~2-3 GB; incrementa tok/s ~2x vs INT8 |
| Contexto activo | 4096 tokens | Todas las rutas | KV cache ~500 MB en INT4; 8192 = ~1 GB; 128K inviable en 8 GB RAM |
| Batch size | 256 | llama.cpp | Throughput máximo sin exceder presupuesto de memoria |
| Threads CPU | 4 | llama.cpp | Solo cores de rendimiento (A78); A55 añaden latencia por su menor IPC |
| GPU layers | 99 (all) | llama.cpp (OpenCL/Vulkan) | Offload completo a Adreno 710; ~30% mejora vs CPU-only |
| Temperature | 0.6 | Todas las rutas | Menor entropía = menor cómputo en softmax sampling |
| Top-K | 40 | Todas las rutas | Reduce candidatos de ~256K vocab a 40; ahorra ciclos de sorting |
| Top-P | 0.9 | Todas las rutas | Nucleus sampling complementario |
| Max tokens | 512 | Todas las rutas | Previene acumulación térmica; genera en ~20-40 segundos |
| Repeat penalty | 1.1 | llama.cpp | Evita loops que desperdician tokens/tiempo |

### 4.2 Presupuesto de Memoria Detallado

| Componente | Consumo estimado |
|---|---|
| Android 16 + HyperOS 3.0 + servicios | ~3.5 GB |
| Modelo E2B INT4 (pesos) | ~2.0-2.6 GB |
| KV cache (ctx=4096, INT4) | ~400-600 MB |
| Runtime (MediaPipe / llama.cpp) | ~200-400 MB |
| **Total en uso** | **~6.1-7.1 GB** |
| **Headroom disponible (de 8 GB físicos)** | **~0.9-1.9 GB** |
| RAM virtual MIUI (swap) | 8 GB adicionales (penalización de latencia en acceso) |

### 4.3 Recomendaciones de Gestión Térmica

1. **Antes de iniciar inferencia:** cerrar aplicaciones en segundo plano para maximizar RAM disponible y reducir carga térmica base.
2. **Durante inferencia sostenida:** implementar un cooldown de 2-3 segundos entre generaciones consecutivas.
3. **Monitoreo programático:** usar `PowerManager.THERMAL_STATUS_*` (Android API 29+) para escalar dinámicamente `max_tokens` y `ctx_size` según estado térmico.
4. **Límite operativo:** para sesiones > 15 minutos, reducir `max_tokens` a 256 y `ctx_size` a 2048.

### 4.4 Protocolo de Orquestación de Herramientas (Tools/Flows)

Gemma 4 E2B implementa un protocolo de tokens especiales para function calling:

```
Definición de herramienta:     <|tool> { JSON Schema } <tool|>
Solicitud de invocación:       <|tool_call> { "name": "...", "arguments": {...} } <tool_call|>
Retorno de resultado:          <|tool_result> { JSON resultado } <tool_result|>
```

**Flujo de orquestación completo:**

```
1. [APP]    → Construir prompt con system message + declaraciones <|tool> ... <tool|>
2. [APP]    → Enviar prompt de usuario al modelo
3. [GEMMA]  → Genera respuesta; si necesita herramienta, emite <|tool_call>
4. [APP]    → Parser detecta <|tool_call> en output
5. [APP]    → Ejecuta la función localmente (API, DB, sensor, etc.)
6. [APP]    → Inyecta resultado en <|tool_result> al contexto
7. [GEMMA]  → Recibe resultado y genera respuesta final al usuario
```

**Restricciones operativas del flujo:**
- Limitar a ≤ 3 herramientas declaradas por prompt para no saturar el contexto en dispositivos con `ctx_size=4096`.
- Cada ciclo tool_call→tool_result consume ~200-500 tokens del presupuesto de contexto.
- Para flujos multi-step (agentes), implementar truncación de historial FIFO manteniendo siempre el system prompt y la última interacción.

---

## Referencias Factuales

| Fuente | URL |
|---|---|
| Google AI — Gemma 4 Model Card | https://ai.google.dev/gemma/docs/core/model_card_4 |
| Google AI — Deploy Gemma on Mobile | https://ai.google.dev/gemma/docs/integrations/mobile |
| Google AI Edge — LLM Inference API Android | https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android |
| Google AI Edge Gallery (GitHub) | https://github.com/google-ai-edge/gallery |
| Gemma 4 Function Calling Docs | https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4 |
| Android Developers — Gemma 4 AICore Preview | https://developer.android.com/blog/posts/announcing-gemma-4-in-the-ai-core-developer-preview |
| llama.cpp Build Docs | https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md |
| llama.cpp GPU on Android (Discussion) | https://github.com/ggml-org/llama.cpp/discussions/16606 |
| Unsloth — Gemma 4 Local Run Guide | https://unsloth.ai/docs/models/gemma-4 |
| Qualcomm — Snapdragon 7s Gen 2 Platform | https://www.qualcomm.com/smartphones/products/7-series/snapdragon-7s-gen-2-mobile-platform |
| Hugging Face — Gemma 4 E2B | https://huggingface.co/google/gemma-4-E2B |
| Gemma 4 Tool Calling Playbook | https://www.gemma4.app/playbooks/tool-calling |
