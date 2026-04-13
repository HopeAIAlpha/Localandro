package com.localandro.gemma4e2b.inference

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.os.RemoteException
import android.util.Log
import com.localandro.gemma4e2b.domain.repository.InferenceConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

/**
 * Bound [Service] that hosts the LiteRT-LM inference engine in an **isolated
 * process** (`android:process=":inference_engine"` in the manifest).
 *
 * This service is the **sole owner** of the native JNI engine
 * (`liblitertlm_jni.so`).  The main process (UI + Orchestrator) never loads
 * the native library; it communicates exclusively through the Android
 * [Messenger] IPC protocol defined in [InferenceIpcProtocol].
 *
 * If this process crashes (OOM, SIGSEGV, …) the main process detects the
 * disconnection via [android.content.ServiceConnection.onServiceDisconnected]
 * and the [RemoteInferenceRepository] triggers automatic reconnection.
 */
class InferenceEngineService : Service() {

    companion object {
        private const val TAG = "InferenceEngineSvc"
    }

    /** The actual inference engine, running exclusively in this process. */
    private lateinit var inferenceRepo: LiteRTLMInferenceRepository

    /** Coroutine scope for asynchronous work inside this service process. */
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** [Messenger] exposed to clients via [onBind]. */
    private lateinit var messenger: Messenger

    // ── Lifecycle ────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Service created — PID ${Process.myPid()}")
        inferenceRepo = LiteRTLMInferenceRepository(applicationContext)
        messenger = Messenger(IncomingHandler(this))
    }

    override fun onBind(intent: Intent?): IBinder {
        Log.i(TAG, "Client bound")
        return messenger.binder
    }

    override fun onDestroy() {
        Log.i(TAG, "Service destroyed — releasing engine")
        serviceScope.cancel()
        runBlocking {
            try {
                inferenceRepo.release()
            } catch (e: Exception) {
                Log.w(TAG, "Error releasing engine on destroy: ${e.message}")
            }
        }
        super.onDestroy()
    }

    // ── Message handler ──────────────────────────────────────────────

    /**
     * Handles incoming [Message]s from the main-process
     * [RemoteInferenceRepository].  Heavy work (initialisation, inference)
     * is dispatched to [serviceScope] so the handler thread is never blocked.
     */
    private class IncomingHandler(
        private val service: InferenceEngineService
    ) : Handler(Looper.getMainLooper()) {

        override fun handleMessage(msg: Message) {
            when (msg.what) {
                InferenceIpcProtocol.MSG_INITIALIZE -> handleInitialize(msg)
                InferenceIpcProtocol.MSG_STREAM_REQUEST -> handleStreamRequest(msg)
                InferenceIpcProtocol.MSG_RELEASE -> handleRelease()
                InferenceIpcProtocol.MSG_IS_INITIALIZED -> handleIsInitialized(msg)
                else -> super.handleMessage(msg)
            }
        }

        // ── MSG_INITIALIZE ──────────────────────────────────────────

        private fun handleInitialize(msg: Message) {
            val replyTo = msg.replyTo ?: return
            val data = msg.data ?: Bundle.EMPTY
            data.classLoader = javaClass.classLoader

            val modelPath = data.getString(InferenceIpcProtocol.KEY_MODEL_PATH) ?: run {
                sendReply(
                    replyTo,
                    InferenceIpcProtocol.MSG_INITIALIZE_RESULT,
                    success = false,
                    error = "Missing model path"
                )
                return
            }

            val config = InferenceConfig(
                maxTokens = data.getInt(InferenceIpcProtocol.KEY_MAX_TOKENS, 512),
                contextSize = data.getInt(InferenceIpcProtocol.KEY_CONTEXT_SIZE, 4096),
                temperature = data.getFloat(InferenceIpcProtocol.KEY_TEMPERATURE, 0.6f),
                topK = data.getInt(InferenceIpcProtocol.KEY_TOP_K, 40),
                topP = data.getFloat(InferenceIpcProtocol.KEY_TOP_P, 0.9f),
                useGpu = data.getBoolean(InferenceIpcProtocol.KEY_USE_GPU, true)
            )

            service.serviceScope.launch {
                val result = service.inferenceRepo.initialize(modelPath, config)
                sendReply(
                    replyTo,
                    InferenceIpcProtocol.MSG_INITIALIZE_RESULT,
                    success = result.isSuccess,
                    error = result.exceptionOrNull()?.localizedMessage
                )
            }
        }

        // ── MSG_STREAM_REQUEST ──────────────────────────────────────

        private fun handleStreamRequest(msg: Message) {
            val replyTo = msg.replyTo ?: return
            val data = msg.data ?: Bundle.EMPTY
            data.classLoader = javaClass.classLoader

            val prompt = data.getString(InferenceIpcProtocol.KEY_PROMPT) ?: run {
                sendStreamError(replyTo, "Missing prompt")
                return
            }

            service.serviceScope.launch {
                try {
                    service.inferenceRepo.streamResponse(prompt)
                        .catch { e ->
                            Log.e(TAG, "Stream error", e)
                            sendStreamError(replyTo, e.localizedMessage ?: "Inference error")
                        }
                        .collect { token ->
                            val tokenMsg = Message.obtain(
                                null,
                                InferenceIpcProtocol.MSG_STREAM_TOKEN
                            )
                            tokenMsg.data = Bundle().apply {
                                putString(InferenceIpcProtocol.KEY_TOKEN, token)
                            }
                            try {
                                replyTo.send(tokenMsg)
                            } catch (e: RemoteException) {
                                Log.w(TAG, "Client disconnected during streaming", e)
                                throw CancellationException("Client disconnected")
                            }
                        }

                    // Signal stream completion.
                    val completeMsg = Message.obtain(
                        null,
                        InferenceIpcProtocol.MSG_STREAM_COMPLETE
                    )
                    try {
                        replyTo.send(completeMsg)
                    } catch (_: RemoteException) {
                        Log.w(TAG, "Client disconnected before completion signal")
                    }
                } catch (e: CancellationException) {
                    throw e // Propagate cancellation normally
                } catch (e: Exception) {
                    Log.e(TAG, "Unhandled stream error", e)
                    sendStreamError(replyTo, e.localizedMessage ?: "Inference error")
                }
            }
        }

        // ── MSG_RELEASE ─────────────────────────────────────────────

        private fun handleRelease() {
            service.serviceScope.launch {
                try {
                    service.inferenceRepo.release()
                } catch (e: Exception) {
                    Log.w(TAG, "Error releasing engine: ${e.message}")
                }
            }
        }

        // ── MSG_IS_INITIALIZED ──────────────────────────────────────

        private fun handleIsInitialized(msg: Message) {
            val replyTo = msg.replyTo ?: return
            val reply = Message.obtain(
                null,
                InferenceIpcProtocol.MSG_IS_INITIALIZED_RESULT
            )
            reply.data = Bundle().apply {
                putBoolean(
                    InferenceIpcProtocol.KEY_RESULT,
                    service.inferenceRepo.isInitialized()
                )
            }
            try {
                replyTo.send(reply)
            } catch (_: RemoteException) { }
        }

        // ── Helpers ─────────────────────────────────────────────────

        private fun sendReply(
            replyTo: Messenger,
            what: Int,
            success: Boolean,
            error: String? = null
        ) {
            val reply = Message.obtain(null, what)
            reply.data = Bundle().apply {
                putBoolean(InferenceIpcProtocol.KEY_SUCCESS, success)
                if (error != null) {
                    putString(InferenceIpcProtocol.KEY_ERROR, error)
                }
            }
            try {
                replyTo.send(reply)
            } catch (_: RemoteException) {
                Log.w(TAG, "Client disconnected before reply")
            }
        }

        private fun sendStreamError(replyTo: Messenger, error: String) {
            val errorMsg = Message.obtain(null, InferenceIpcProtocol.MSG_STREAM_ERROR)
            errorMsg.data = Bundle().apply {
                putString(InferenceIpcProtocol.KEY_ERROR, error)
            }
            try {
                replyTo.send(errorMsg)
            } catch (_: RemoteException) {
                Log.w(TAG, "Client disconnected before error reply")
            }
        }
    }
}
