package com.localandro.gemma4e2b.inference

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import android.util.Log
import com.localandro.gemma4e2b.domain.repository.InferenceConfig
import com.localandro.gemma4e2b.domain.repository.InferenceRepository
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.channels.SendChannel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout
import java.util.concurrent.CopyOnWriteArraySet
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Exception thrown when the `:inference_engine` service process has crashed
 * or become unreachable over IPC.
 *
 * The [com.localandro.gemma4e2b.agent.ActionOrchestrator] catches this to
 * trigger FSM recovery (→ PLANNING) and automatic service reconnection
 * instead of treating the failure as a fatal error.
 */
class InferenceEngineDisconnectedException(
    message: String = "Inference engine process disconnected",
    cause: Throwable? = null
) : RuntimeException(message, cause)

/**
 * IPC-based implementation of [InferenceRepository] that delegates all
 * inference work to [InferenceEngineService] running in the
 * `:inference_engine` process.
 *
 * From the caller's perspective this behaves identically to the direct
 * [LiteRTLMInferenceRepository] — same interface, same [Flow]-based
 * streaming.  Under the hood it communicates via Android [Messenger] IPC
 * and transparently handles service binding, crash detection (via
 * [ServiceConnection.onServiceDisconnected]), and automatic reconnection.
 *
 * **Thread safety:** All public methods are safe to call from any coroutine
 * dispatcher.  The IPC reply handlers run on the main [Looper] and bridge
 * results into coroutine primitives ([CompletableDeferred], channel).
 *
 * @param context Application context used for [Context.bindService].
 */
class RemoteInferenceRepository(
    private val context: Context
) : InferenceRepository {

    companion object {
        private const val TAG = "RemoteInferenceRepo"

        /** Maximum time to wait for the service to connect after binding. */
        private const val BIND_TIMEOUT_MS = 15_000L

        /** Maximum time to wait for an [InferenceIpcProtocol.MSG_INITIALIZE_RESULT]. */
        private const val INIT_TIMEOUT_MS = 60_000L
    }

    // ── Connection state ─────────────────────────────────────────────

    /** Messenger for sending commands to the service. `null` when unbound. */
    @Volatile
    private var serviceMessenger: Messenger? = null

    /** Whether the remote engine has been successfully initialised. */
    @Volatile
    private var initialized = false

    /** Serialises bind/unbind operations. */
    private val bindMutex = Mutex()

    /** Signals when [ServiceConnection.onServiceConnected] fires. */
    @Volatile
    private var connectionDeferred = CompletableDeferred<Unit>()

    /** Whether [Context.bindService] has been called and not yet unbound. */
    private var isBound = false

    // ── Reconnection cache ───────────────────────────────────────────

    /** Cached model path from the last successful [initialize] call. */
    private var lastModelPath: String? = null

    /** Cached config from the last successful [initialize] call. */
    private var lastConfig: InferenceConfig? = null

    // ── Active stream tracking ───────────────────────────────────────

    /**
     * Thread-safe set of active stream channels.  When the service process
     * dies ([onServiceDisconnected]), every channel is closed with an
     * [InferenceEngineDisconnectedException] so that collectors in the
     * main process receive a prompt error instead of hanging forever.
     */
    private val activeChannels = CopyOnWriteArraySet<SendChannel<String>>()

    // ── ServiceConnection ────────────────────────────────────────────

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            Log.i(TAG, "Bound to InferenceEngineService")
            serviceMessenger = Messenger(service)
            if (!connectionDeferred.isCompleted) {
                connectionDeferred.complete(Unit)
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            Log.w(TAG, "InferenceEngineService disconnected — process crashed")
            serviceMessenger = null
            initialized = false

            // Fail all in-flight streams immediately.
            val error = InferenceEngineDisconnectedException(
                "Inference engine process crashed"
            )
            for (ch in activeChannels) {
                ch.close(error)
            }
            activeChannels.clear()

            // Prepare for a fresh reconnection attempt.
            connectionDeferred = CompletableDeferred()
        }
    }

    // ── InferenceRepository – Lifecycle ──────────────────────────────

    override suspend fun initialize(
        modelPath: String,
        config: InferenceConfig
    ): Result<Unit> = try {
        ensureConnected()

        val messenger = serviceMessenger
            ?: throw InferenceEngineDisconnectedException(
                "Service not connected after bind"
            )

        withTimeout(INIT_TIMEOUT_MS) {
            suspendCancellableCoroutine { cont ->
                val replyHandler = object : Handler(Looper.getMainLooper()) {
                    override fun handleMessage(msg: Message) {
                        if (msg.what != InferenceIpcProtocol.MSG_INITIALIZE_RESULT) return
                        val data = msg.data ?: Bundle.EMPTY
                        data.classLoader = javaClass.classLoader
                        val success = data.getBoolean(InferenceIpcProtocol.KEY_SUCCESS)
                        if (success) {
                            initialized = true
                            lastModelPath = modelPath
                            lastConfig = config
                            if (cont.isActive) cont.resume(Unit)
                        } else {
                            val error = data.getString(InferenceIpcProtocol.KEY_ERROR)
                                ?: "Initialization failed"
                            if (cont.isActive) cont.resumeWithException(RuntimeException(error))
                        }
                    }
                }

                val ipcMsg = Message.obtain(
                    null,
                    InferenceIpcProtocol.MSG_INITIALIZE
                ).apply {
                    replyTo = Messenger(replyHandler)
                    data = Bundle().apply {
                        putString(InferenceIpcProtocol.KEY_MODEL_PATH, modelPath)
                        putInt(InferenceIpcProtocol.KEY_MAX_TOKENS, config.maxTokens)
                        putInt(InferenceIpcProtocol.KEY_CONTEXT_SIZE, config.contextSize)
                        putFloat(InferenceIpcProtocol.KEY_TEMPERATURE, config.temperature)
                        putInt(InferenceIpcProtocol.KEY_TOP_K, config.topK)
                        putFloat(InferenceIpcProtocol.KEY_TOP_P, config.topP)
                        putBoolean(InferenceIpcProtocol.KEY_USE_GPU, config.useGpu)
                    }
                }

                try {
                    messenger.send(ipcMsg)
                } catch (e: RemoteException) {
                    if (cont.isActive) {
                        cont.resumeWithException(
                            InferenceEngineDisconnectedException(
                                "Failed to send initialize command", e
                            )
                        )
                    }
                }
            }
        }

        Result.success(Unit)
    } catch (e: CancellationException) {
        throw e // Never swallow coroutine cancellation
    } catch (e: Exception) {
        Result.failure(e)
    }

    override fun isInitialized(): Boolean =
        initialized && serviceMessenger != null

    override suspend fun release() {
        try {
            serviceMessenger?.let { m ->
                val msg = Message.obtain(null, InferenceIpcProtocol.MSG_RELEASE)
                try {
                    m.send(msg)
                } catch (_: RemoteException) { }
            }
        } finally {
            initialized = false
            if (isBound) {
                try {
                    context.unbindService(serviceConnection)
                } catch (_: IllegalArgumentException) { }
                isBound = false
            }
            serviceMessenger = null
        }
    }

    // ── InferenceRepository – Inference ──────────────────────────────

    override fun streamResponse(prompt: String): Flow<String> = callbackFlow {
        // If the service died since the last call, attempt auto-reconnect
        // and re-initialise the engine in the new process.
        if (serviceMessenger == null && lastModelPath != null) {
            Log.i(TAG, "Auto-reconnecting to inference engine…")
            try {
                ensureConnected()
                reinitializeEngine()
            } catch (e: Exception) {
                throw InferenceEngineDisconnectedException(
                    "Auto-reconnect failed", e
                )
            }
        }

        val messenger = serviceMessenger
            ?: throw InferenceEngineDisconnectedException("Service unavailable")

        // Register this channel for disconnect-based cleanup.
        activeChannels.add(channel)

        val replyHandler = object : Handler(Looper.getMainLooper()) {
            override fun handleMessage(msg: Message) {
                when (msg.what) {
                    InferenceIpcProtocol.MSG_STREAM_TOKEN -> {
                        val data = msg.data ?: Bundle.EMPTY
                        data.classLoader = javaClass.classLoader
                        val token = data.getString(
                            InferenceIpcProtocol.KEY_TOKEN
                        ) ?: ""
                        trySend(token)
                    }
                    InferenceIpcProtocol.MSG_STREAM_COMPLETE -> {
                        channel.close()
                    }
                    InferenceIpcProtocol.MSG_STREAM_ERROR -> {
                        val data = msg.data ?: Bundle.EMPTY
                        data.classLoader = javaClass.classLoader
                        val error = data.getString(
                            InferenceIpcProtocol.KEY_ERROR
                        ) ?: "Inference error"
                        channel.close(RuntimeException(error))
                    }
                }
            }
        }

        val requestMsg = Message.obtain(
            null,
            InferenceIpcProtocol.MSG_STREAM_REQUEST
        ).apply {
            replyTo = Messenger(replyHandler)
            data = Bundle().apply {
                putString(InferenceIpcProtocol.KEY_PROMPT, prompt)
            }
        }

        try {
            messenger.send(requestMsg)
        } catch (e: RemoteException) {
            throw InferenceEngineDisconnectedException(
                "Failed to send stream request — engine process may have crashed", e
            )
        }

        awaitClose {
            activeChannels.remove(channel)
        }
    }

    // ── Internal: connection management ──────────────────────────────

    /**
     * Ensures the service is bound and the [Messenger] is available.
     * If not yet connected, binds to [InferenceEngineService] and waits
     * up to [BIND_TIMEOUT_MS] for [ServiceConnection.onServiceConnected].
     */
    private suspend fun ensureConnected() {
        if (serviceMessenger != null) return

        bindMutex.withLock {
            // Double-check after acquiring the lock.
            if (serviceMessenger != null) return

            connectionDeferred = CompletableDeferred()

            val intent = Intent(context, InferenceEngineService::class.java)
            isBound = context.bindService(
                intent,
                serviceConnection,
                Context.BIND_AUTO_CREATE
            )

            if (!isBound) {
                throw InferenceEngineDisconnectedException(
                    "bindService returned false — cannot reach InferenceEngineService"
                )
            }

            withTimeout(BIND_TIMEOUT_MS) {
                connectionDeferred.await()
            }
        }
    }

    /**
     * Re-initialises the engine in a freshly-started service process
     * using the cached model path and config from the previous
     * [initialize] call.
     */
    private suspend fun reinitializeEngine() {
        val path = lastModelPath
            ?: throw InferenceEngineDisconnectedException(
                "No cached model path for reconnection"
            )
        val cfg = lastConfig ?: InferenceConfig()

        val result = initialize(path, cfg)
        if (result.isFailure) {
            throw InferenceEngineDisconnectedException(
                "Re-initialization after reconnect failed: " +
                    "${result.exceptionOrNull()?.message}",
                result.exceptionOrNull()
            )
        }
    }
}
