package com.localandro.gemma4e2b.download

import android.content.Context
import com.localandro.gemma4e2b.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Manages authenticated download of the Gemma 4 E2B model from Hugging Face.
 *
 * Executes on [Dispatchers.IO] and emits [DownloadState] updates with real
 * progress (0–100 %) so the UI can display a determinate progress bar.
 *
 * The downloaded file is saved directly into [Context.getFilesDir] to comply
 * with Android Scoped Storage restrictions.
 */
class ModelDownloadManager(private val context: Context) {

    companion object {
        /** Public Hugging Face URL for the Gemma 4 E2B LiteRT-LM task file. */
        const val MODEL_URL =
            "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task?download=true"

        /** Local file name stored inside filesDir. */
        const val MODEL_FILE_NAME = "gemma-4-E2B-it-web.task"

        private const val BUFFER_SIZE = 8 * 1024 // 8 KB
    }

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.MINUTES)
        .writeTimeout(5, TimeUnit.MINUTES)
        .followRedirects(true)
        .followSslRedirects(true)
        .build()

    /** Returns the expected model [File] inside the app's private storage. */
    fun getModelFile(): File = File(context.filesDir, MODEL_FILE_NAME)

    /** Returns `true` when the model file already exists on disk. */
    fun isModelDownloaded(): Boolean = getModelFile().exists() && getModelFile().length() > 0

    /**
     * Starts the authenticated download and emits [DownloadState] updates.
     *
     * The Hugging Face token is read from [BuildConfig.HF_TOKEN], which is
     * injected at build time via the `HF_TOKEN` environment variable.
     */
    fun download(): Flow<DownloadState> = flow {
        emit(DownloadState.Starting)

        val destination = getModelFile()
        val tempFile = File(context.filesDir, "$MODEL_FILE_NAME.tmp")

        try {
            val request = Request.Builder()
                .url(MODEL_URL)
                .addHeader("Authorization", "Bearer ${BuildConfig.HF_TOKEN}")
                .build()

            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                emit(DownloadState.Error("HTTP ${response.code}: ${response.message}"))
                return@flow
            }

            val body = response.body ?: run {
                emit(DownloadState.Error("Empty response body"))
                return@flow
            }

            val contentLength = body.contentLength()
            var bytesRead: Long = 0

            body.byteStream().use { input ->
                tempFile.outputStream().buffered().use { output ->
                    val buffer = ByteArray(BUFFER_SIZE)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        bytesRead += read

                        val progress = if (contentLength > 0) {
                            ((bytesRead * 100) / contentLength).toInt().coerceIn(0, 100)
                        } else {
                            -1 // Indeterminate
                        }
                        emit(DownloadState.Progress(progress, bytesRead, contentLength))
                    }
                }
            }

            // Atomic rename: only overwrite if download completed successfully.
            if (!tempFile.renameTo(destination)) {
                tempFile.copyTo(destination, overwrite = true)
                tempFile.delete()
            }

            emit(DownloadState.Completed(destination.absolutePath))
        } catch (e: IOException) {
            tempFile.delete()
            emit(DownloadState.Error("Download failed: ${e.localizedMessage}"))
        } catch (e: Exception) {
            tempFile.delete()
            emit(DownloadState.Error("Unexpected error: ${e.localizedMessage}"))
        }
    }.flowOn(Dispatchers.IO)
}

/** Represents the state of the model download operation. */
sealed class DownloadState {
    data object Starting : DownloadState()

    data class Progress(
        /** Percentage 0–100, or -1 when content-length is unknown. */
        val percent: Int,
        val bytesDownloaded: Long,
        val totalBytes: Long
    ) : DownloadState()

    data class Completed(val filePath: String) : DownloadState()

    data class Error(val message: String) : DownloadState()
}
