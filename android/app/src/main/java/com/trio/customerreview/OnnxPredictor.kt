package com.trio.customerreview

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import kotlin.math.exp

data class PredictionResult(
    val label: String,
    val confidence: Float
)

class OnnxPredictor(
    private val context: Context,
    private val modelFileName: String = "model.onnx"
) : Closeable {

    private val environment: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private val sessionOptions =
        OrtSession.SessionOptions()

    private val session: OrtSession

    init {

        // ----------------------------------------------------
        // Konfigurasi thread agar penggunaan CPU tidak berlebihan
        // ----------------------------------------------------

        sessionOptions.setIntraOpNumThreads(2)
        sessionOptions.setInterOpNumThreads(1)

        // ----------------------------------------------------
        // Salin model dari assets ke internal storage
        // hanya jika belum pernah disalin.
        // ----------------------------------------------------

        val modelFile = copyModelToInternalStorage()

        // ----------------------------------------------------
        // Load model langsung dari PATH file.
        //
        // Ini menghindari readBytes() yang sebelumnya membaca
        // seluruh model 120 MB ke Java heap.
        // ----------------------------------------------------

        session = environment.createSession(
            modelFile.absolutePath,
            sessionOptions
        )
    }

    private fun copyModelToInternalStorage(): File {

        val targetFile = File(
            context.filesDir,
            modelFileName
        )

        // Kalau model sudah pernah dicopy,
        // tidak perlu copy lagi.
        if (targetFile.exists() && targetFile.length() > 0L) {
            return targetFile
        }

        // Jika ada file rusak/parsial, hapus dulu.
        if (targetFile.exists()) {
            targetFile.delete()
        }

        context.assets
            .open(modelFileName)
            .use { inputStream ->

                targetFile.outputStream()
                    .use { outputStream ->

                        val buffer = ByteArray(1024 * 1024)

                        var bytesRead: Int

                        while (
                            inputStream.read(buffer).also {
                                bytesRead = it
                            } != -1
                        ) {
                            outputStream.write(
                                buffer,
                                0,
                                bytesRead
                            )
                        }

                        outputStream.flush()
                    }
            }

        require(
            targetFile.exists() &&
                    targetFile.length() > 0L
        ) {
            "Model ONNX gagal disalin ke internal storage"
        }

        return targetFile
    }

    fun predict(
        inputIds: LongArray,
        attentionMask: LongArray,
        tokenTypeIds: LongArray
    ): PredictionResult {

        require(inputIds.isNotEmpty()) {
            "inputIds tidak boleh kosong"
        }

        require(
            inputIds.size == attentionMask.size
        ) {
            "inputIds dan attentionMask harus memiliki panjang yang sama"
        }

        require(
            inputIds.size == tokenTypeIds.size
        ) {
            "inputIds dan tokenTypeIds harus memiliki panjang yang sama"
        }

        val inputIdsTensor =
            OnnxTensor.createTensor(
                environment,
                arrayOf(inputIds)
            )

        val attentionMaskTensor =
            OnnxTensor.createTensor(
                environment,
                arrayOf(attentionMask)
            )

        val tokenTypeIdsTensor =
            OnnxTensor.createTensor(
                environment,
                arrayOf(tokenTypeIds)
            )

        try {

            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionMaskTensor,
                "token_type_ids" to tokenTypeIdsTensor
            )

            session.run(inputs).use { outputs ->

                val logits =
                    outputs[0].value as Array<FloatArray>

                require(
                    logits.isNotEmpty() &&
                            logits[0].size >= 2
                ) {
                    "Output logits model tidak valid"
                }

                val negativeLogit =
                    logits[0][0]

                val positiveLogit =
                    logits[0][1]

                val maxLogit =
                    maxOf(
                        negativeLogit,
                        positiveLogit
                    )

                val negativeExp =
                    exp(
                        (
                                negativeLogit -
                                        maxLogit
                                ).toDouble()
                    )

                val positiveExp =
                    exp(
                        (
                                positiveLogit -
                                        maxLogit
                                ).toDouble()
                    )

                val total =
                    negativeExp + positiveExp

                val negativeProbability =
                    (
                            negativeExp / total
                            ).toFloat()

                val positiveProbability =
                    (
                            positiveExp / total
                            ).toFloat()

                return if (
                    positiveProbability >=
                    negativeProbability
                ) {

                    PredictionResult(
                        label = "POSITIF",
                        confidence = positiveProbability
                    )

                } else {

                    PredictionResult(
                        label = "NEGATIF",
                        confidence = negativeProbability
                    )
                }
            }

        } finally {

            inputIdsTensor.close()
            attentionMaskTensor.close()
            tokenTypeIdsTensor.close()
        }
    }

    override fun close() {

        session.close()

        sessionOptions.close()
    }
}