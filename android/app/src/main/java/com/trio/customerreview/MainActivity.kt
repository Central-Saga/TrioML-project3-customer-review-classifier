package com.trio.customerreview

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var etReview: EditText
    private lateinit var btnPredict: Button
    private lateinit var tvSentiment: TextView
    private lateinit var tvConfidence: TextView

    // Tokenizer membaca vocab.txt dari assets.
    private lateinit var tokenizer: TextTokenizer

    // Predictor akan memuat model ONNX saat pertama kali digunakan.
    private var onnxPredictor: OnnxPredictor? = null

    // Jalankan inference di background thread agar UI tidak freeze.
    private val executor: ExecutorService =
        Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        // ----------------------------------------------------
        // Hubungkan View dari XML
        // ----------------------------------------------------

        etReview = findViewById(R.id.etReview)
        btnPredict = findViewById(R.id.btnPredict)
        tvSentiment = findViewById(R.id.tvSentiment)
        tvConfidence = findViewById(R.id.tvConfidence)

        // ----------------------------------------------------
        // Inisialisasi tokenizer
        // ----------------------------------------------------

        try {
            tokenizer = TextTokenizer(
                context = this,
                vocabFileName = "vocab.txt",
                maxLength = 128
            )

        } catch (e: Exception) {

            tvSentiment.text = "ERROR"

            tvConfidence.text =
                e.message ?: "Tokenizer gagal dimuat"

            btnPredict.isEnabled = false

            return
        }

        // ----------------------------------------------------
        // Tombol prediksi
        // ----------------------------------------------------

        btnPredict.setOnClickListener {
            predictReview()
        }
    }

    private fun predictReview() {

        val review = etReview.text
            .toString()
            .trim()

        // ----------------------------------------------------
        // Validasi input
        // ----------------------------------------------------

        if (review.isEmpty()) {

            etReview.error =
                "Review tidak boleh kosong"

            etReview.requestFocus()

            return
        }

        // ----------------------------------------------------
        // Ubah UI saat proses
        // ----------------------------------------------------

        btnPredict.isEnabled = false
        btnPredict.text = "MEMPROSES..."

        tvSentiment.text = "Memproses..."
        tvConfidence.text =
            "Model sedang melakukan prediksi offline"

        // ----------------------------------------------------
        // Jalankan inference di background
        // ----------------------------------------------------

        executor.execute {

            try {

                // ------------------------------------------------
                // Load ONNX predictor saat pertama kali dipakai
                // ------------------------------------------------

                val predictor =
                    onnxPredictor ?: synchronized(this) {

                        onnxPredictor
                            ?: OnnxPredictor(
                                context = this
                            ).also {
                                onnxPredictor = it
                            }
                    }

                // ------------------------------------------------
                // Tokenisasi review
                // ------------------------------------------------

                val encoded =
                    tokenizer.encode(review)

                // ------------------------------------------------
                // Jalankan model
                // ------------------------------------------------

                val result =
                    predictor.predict(
                        inputIds = encoded.inputIds,
                        attentionMask = encoded.attentionMask,
                        tokenTypeIds = encoded.tokenTypeIds
                    )

                // ------------------------------------------------
                // Update UI
                // ------------------------------------------------

                runOnUiThread {

                    tvSentiment.text =
                        result.label

                    tvConfidence.text =
                        "Confidence: %.2f%%".format(
                            result.confidence * 100f
                        )

                    btnPredict.isEnabled = true
                    btnPredict.text =
                        "PREDIKSI SENTIMEN"
                }

            } catch (e: Exception) {

                // ------------------------------------------------
                // Tangani error inference
                // ------------------------------------------------

                runOnUiThread {

                    tvSentiment.text =
                        "ERROR"

                    tvConfidence.text =
                        e.message
                            ?: "Gagal menjalankan model"

                    btnPredict.isEnabled = true
                    btnPredict.text =
                        "PREDIKSI SENTIMEN"
                }
            }
        }
    }

    override fun onDestroy() {

        // Hentikan background executor
        executor.shutdownNow()

        // Tutup ONNX session
        onnxPredictor?.close()
        onnxPredictor = null

        super.onDestroy()
    }
}