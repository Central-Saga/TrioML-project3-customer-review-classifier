package com.trio.customerreview

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

class MainActivity : AppCompatActivity() {

    private lateinit var etReview: EditText
    private lateinit var btnPredict: Button
    private lateinit var tvSentiment: TextView
    private lateinit var tvConfidence: TextView

    // Android Emulator menggunakan 10.0.2.2
    // untuk mengakses localhost komputer host.
    private val apiUrl = "http://10.0.2.2:8000/predict"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Hubungkan View dari XML
        etReview = findViewById(R.id.etReview)
        btnPredict = findViewById(R.id.btnPredict)
        tvSentiment = findViewById(R.id.tvSentiment)
        tvConfidence = findViewById(R.id.tvConfidence)

        // Tombol prediksi
        btnPredict.setOnClickListener {
            predictReview()
        }
    }

    private fun predictReview() {

        val review = etReview.text.toString().trim()

        // Validasi input
        if (review.isEmpty()) {
            etReview.error = "Review tidak boleh kosong"
            etReview.requestFocus()
            return
        }

        // Ubah tampilan tombol saat proses
        btnPredict.isEnabled = false
        btnPredict.text = "Memproses..."

        tvSentiment.text = "Memproses..."
        tvConfidence.text = "Mohon tunggu"

        Thread {

            var connection: HttpURLConnection? = null

            try {

                // Membuka koneksi ke API
                val url = URL(apiUrl)
                connection = url.openConnection() as HttpURLConnection

                connection.requestMethod = "POST"
                connection.connectTimeout = 10000
                connection.readTimeout = 10000

                connection.setRequestProperty(
                    "Content-Type",
                    "application/json"
                )

                connection.setRequestProperty(
                    "Accept",
                    "application/json"
                )

                connection.doOutput = true

                // Escape string JSON dengan JSONObject
                val jsonBody = JSONObject()
                jsonBody.put("review", review)

                // Kirim request ke API
                connection.outputStream.use { output ->
                    output.write(
                        jsonBody.toString().toByteArray(Charsets.UTF_8)
                    )
                    output.flush()
                }

                // Ambil status HTTP
                val responseCode = connection.responseCode

                val inputStream =
                    if (responseCode in 200..299) {
                        connection.inputStream
                    } else {
                        connection.errorStream
                    }

                val response = BufferedReader(
                    InputStreamReader(
                        inputStream,
                        Charsets.UTF_8
                    )
                ).use { reader ->
                    reader.readText()
                }

                // Jika request berhasil
                if (responseCode in 200..299) {

                    val jsonResponse = JSONObject(response)

                    val sentiment =
                        jsonResponse.getString("sentiment")

                    val confidence =
                        jsonResponse.getDouble("confidence_percent")

                    runOnUiThread {

                        tvSentiment.text = sentiment

                        tvConfidence.text =
                            "Confidence: %.2f%%".format(confidence)

                        btnPredict.isEnabled = true
                        btnPredict.text = "PREDIKSI SENTIMEN"
                    }

                } else {

                    // Jika API mengembalikan error
                    runOnUiThread {

                        tvSentiment.text = "ERROR"

                        tvConfidence.text =
                            "API mengembalikan error ($responseCode)"

                        btnPredict.isEnabled = true
                        btnPredict.text = "PREDIKSI SENTIMEN"
                    }
                }

            } catch (e: Exception) {

                // Jika tidak dapat terhubung ke API
                runOnUiThread {

                    tvSentiment.text = "GAGAL TERHUBUNG"

                    tvConfidence.text =
                        e.message ?: "Tidak dapat mengakses API"

                    btnPredict.isEnabled = true
                    btnPredict.text = "PREDIKSI SENTIMEN"
                }

            } finally {

                connection?.disconnect()
            }
        }.start()
    }
}