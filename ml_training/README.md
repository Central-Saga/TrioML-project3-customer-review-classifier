# Customer Review Classification — ML Training

## Overview

Folder ini berisi notebook untuk proses pengembangan model klasifikasi sentimen review pelanggan. Seluruh eksperimen ML didokumentasikan dalam satu notebook, mulai dari eksplorasi data hingga persiapan model untuk deployment Android.

## Notebook

`notebooks/customer_review_classification_ml_pipeline.ipynb`

Notebook mencakup:

1. Exploratory Data Analysis (EDA).
2. Pemeriksaan distribusi dataset dan label.
3. Baseline klasifikasi menggunakan TF-IDF dan Logistic Regression.
4. Tokenisasi teks untuk IndoBERT.
5. Fine-tuning IndoBERT untuk klasifikasi dua kelas.
6. Evaluasi model menggunakan metrik klasifikasi.
7. Pemilihan checkpoint terbaik.
8. Ekspor model ke ONNX dan quantization INT8 untuk deployment.

## Label Klasifikasi

| Label | Sentimen |
|---|---|
| 0 | Negatif |
| 1 | Positif |

## Model

Model berbasis `indobenchmark/indobert-base-p1` digunakan untuk klasifikasi sentimen.

Model hasil deployment disimpan pada project Android di:

`android/app/src/main/assets/model.onnx`

Tokenizer dan file konfigurasi yang diperlukan juga ditempatkan di folder `assets` Android.

## Environment

Eksperimen training dilakukan menggunakan Google Colab dengan akselerator GPU apabila tersedia.

Kebutuhan utama meliputi:

- Python
- pandas dan NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- ONNX dan ONNX Runtime

Versi library yang tepat mengikuti konfigurasi dan instalasi di notebook.

## Dataset

Notebook menggunakan dataset review yang telah disiapkan dan diberi label sentimen. Pastikan file dataset tersedia pada lokasi yang diminta oleh notebook sebelum menjalankan sel yang membaca data.

Dataset dan hasil pemrosesan dapat berukuran besar, sehingga tidak semuanya harus disimpan di repository Git.

## Deployment Android

Model ONNX yang sudah disiapkan digunakan oleh aplikasi Android melalui ONNX Runtime.

Alur inferensi:

1. Pengguna memasukkan teks review.
2. `TextTokenizer.kt` memproses teks menjadi input model.
3. `OnnxPredictor.kt` menjalankan inferensi dengan `model.onnx`.
4. Aplikasi menampilkan sentimen dan confidence.

Inferensi dijalankan secara lokal di Android sehingga aplikasi tidak memerlukan API untuk memprediksi sentimen.

## Struktur Repository

```text
CustomerReviewClassifier/
+-- android/
¦   +-- app/
¦       +-- src/main/assets/
¦           +-- model.onnx
¦           +-- vocab.txt
¦           +-- tokenizer.json
¦           +-- ...
+-- ml_training/
    +-- README.md
    +-- notebooks/
        +-- customer_review_classification_ml_pipeline.ipynb
```

## Catatan

Notebook merupakan dokumentasi eksperimen dan proses pengembangan model. Untuk mereproduksi training, gunakan dataset, konfigurasi, checkpoint, dan environment yang sesuai dengan eksperimen asli.

Model deployment berada di project Android dan dikelola menggunakan Git LFS.
