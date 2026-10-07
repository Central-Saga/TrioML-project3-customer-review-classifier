# ==========================================
# Customer Review Classification API
# ==========================================

import re
import html
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==========================================
# 1. Load Model dan TF-IDF
# ==========================================

MODEL_PATH = "logistic_regression_best.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)


# ==========================================
# 2. Setup Preprocessing
# ==========================================

# Stopword Bahasa Indonesia
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

# Kata negasi penting untuk klasifikasi sentimen
important_words = {
    "tidak",
    "bukan",
    "belum",
    "jangan",
    "kurang",
}

# Jangan hapus kata negasi
stopwords = stopwords - important_words


# Stemmer Bahasa Indonesia
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# ==========================================
# 3. Kamus Normalisasi
# ==========================================

normalization_dict = {
    # Bentuk "tidak"
    "gaada": "tidak ada",
    "gada": "tidak ada",
    "gabisa": "tidak bisa",
    "nggak": "tidak",
    "ngga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ga": "tidak",

    # Bentuk "sudah"
    "udh": "sudah",
    "udah": "sudah",
    "uda": "sudah",

    # Bentuk "belum"
    "blm": "belum",

    # Kata umum
    "yg": "yang",
    "sy": "saya",
    "lg": "lagi",

    # Kata umum lainnya
    "krn": "karena",
    "karna": "karena",
    "dgn": "dengan",
    "dr": "dari",
    "sm": "sama",
    "sma": "sama",
    "tp": "tapi",
    "tpi": "tapi",

    # Bentuk informal
    "bgt": "banget",
    "bangettt": "banget",
    "klo": "kalau",
    "kalo": "kalau",
    "aja": "saja",
    "pke": "pakai",
    "pake": "pakai",
    "smp": "sampai",
    "sampe": "sampai",
    "dpt": "dapat",
    "dapet": "dapat",
    "trs": "terus",
    "trus": "terus",

    # Terima kasih
    "thx": "terima kasih",
    "thanks": "terima kasih",
    "makasih": "terima kasih",
    "mksh": "terima kasih",
    "terimakasih": "terima kasih",
    "ty": "terima kasih",

    # Sapaan
    "sis": "kak",
    "bro": "kak",

    # Typo umum
    "cepet": "cepat",
    "sanagt": "sangat",
}


# ==========================================
# 4. Fungsi Preprocessing
# ==========================================

def clean_basic(text: str) -> str:
    """
    Cleaning dasar:
    - HTML entity
    - lowercase
    - URL
    - newline/tab
    - spasi berlebih
    """

    text = str(text)

    # HTML entity
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Hapus URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Hapus newline, tab, carriage return
    text = re.sub(
        r"[\r\n\t]+",
        " ",
        text
    )

    # Rapikan spasi
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def normalize_text(text: str) -> str:
    """
    Normalisasi kata informal/singkatan.
    """

    text = str(text)

    # Kata yang lebih panjang diproses lebih dahulu
    sorted_words = sorted(
        normalization_dict.keys(),
        key=len,
        reverse=True
    )

    for word in sorted_words:
        replacement = normalization_dict[word]

        pattern = (
            r"(?<!\w)"
            + re.escape(word)
            + r"(?!\w)"
        )

        text = re.sub(
            pattern,
            replacement,
            text
        )

    return text


def tokenize_text(text: str) -> list[str]:
    """
    Tokenisasi sederhana seperti pipeline training.
    """

    text = str(text)

    # Pisahkan tanda baca tertentu
    text = re.sub(
        r"([.,!?;:()])",
        r" \1 ",
        text
    )

    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Hapus stopword tetapi pertahankan kata negasi.
    """

    return [
        token
        for token in tokens
        if token not in stopwords
    ]


def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Stemming Bahasa Indonesia.
    """

    result = []

    for token in tokens:

        # Jangan stemming angka
        if re.fullmatch(
            r"\d+([.,]\d+)?",
            token
        ):
            result.append(token)
            continue

        # Jangan stemming tanda baca
        if not re.search(
            r"[a-zA-ZÀ-ÿ]",
            token
        ):
            result.append(token)
            continue

        stemmed = stemmer.stem(token)

        # Jika Sastrawi menghasilkan string kosong,
        # abaikan token tersebut
        if stemmed.strip():
            result.append(stemmed)

    return result


def preprocess_text(text: str) -> str:
    """
    Pipeline preprocessing yang mengikuti
    pipeline training.
    """

    # Cleaning dasar
    text = clean_basic(text)

    # Normalisasi
    text = normalize_text(text)

    # Tokenisasi
    tokens = tokenize_text(text)

    # Stopword removal
    tokens = remove_stopwords(tokens)

    # Stemming
    stemmed_tokens = stem_tokens(tokens)

    # Penanganan khusus "ok"
    if re.fullmatch(
        r"(ok\s*)+",
        normalize_text(clean_basic(text))
    ):
        stemmed_tokens = tokens

    # Gabungkan kembali
    final_text = " ".join(stemmed_tokens)

    # Hapus tanda baca/simbol
    final_text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        final_text
    )

    # Rapikan spasi
    final_text = re.sub(
        r"\s+",
        " ",
        final_text
    ).strip()

    return final_text


# ==========================================
# 5. FastAPI
# ==========================================

app = FastAPI(
    title="Customer Review Classification API",
    description="API klasifikasi sentimen customer review",
    version="1.0.0"
)


# ==========================================
# 6. CORS
# ==========================================

# Agar Android/app lain dapat mengakses API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 7. Request Model
# ==========================================

class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        description="Review pelanggan yang ingin diklasifikasikan"
    )


# ==========================================
# 8. Root Endpoint
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Customer Review Classification API",
        "status": "running",
        "model": "Logistic Regression",
        "model_parameter": "C=5.0"
    }


# ==========================================
# 9. Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict(request: ReviewRequest):

    original_review = request.review.strip()

    if not original_review:
        raise HTTPException(
            status_code=400,
            detail="Review tidak boleh kosong."
        )

    # Preprocessing
    processed_text = preprocess_text(
        original_review
    )

    # Cek hasil preprocessing
    if not processed_text:
        raise HTTPException(
            status_code=400,
            detail="Review tidak memiliki teks yang dapat diproses."
        )

    # TF-IDF
    text_tfidf = tfidf.transform(
        [processed_text]
    )

    # Prediksi
    prediction = model.predict(
        text_tfidf
    )[0]

    # Probabilitas
    probabilities = model.predict_proba(
        text_tfidf
    )[0]

    confidence = float(
        probabilities[prediction]
    )

    # Label
    sentiment = (
        "POSITIF"
        if prediction == 1
        else "NEGATIF"
    )

    return {
        "review": original_review,
        "processed_text": processed_text,
        "sentiment": sentiment,
        "label": int(prediction),
        "confidence": round(confidence, 4),
        "confidence_percent": round(
            confidence * 100,
            2
        )
    }