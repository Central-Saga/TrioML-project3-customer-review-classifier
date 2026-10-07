# ==========================================
# Dockerfile - Customer Review Classification API
# ==========================================

# Gunakan Python 3.13 agar kompatibel dengan
# scikit-learn 1.6.1 yang digunakan saat training model
FROM python:3.13-slim

# Supaya output Python langsung muncul di terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Folder kerja di dalam container
WORKDIR /app

# Copy daftar dependency
COPY requirements.txt .

# Install dependency
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code dan model
COPY app.py .
COPY logistic_regression_best.pkl .
COPY tfidf_vectorizer.pkl .

# Port FastAPI
EXPOSE 8000

# Jalankan FastAPI menggunakan Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]