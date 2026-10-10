FROM python:3.11-slim

WORKDIR /app

# Install Dependensi Sistem yang Dibutuhkan untuk psycopg2
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Salin dan Instal Library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin Seluruh Kode Proyek ke dalam Container
COPY . .

# Port Default FastAPI
EXPOSE 8000

# Perintah untuk Menjalankan FastAPI Menggunakan uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]