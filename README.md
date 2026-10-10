# Resignal: Employee Attrition Prediction & Decision Support System

**Final Project Dibimbing Data Science & AI Machine Learning Batch 42_Hassan Taufiqurrahman**

> **Resignal** adalah sistem berbasis model *Machine Learning* dan *Web Application* end-to-end yang dirancang untuk memprediksi risiko karyawan *resign* (*attrition*) secara presisi, dilengkapi dengan analisis penjelasan model (*Occlusion Sensitivity*) serta penyimpanan riwayat data berbasis relasional database.

---

## 📌 Project Overview

Proyek ini dibangun untuk melakukan **deteksi dini (*early warning system*)** terhadap potensi pengunduran diri karyawan berdasarkan data historis. Alur kerja utama proyek meliputi:
1. **Data Preprocessing & Training:** Melakukan pembersihan data, penanganan ketidakseimbangan data, dan pelatihan model klasifikasi hingga menghasilkan model terbaik (`best_model.pkl`).
2. **Backend API (FastAPI):** Menyediakan titik akses *endpoint* yang memproses inferensi *machine learning* secara *real-time*.
3. **Database Integration (PostgreSQL):** Menyimpan setiap input data dan riwayat hasil prediksi secara permanen menggunakan kontainer basis data relasional.
4. **Interactive Frontend (Streamlit):** Menyediakan antarmuka pengguna berbasis web yang interaktif dengan navigasi multi-langkah dan laporan visual yang mudah dipahami.
5. **Containerization (Docker Compose):** Menyatukan seluruh layanan (*database, backend, frontend*) agar mudah dijalankan dan di-deploy secara konsisten.

---

## 🗂️ Repository Structure

```text
Final Project Dibimbing DS/
│
├── .github/workflows/ci.yml # Konfigurasi GitHub Actions untuk CI/CD pipeline
├── api/
│   └── main.py              # Endpoint backend utama berbasis FastAPI
├── data/
│   ├── employee_attrition_dataset.csv # Dataset mentah keseluruhan
│   ├── train_data.csv       # Dataset train hasil data.py
│   └── test_data.csv        # Dataset test hasil data.py
├── models/
│   └── best_model.pkl       # Model Machine Learning terbaik hasil train.py
├── notebook/
│   └── employee_attrition_prediction.ipynb # Jupyter Notebook eksplorasi & modeling
├── src/
│   ├── config.py            # Konfigurasi parameter dan variabel global
│   ├── crud.py              # Operasi database (create, read, & update database)
│   ├── data.py              # Skrip pemrosesan awal dan pembersihan data
│   ├── database.py          # Konfigurasi koneksi database PostgreSQL (SQLAlchemy)
│   ├── predict.py           # Modul logika inferensi model
│   ├── table.py             # Definisi skema tabel database (ORM models)
│   └── train.py             # Skrip data preprocessing, pelatihan. hyperparameter tuning, dan evaluasi model Machine Learning
│
├── .gitignore               # Daftar file/folder yang diabaikan oleh Git
├── app.py                   # Frontend aplikasi utama berbasis Streamlit
├── Dockerfile               # Konfigurasi container image untuk aplikasi
├── docker-compose.yml       # Orkestrasi multi-container (PostgreSQL, FastAPI, Streamlit)
├── requirements.txt         # Daftar pustaka/dependencies Python
└── README.md                # Dokumentasi proyek
```

---

## 🚀 How to Run (Local Deployment via Docker)

### 1. Clone Repository
Buka terminal/command prompt, lalu jalankan perintah berikut untuk meng-clone repositori ini di device Anda:

```bash
git clone https://github.com/hassantaufiqurrahman/Resignal-Employee-Attrition-Prediction.git
cd Resignal-Employee-Attrition-Prediction
```

### 2. Build dan Jalankan Container
```bash
docker-compose up --build -d
```

### 3. Akses Layanan Lokal
* Streamlit Frontend Application: http://localhost:8501
* FastAPI Interactive Documentation (Swagger UI): http://localhost:8000/docs