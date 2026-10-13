# Resignal: Employee Attrition Prediction

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

## ⚙️ Employee Attrition Prediction Model Development

### Prediction Model Preparation
Dalam tahap ini dilakukan persiapan pembangunan model prediksi untuk membuat strategi pembangunan model prediksi karyawan resign menggunakan **jupyter notebook**:
  1. **Data understanding**
  2. **Exploratory data analysis**
  3. **Baseline model evaluation**
  4. **Data preprocessing**
  5. **Training, hyperparameter tuning, and evaluation machine learning model**
  6. **Employee status prediction and feature importance**

### Prediction Model Development
Setelah dilakukan persiapan pembangunan model prediksi, proses pembangunan model prediksi karyawan resign dibungkus ke dalam modul-modul skrip python terstruktur untuk diintegrasikan secara langsung dengan layanan **FastAPI** dan **Streamlit**:
  1. `src/config.py`: Mengelola variabel konfigurasi, jalur folder (*paths*), dan parameter global proyek.
  2. `src/data.py`: Mengeksekusi proses impor dataset, pengecekan dataset, pembagian dataset, dan penghapusan feature yang tidak digunakan dalam model machine learning.
  3. `src/train.py`: Mengeksekusi evaluasi model baseline, data preprocessing, pelatihan, hyperparameter tuning, evaluasi, dan pemilihan model machine learning terbaik.
  4. `src/predict.py`: Menyediakan fungsi prediksi untuk memuat model terbaik dan memproses input data baru dari API.
  5. `app.py`: Mengeksekusi pencarian local feature importance menggunakan occlusion sensitivity pada page 6.

---

## 🗂️ Repository Structure

```text
Resignal-Employee-Attrition-Prediction
│
├── .github/workflows/ci.yml # Konfigurasi GitHub Actions untuk CI/CD pipeline
├── api
│   └── main.py              # Endpoint backend utama berbasis FastAPI
├── data
│   ├── employee_attrition_dataset.csv # Dataset mentah
│   ├── train_data.csv       # Data training hasil data.py
│   └── test_data.csv        # Data testing hasil data.py
├── model
│   └── best_model.pkl       # Model machine learning terbaik hasil train.py
├── notebook
│   └── employee_attrition_prediction.ipynb # Persiapan pembangunan model prediksi karyawan resign
├── src
│   ├── config.py            # Konfigurasi parameter dan variabel global
│   ├── crud.py              # Operasi database (create, read, & update database)
│   ├── data.py              # Skrip impor dataset, pengecekan dataset, pembagian dataset, dan penghapusan feature yang tidak digunakan dalam model machine learning
│   ├── database.py          # Konfigurasi koneksi database PostgreSQL (SQLAlchemy)
│   ├── predict.py           # Modul logika inferensi model prediksi
│   ├── table.py             # Definisi skema tabel database (ORM models)
│   └── train.py             # Skrip evaluasi model baseline, data preprocessing, pelatihan, hyperparameter tuning, evaluasi, dan pemilihan model machine learning terbaik
│
├── .gitignore               # Daftar file/folder yang diabaikan oleh Git
├── app.py                   # Frontend aplikasi utama berbasis Streamlit
├── Dockerfile               # Konfigurasi container image untuk aplikasi
├── docker-compose.yml       # Orkestrasi multi-container (PostgreSQL, FastAPI, Streamlit)
├── requirements.txt         # Daftar pustaka/dependencies Python
└── README.md                # Dokumentasi proyek
```

---

## 🚀 Local Deployment Running Guideline

### 1. Clone Repository
Buka terminal/command prompt, lalu jalankan perintah berikut untuk meng-clone repositori ini di device Anda:

```bash
git clone https://github.com/hassantaufiqurrahman/Resignal-Employee-Attrition-Prediction.git
cd Resignal-Employee-Attrition-Prediction
```

### 2. Build dan Running Container
```bash
docker compose up --build
```

### 3. Access Local Services
* Streamlit Frontend Application: http://localhost:8501
* FastAPI Backend Documentation (Swagger UI): http://localhost:8000/docs