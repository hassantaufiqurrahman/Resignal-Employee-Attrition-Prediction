import os
import pandas as pd
from pycaret.classification import *
from src.config import TARGET_COL, SESSION_ID, MODELS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH
from src.data import get_train_test_data

# Function untuk Pembuatan Feature Baru
def create_features(df):
    df = df.copy()

    # Financial Ratios
    df["Income_Per_Year_At_Company"] = (
        df["Monthly_Income"] * 12 / (df["Years_At_Company"] + 1)
    )
    df["Income_Per_Project"] = df["Monthly_Income"] / (
        df["Number_Of_Projects"]
    )
    df["Income_Per_Age"] = df["Monthly_Income"] / df["Age"]

    # Career Ratios
    df["Experience_Ratio"] = df["Years_In_Current_Role"] / (
        df["Years_At_Company"] + 1
    )

    # Aggregasi Satisfaction
    sat_cols = [
        "Job_Satisfaction",
        "Work_Life_Balance",
        "Environment_Satisfaction",
        "Relationship_Satisfaction",
    ]
    df["Avg_Satisfaction"] = df[sat_cols].mean(axis=1)

    # Workload & Performance
    df["Workload_Index"] = df["Number_Of_Projects"] * df["Team_Size"]
    df["Projects_Per_Team_Member"] = df["Number_Of_Projects"] / (
        df["Team_Size"]
    )

    # Commute
    df["Commute_Speed_Index"] = df["Distance_From_Home"] / (
        df["Commute_Time_Minutes"]
    )

    return df

# Function untuk Data Preprocessing, Pelatihan, & Tuning 3 Model Terbaik
def run_training():
    # Load "train_data" dan "test_data"
    print("Memeriksa dan menyiapkan file split data di folder data/...")
    get_train_test_data()
    print(f"Memuat train_data dari: {TRAIN_DATA_PATH}")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Memuat test_data dari: {TEST_DATA_PATH}")
    test_data = pd.read_csv(TEST_DATA_PATH)
    
    print(f"Berhasil dimuat! Jumlah baris train_data: {len(train_data)}, test_data: {len(test_data)}")

    # Melakukan Feature Engineering
    print("Melakukan feature engineering...")
    train_data = create_features(train_data)
    test_data = create_features(test_data)

    # Melakukan Preprocessing, Pelatihan, dan Evaluasi Awal Model
    print("\nMelakukan Preprocessing, Pelatihan, dan Evaluasi Awal Model...")
    clf_setup = setup(
        data=train_data,
        test_data=test_data,
        target=TARGET_COL,
        index=False,
        normalize=True,
        normalize_method='zscore',
        fix_imbalance=True,
        session_id=SESSION_ID,
        verbose=True
    )

    # Membandingkan & Mengambil 3 Model Teratas
    print("\n--- Membandingkan Model & Mengambil 3 Model Teratas (Top-3) ---")
    top_3_models = compare_models(n_select=3, sort='AUC')

    # Melakukan Hyperparameter Tuning pada Ketiga Model Teratas
    print("\nMelakukan hyperparameter tuning pada ketiga model teratas...")
    tuned_top_3_models = []
    
    for i, model in enumerate(top_3_models, start=1):
        model_name = type(model).__name__
        print(f"\n--- Sedang Tuning Model ke-{i}: {model_name} ---")
        
        try:
            # Tuning dengan "n_iter"
            tuned_model = tune_model(
                model,
                optimize='AUC',
                n_iter=30,
                choose_better=True
            )
        except Exception as e:
            # Fallback jika model tidak mendukung "n_iter" 
            print(f"Catatan: tune_model dengan n_iter gagal/tidak didukung untuk {model_name}. Menggunakan default n_iter...")
            tuned_model = tune_model(
                model,
                optimize='AUC',
                choose_better=True
            )
            
        tuned_top_3_models.append(tuned_model)
        
        # Menampilkan Parameter Terbaik
        print(f"\nParameter Terbaik untuk {model_name}:")
        if hasattr(tuned_model, "get_params"):
            for param, val in list(tuned_model.get_params().items())[:12]:
                print(f"  - {param}: {val}")

    print("\nProses training dan hyperparameter tuning 3 model terbaik selesai!")
    return tuned_top_3_models

# Function untuk Evaluasi Akhir Model & Penyimpanan
def evaluate_and_save_models():
    print("Memanggil proses training & tuning dari run_training()...")
    tuned_top_3_models = run_training()

    best_recall = -1.0
    best_model_obj = None
    best_model_name = ""
    evaluated_results = []

    print("\n" + "="*50)
    print("--- Evaluasi Lengkap TOP-3 Model Pada Data Testing ---")
    print("="*50)

    for i, model in enumerate(tuned_top_3_models, start=1):
        model_name = type(model).__name__
        print(f"\n[Model ke-{i}: {model_name}]")

        # Prediksi Pada Test Data
        predictions = predict_model(model)
        
        # Mengambil Tabel Metrik Hasil Evaluasi Data Testing
        eval_metrics = pull()
        
        # Mengambil Nilai Recall Secara Aman Terlepas dari Format Struktur Data "Pull()""
        try:
            if 'Recall' in eval_metrics.columns:
                recall_score = eval_metrics['Recall'].iloc[0]
            elif 'Recall' in eval_metrics.index:
                recall_score = eval_metrics.loc['Recall'].values[0]
            else:
                recall_score = eval_metrics.iloc[0]['Recall']
        except Exception:
            recall_score = eval_metrics.iloc[-1, -1]

        print(f"-> Skor Recall Tercatat: {recall_score:.4f}")

        evaluated_results.append({
            "model_object": model,
            "model_name": model_name,
            "recall": recall_score
        })

        # Memilih Model Terbaik Berdasarkan Nilai Recall Tertinggi
        if recall_score > best_recall:
            best_recall = recall_score
            best_model_obj = model
            best_model_name = model_name

    print("\n" + "="*50)
    print(f"[HASIL Akhir Evaluasi Model]")
    print(f"Model Terbaik Berdasarkan Recall Tertinggi: {best_model_name} (Recall: {best_recall:.4f})")
    print("="*50)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Mengurutkan Model Berdasarkan Perolehan Recall Tertinggi ke Terendah
    evaluated_results = sorted(evaluated_results, key=lambda x: x['recall'], reverse=True)

    # Menyimpan Model Terbaik ke Folder "models"
    print("\nMenyimpan model terbaik ke dalam folder models/...")

    best_res = evaluated_results[0]
    model_obj = best_res['model_object']

    save_path = os.path.join(MODELS_DIR, "best_model")
    print(f"-> Menyimpan Model Terbaik ({best_res['model_name']} - Recall: {best_res['recall']:.4f}) sebagai: best_model.pkl")

    clean_save_path = save_path.replace(".pkl", "") if save_path.endswith(".pkl") else save_path

    save_model(model_obj, model_name=clean_save_path, verbose=False)

    print("\nPenyimpanan file best_model.pkl berhasil diselesaikan!")

if __name__ == "__main__":
    evaluate_and_save_models()