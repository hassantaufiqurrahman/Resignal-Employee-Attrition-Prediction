import os
import pandas as pd
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.config import TARGET_COL, SESSION_ID, MODELS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH, DATASET_PATH, COLS_TO_DROP
from src.data import get_train_test_data

# Function untuk Feature Engineering Model Machine Learning
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

# Function untuk Evaluasi Model Baseline
def evaluate_baseline():
    print("\n" + "="*50)
    print("--- Menjalankan Evaluasi Model Baseline (Manual) ---")
    print("="*50)
    
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Berhasil memuat dataset baseline dari {DATASET_PATH}. Total baris awal: {len(df)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset tidak ditemukan untuk model baseline di path: {DATASET_PATH}")

    # Identifikasi dan Membersihkan Duplikat Jika Ada
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Ditemukan dan menghapus {duplicate_count} baris duplikat pada baseline.")

    # Split Dataset
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=SESSION_ID, 
        stratify=df[TARGET_COL]
    )
    
    print(f"Jumlah baris data training baseline : {train_df.shape}")
    print(f"Jumlah baris data testing baseline  : {test_df.shape}")

    # Hapus Feature yang Tidak Dipakai Pada Data Training & Testing
    id_cols_to_drop = [col for col in ['Employee_ID'] if col in train_df.columns]
    if id_cols_to_drop:
        train_df = train_df.drop(columns=id_cols_to_drop)
        test_df = test_df.drop(columns=id_cols_to_drop)
        print(f"Baseline: feature yang dihapus dari train dan test: {id_cols_to_drop}")

    # Menghitung Rata-rata "Attrition_Risk_Score" Kelompok Karyawan Resign Sebagai Threshold
    mean_resigned_score_train = train_df[train_df[TARGET_COL] == 'Yes']['Attrition_Risk_Score'].mean()
    threshold_used = round(mean_resigned_score_train, 2)
    print(f"Threshold Baseline (dihitung dari rata-rata data training): {threshold_used}")

    # Evaluasi Model Baseline pada Data Testing
    y_true_test = (test_df[TARGET_COL] == 'Yes').astype(int)
    y_pred_baseline = (test_df['Attrition_Risk_Score'] >= threshold_used).astype(int)

    # Menghitung Metrik Evaluasi Model Baseline
    acc = accuracy_score(y_true_test, y_pred_baseline)
    prec = precision_score(y_true_test, y_pred_baseline, zero_division=0)
    baseline_recall = recall_score(y_true_test, y_pred_baseline, zero_division=0)
    f1 = f1_score(y_true_test, y_pred_baseline, zero_division=0)
    roc_auc = roc_auc_score(y_true_test, test_df['Attrition_Risk_Score'] / 100.0)

    print("\n--- Hasil Evaluasi Model Baseline ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {baseline_recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("="*50)
    
    return baseline_recall

# Function untuk Data Preprocessing, Pelatihan, & Tuning 3 Model Machine Learning Terbaik (PyCaret)
def run_training():
    print("Memeriksa dan menyiapkan file split data di folder data/... untuk model machine learning")
    get_train_test_data()
    
    print(f"Memuat train_data dari: {TRAIN_DATA_PATH}")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Memuat test_data dari: {TEST_DATA_PATH}")
    test_data = pd.read_csv(TEST_DATA_PATH)
    
    print(f"Berhasil dimuat! jumlah baris train_data: {len(train_data)}, test_data: {len(test_data)}")

    # Melakukan Feature Engineering Model Machine Learning
    print("Melakukan feature engineering untuk model machine learning...")
    train_data = create_features(train_data)
    test_data = create_features(test_data)

    # Melakukan Preprocessing, Pelatihan, dan Evaluasi Awal Model via PyCaret
    print("\nMelakukan data preprocessing, pelatihan, dan evaluasi awal model machine learning...")
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

    # Membandingkan & Mengambil 3 Model Machine Learning Teratas
    print("\n--- Membandingkan & mengambil 3 model machine learning teratas (Top-3) ---")
    top_3_models = compare_models(n_select=3, sort='AUC')

    # Melakukan Hyperparameter Tuning pada Ketiga Model Teratas
    print("\nMelakukan hyperparameter tuning pada ketiga model teratas...")
    tuned_top_3_models = []
    
    for i, model in enumerate(top_3_models, start=1):
        model_name = type(model).__name__
        print(f"\n--- Sedang tuning model ke-{i}: {model_name} ---")
        
        try:
            tuned_model = tune_model(
                model,
                optimize='AUC',
                n_iter=30,
                choose_better=True
            )
        except Exception as e:
            print(f"Catatan: tune_model dengan n_iter gagal/tidak didukung untuk {model_name}. Menggunakan default n_iter...")
            tuned_model = tune_model(
                model,
                optimize='AUC',
                choose_better=True
            )
            
        tuned_top_3_models.append(tuned_model)
        
        # Menampilkan Parameter Terbaik
        print(f"\nParameter terbaik untuk {model_name}:")
        if hasattr(tuned_model, "get_params"):
            for param, val in list(tuned_model.get_params().items())[:12]:
                print(f"  - {param}: {val}")

    print("\nProses training dan hyperparameter tuning 3 model machine learning terbaik selesai!")
    return tuned_top_3_models

# Function untuk Evaluasi Akhir Model Machine Learning, Validasi, dan Penyimpanan Model Machine Learning Terbaik
def evaluate_and_save_models():
    # Evaluasi Model Baseline
    baseline_recall = evaluate_baseline()

    # Menjalankan Proses Data Preprocessing, Pelatihan, & Tuning 3 Model Machine Learning Terbaik (PyCaret)
    print("Memanggil proses training & tuning dari run_training()...")
    tuned_top_3_models = run_training()

    best_recall = -1.0
    best_model_obj = None
    best_model_name = ""
    evaluated_results = []

    print("\n" + "="*50)
    print("--- Hasil Evaluasi Top-3 Model Machine Learning Pada Data Testing ---")
    print("="*50)

    for i, model in enumerate(tuned_top_3_models, start=1):
        model_name = type(model).__name__
        print(f"\n[Model ke-{i}: {model_name}]")

        # Evaluasi Pada Data Testing
        predictions = predict_model(model)
        eval_metrics = pull()
        
        # Mengambil Nilai Recall dari Masing-masing Top-3 Model Machine Learning
        try:
            if 'Recall' in eval_metrics.columns:
                recall_score = eval_metrics['Recall'].iloc[0]
            elif 'Recall' in eval_metrics.index:
                recall_score = eval_metrics.loc['Recall'].values[0]
            else:
                recall_score = eval_metrics.iloc[0]['Recall']
        except Exception:
            recall_score = eval_metrics.iloc[-1, -1]

        print(f"-> Skor recall tercatat: {recall_score:.4f}")

        evaluated_results.append({
            "model_object": model,
            "model_name": model_name,
            "recall": recall_score
        })

        if recall_score > best_recall:
            best_recall = recall_score
            best_model_obj = model
            best_model_name = model_name

    print("\n" + "="*50)
    print(f"Model Machine Learning Terbaik Berdasarkan Recall Tertinggi: {best_model_name} (Recall: {best_recall:.4f})")
    print("="*50)

    # Validasi Perbandingan Model Machine Learning Terbaik dengan Model Baseline
    print("\n" + "="*50)
    print("--- Validasi Perbandingan Model Machine Learning Terbaik dengan Model Baseline ---")
    print(f"Nilai recall model machine learning terbaik ({best_model_name}) : {best_recall:.4f}")
    print(f"Nilai recall model baseline (manual risk score): {baseline_recall:.4f}")
    
    if best_recall > baseline_recall:
        print("Model machine learning terbaik memiliki nilai recall yang lebih besar dibandingkan model baseline.")
    else:
        print("Performa model machine learning berada di bawah atau setara dengan mode baseline. Silakan tinjau kembali feature atau parameter.")
    print("="*50)

    # Memastikan Folder "model" Ada
    os.makedirs(MODELS_DIR, exist_ok=True)
    evaluated_results = sorted(evaluated_results, key=lambda x: x['recall'], reverse=True)

    # Menyimpan Model Machine Learning Terbaik ke Folder "models"
    print("\nMenyimpan model machine learning terbaik ke dalam folder model...")
    best_res = evaluated_results[0]
    model_obj = best_res['model_object']

    save_path = os.path.join(MODELS_DIR, "best_model")
    print(f"-> Menyimpan model machine learning terbaik ({best_res['model_name']} - recall: {best_res['recall']:.4f}) sebagai: best_model.pkl")

    clean_save_path = save_path.replace(".pkl", "") if save_path.endswith(".pkl") else save_path
    save_model(model_obj, model_name=clean_save_path, verbose=False)

    print("\nFile best_model.pkl berhasil disimpan di folder model!")

if __name__ == "__main__":
    evaluate_and_save_models()