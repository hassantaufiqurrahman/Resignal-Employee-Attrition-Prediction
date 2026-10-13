import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATASET_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, COLS_TO_DROP, SESSION_ID, TARGET_COL

def get_train_test_data():
    # Load Dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Berhasil memuat dataset dari {DATASET_PATH}. Total baris awal: {len(df)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File dataset tidak ditemukan di path: {DATASET_PATH}")

    # Cek & Hapus Duplikat
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Ditemukan dan menghapus {duplicate_count} baris duplikat.")

    # Split Dataset
    train_data, test_data = train_test_split(
        df, 
        test_size=0.2, 
        random_state=SESSION_ID, 
        stratify=df[TARGET_COL]
    )
    
    print(f"Jumlah baris data training : {train_data.shape}")
    print(f"Jumlah baris data testing  : {test_data.shape}")

    # Hapus Feature yang Tidak Dipakai Pada Data Training & Testing
    existing_cols_to_drop = [col for col in COLS_TO_DROP if col in train_data.columns]
    if existing_cols_to_drop:
        train_data = train_data.drop(columns=existing_cols_to_drop)
        test_data = test_data.drop(columns=existing_cols_to_drop)
        print(f"Feature berikut telah dihapus dari data train dan test: {existing_cols_to_drop}")

    # Memastikan Folder "data" Ada 
    os.makedirs(os.path.dirname(TRAIN_DATA_PATH), exist_ok=True)

    # Simpan dan Ekspor Hasil Split Data ke Folder "data"
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)
    print(f"File train_data.csv dan test_data.csv berhasil disimpan di folder data/")

    return train_data, test_data

if __name__ == "__main__":
    get_train_test_data()