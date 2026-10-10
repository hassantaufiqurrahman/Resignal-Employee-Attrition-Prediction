from sqlalchemy.orm import Session
from src.table import PredictionLog

# Function untuk Menyimpan Data Input Form, Hasil Prediksi, dan Probabilitas ke Dalam Database
def create_prediction_log(db: Session, input_data: dict, prediction_result: str, probability: float):
    db_item = PredictionLog(
        **input_data,
        prediction_status=prediction_result,
        probability=probability
    )
    
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# Function untuk Mengambil Daftar Riwayat Prediksi dari Database
def get_prediction_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).offset(skip).limit(limit).all()