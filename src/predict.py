import os
import pandas as pd
from pycaret.classification import load_model, predict_model
from src.config import MODELS_DIR

# Function untuk Feature Engineering
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


class AttritionPredictor:
    # Inisialisasi Predictor dengan "best_model.pkl"
    def __init__(self, model_name="best_model"):
        self.model_path = os.path.join(MODELS_DIR, model_name)
        print(f"Memuat model dari: {self.model_path}.pkl ...")
        try:
            self.model = load_model(self.model_path)
            print("Model berhasil dimuat!")
        except Exception as e:
            raise FileNotFoundError(
                f"Gagal memuat model di {self.model_path}.pkl. Pastikan train.py sudah dijalankan terlebih dahulu! Error: {e}"
            )

    # Function untuk Prediksi
    def predict(self, input_data: dict) -> dict:
        df_input = pd.DataFrame([input_data])

        df_processed = create_features(df_input)

        predictions = predict_model(self.model, data=df_processed)

        raw_label = predictions['prediction_label'].iloc[0]  
        raw_score = predictions['prediction_score'].iloc[0] 

        if str(raw_label).lower() in ['yes', '1', 'resign']:
            prediction_status = "Resign"
            resign_probability = float(raw_score) 
        else:
            prediction_status = "Stayed"
            resign_probability = float(1.0 - raw_score)

        return {
            "prediction_status": prediction_status,
            "probability": round(resign_probability, 4),
            "raw_predictions": predictions.to_dict(orient="records")[0]
        }

# Function Instan yang Bisa Langsung Dipanggil oleh FastAPI / Streamlit
def make_prediction(input_data: dict, model_name="best_model"):
    predictor = AttritionPredictor(model_name=model_name)
    return predictor.predict(input_data)