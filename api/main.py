import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from src.config import MODELS_DIR
from src.predict import make_prediction
from src.database import engine, Base, get_db  
from src.crud import create_prediction_log
from src.table import PredictionLog                  

# Inisialisasi FastAPI
app = FastAPI(
    title="Resignal: Employee Attrition Prediction",
    description="Specialized API for Predicting Employee Resignation. Made by Hassan Taufiqurrahman for Final Project Dibimbing Data Science & AI Machine Learning Batch 42",
    version="1.0"
)

# Inisialisasi Pembuatan Tabel Database
Base.metadata.create_all(bind=engine)

# Inisialisasi Global untuk Pengecekan Model di Health Check
model_file_path = os.path.join(MODELS_DIR, "best_model.pkl")
is_model_available = os.path.exists(model_file_path)

# Schema Input Data Karyawan
class EmployeeInput(BaseModel):
    Age: int = Field(..., json_schema_extra={"example": 26})
    Gender: str = Field(..., json_schema_extra={"example": "Female"})
    Marital_Status: str = Field(..., json_schema_extra={"example": "Divorced"})
    Education_Level: str = Field(..., json_schema_extra={"example": "Bachelor"})
    Department: str = Field(..., json_schema_extra={"example": "Marketing"})
    Job_Role: str = Field(..., json_schema_extra={"example": "Marketing Manager"})
    Employment_Type: str = Field(..., json_schema_extra={"example": "Part-Time"})
    Years_At_Company: int = Field(..., json_schema_extra={"example": 1})
    Years_In_Current_Role: int = Field(..., json_schema_extra={"example": 0})
    Monthly_Income: int = Field(..., json_schema_extra={"example": 7673})
    Salary_Hike_Percent: float = Field(..., json_schema_extra={"example": 11.0})
    Job_Level: int = Field(..., json_schema_extra={"example": 2})
    Performance_Rating: int = Field(..., json_schema_extra={"example": 2})
    Job_Satisfaction: int = Field(..., json_schema_extra={"example": 1})
    Work_Life_Balance: int = Field(..., json_schema_extra={"example": 2})
    Environment_Satisfaction: int = Field(..., json_schema_extra={"example": 4})
    Relationship_Satisfaction: int = Field(..., json_schema_extra={"example": 3})
    Training_Hours_Last_Year: int = Field(..., json_schema_extra={"example": 58})
    Overtime: str = Field(..., json_schema_extra={"example": "Yes"})
    Remote_Work: str = Field(..., json_schema_extra={"example": "No"})
    Business_Travel: str = Field(..., json_schema_extra={"example": "Never"})
    Distance_From_Home: int = Field(..., json_schema_extra={"example": 6})
    Commute_Time_Minutes: int = Field(..., json_schema_extra={"example": 23})
    Number_Of_Projects: int = Field(..., json_schema_extra={"example": 7})
    Team_Size: int = Field(..., json_schema_extra={"example": 7})
    Manager_Rating: int = Field(..., json_schema_extra={"example": 3})
    Absence_Days: int = Field(..., json_schema_extra={"example": 7})
    Promotion_Last_5_Years: str = Field(..., json_schema_extra={"example": "No"})
    Stock_Option: str = Field(..., json_schema_extra={"example": "No"})

# Endpoint Health Check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": is_model_available
    }

# Endpoint Predict dengan Integrasi Database
@app.post("/predict")
def predict_attrition(data: EmployeeInput, db: Session = Depends(get_db)):
    try:
        input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    
        result = make_prediction(input_dict, model_name="best_model")
        
        pred_label = 1 if result["prediction_status"] == "Resign" else 0
        pred_score = result["probability"]
        str_prediction_result = result["prediction_status"]

        # Menyimpann Data Input dan Hasil Prediksi ke Database
        create_prediction_log(
            db=db,
            input_data=input_dict,
            prediction_result=str_prediction_result,
            probability=pred_score
        )

        return {
            "status": "success",
            "prediction_label": pred_label,
            "prediction_result": str_prediction_result,
            "resign_probability": pred_score,
            "message": "The prediction was successfully made and logged to the database!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while processing the prediction: {str(e)}")