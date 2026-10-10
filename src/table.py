from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from src.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # Inisialisasi Kolom Input 
    Age = Column(Integer)
    Gender = Column(String)
    Marital_Status = Column(String)
    Education_Level = Column(String)
    Department = Column(String)
    Job_Role = Column(String)
    Employment_Type = Column(String)
    Years_At_Company = Column(Integer)
    Years_In_Current_Role = Column(Integer)
    Monthly_Income = Column(Integer)
    Salary_Hike_Percent = Column(Float)
    Job_Level = Column(Integer)
    Performance_Rating = Column(Integer)
    Job_Satisfaction = Column(Integer)
    Work_Life_Balance = Column(Integer)
    Environment_Satisfaction = Column(Integer)
    Relationship_Satisfaction = Column(Integer)
    Training_Hours_Last_Year = Column(Integer)
    Overtime = Column(String)
    Remote_Work = Column(String)
    Business_Travel = Column(String)
    Distance_From_Home = Column(Integer)
    Commute_Time_Minutes = Column(Integer)
    Number_Of_Projects = Column(Integer)
    Team_Size = Column(Integer)
    Manager_Rating = Column(Integer)
    Absence_Days = Column(Integer)
    Promotion_Last_5_Years = Column(String)
    Stock_Option = Column(String)
    
    # Kolom Hasil Prediksi dari Model
    prediction_status = Column(String)  
    probability = Column(Float)         
    
    # Waktu saat Prediksi Dilakukan
    created_at = Column(DateTime, default=datetime.utcnow)