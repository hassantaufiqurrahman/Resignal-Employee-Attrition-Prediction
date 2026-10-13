import os

# Mendapatkan Direktori Root Project Secara Otomatis
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# Path Folder & File
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "model")

# Path File Dataset & Hasil Split
DATASET_PATH = os.path.join(DATA_DIR, "employee_attrition_dataset.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.csv")

# Konfigurasi PyCaret & Model
TARGET_COL = "Attrition"
COLS_TO_DROP = ["Employee_ID", "Attrition_Risk_Score"]
SESSION_ID = 42

# Batasan Validasi Input User
INPUT_CONSTRAINTS = {
    "Age": {
        "type": "numeric",
        "min_value": 18.0,
        "max_value": 60.0,
    },
    "Monthly_Income": {
        "type": "numeric",
        "min_value": 1000.0,
        "max_value": 25000.0,
    },
    "Years_At_Company": {
        "type": "numeric",
        "min_value": 0.0,
        "max_value": 33.0,
    },
    "Years_In_Current_Role": {
        "type": "numeric",
        "min_value": 0.0,
        "max_value": 33.0,
    },
    "Salary_Hike_Percent": {
        "type": "numeric",
        "min_value": 0.0,
        "max_value": 27,
    },
    "Distance_From_Home": {
        "type": "numeric",
        "min_value": 1,
        "max_value": 45,
    },
    "Commute_Time_Minutes": {
        "type": "numeric",
        "min_value": 5.0,
        "max_value": 150,
    },
    "Number_Of_Projects": {
        "type": "numeric",
        "min_value": 1.0,
        "max_value": 12,
    },
    "Team_Size": {
        "type": "numeric",
        "min_value": 2.0,
        "max_value": 24.0,
    },
    "Training_Hours_Last_Year": {
        "type": "numeric",
        "min_value": 0.0,
        "max_value": 70,
    },
    "Absence_Days": {
        "type": "numeric",
        "min_value": 0.0,
        "max_value": 13,
    },

    "Gender": {
        "type": "categorical",
        "choices": ["Male", "Female"],
    },
    "Marital_Status": {
        "type": "categorical",
        "choices": ["Single", "Married", "Divorced"],
    },
    "Education_Level": {
        "type": "categorical",
        "choices": ["High School", "Bachelor", "Master", "PhD"],
    },
    "Department": {
        "type": "categorical",
        "choices": ["Marketing", "IT", "Operations", "HR", "Sales", "Finance"],
    },
    "Job_Role": {
        "type": "categorical",
        "choices": [
            "Marketing Manager", "SEO Specialist", "IT Manager", "Quality Analyst",
            "Marketing Executive", "HR Executive", "Operations Manager", "Content Strategist",
            "Logistics Coordinator", "Business Development", "Software Engineer", "Financial Analyst",
            "Sales Manager", "System Administrator", "Operations Executive", "HR Manager",
            "Finance Manager", "Recruiter", "Account Executive", "Data Analyst",
            "Accountant", "Auditor", "Sales Executive", "Training Specialist"
        ],
    },
    "Employment_Type": {
        "type": "categorical",
        "choices": ["Part-Time", "Contract", "Full-Time"],
    },
    "Business_Travel": {
        "type": "categorical",
        "choices": ["Never", "Rarely", "Frequently"],
    },
    "Overtime": {
        "type": "categorical",
        "choices": ["Yes", "No"],
    },
    "Remote_Work": {
        "type": "categorical",
        "choices": ["Yes", "No"],
    },
    "Promotion_Last_5_Years": {
        "type": "categorical",
        "choices": ["Yes", "No"],
    },
    "Stock_Option": {
        "type": "categorical",
        "choices": ["Yes", "No"],
    },

    "Job_Level": {
        "type": "categorical",
        "choices": [1, 2, 3, 4, 5],
    },
    "Performance_Rating": {
        "type": "categorical",
        "choices": [1, 2, 3, 4, 5],
    },
    "Job_Satisfaction": {
        "type": "categorical",
        "choices": [1, 2, 3, 4, 5],
    },
    "Work_Life_Balance": {
        "type": "categorical",
        "choices": [1, 2, 3, 4, 5],
    },
    "Environment_Satisfaction": {
        "type": "categorical",
        "choices": [1, 2, 3, 4, 5],
    },
    "Relationship_Satisfaction": {
        "type": "categorical",
        "choices": [1, 2, 3, 4, 5],
    },
    "Manager_Rating": {
        "type": "categorical",
        "choices": [1, 2, 3, 4, 5],
    },
}