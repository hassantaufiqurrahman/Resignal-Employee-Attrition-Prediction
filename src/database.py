from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Format URL Database
SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://postgres:resignal_fpdsml@db:5432/resignal_attrition_prediction"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Function untuk Mendapatkan Sesi Database saat API Dipanggil
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()