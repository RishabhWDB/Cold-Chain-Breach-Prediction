import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

class ShipmentFeatures(BaseModel):
    inside_temp_mean: float
    inside_temp_max: float
    inside_temp_std: float
    outside_temp_mean: float
    outside_temp_max: float
    outside_temp_std: float
    humidity_mean: float
    humidity_max: float
    humidity_std: float
    door_open_count: int
    minutes_above_threshold: int
    max_consecutive_violations: int
    temp_humidity_interaction: float
    product_type_dairy: bool
    product_type_fish: bool
    product_type_pharma: bool
    product_type_produce: bool
    cooling_system_premium: bool
    cooling_system_standard: bool

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(features: ShipmentFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    return {
        "label" : "breached" if prediction[0] == 1 else "safe",
        "confidence" :float(prediction_proba[0].max())
    }

@app.get("/health")
def health():
    return {"status":"healthy"}
