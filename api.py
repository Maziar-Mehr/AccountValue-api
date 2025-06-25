# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from typing import List, Optional
import os
import joblib

from src.preprocessing import load_model_and_preprocessing_objects, preprocess_new_data

MODEL_PATH = 'xgboost_account_value_model.pkl'
PREPROCESSING_PATH = 'preprocessing_objects.pkl'

try:
    model, preprocessing_objects = load_model_and_preprocessing_objects(MODEL_PATH, PREPROCESSING_PATH)
except Exception as e:
    print(f"FATAL ERROR: Could not load model or preprocessing objects. Ensure they are in the same directory as api.py and are not corrupted. Error: {e}")
    raise RuntimeError(f"Failed to load essential model components: {e}")

class PredictionInput(BaseModel):
    year_established: Optional[int] = Field(None, description="Year the business was established")
    total_payroll: Optional[float] = Field(None, description="Total annual payroll of the business")
    num_employees: Optional[float] = Field(None, description="Number of employees")
    annual_revenue: Optional[float] = Field(None, description="Annual revenue of the business")
    state: Optional[str] = Field(None, description="State of the business (e.g., 'California', 'CA')")
    industry: Optional[str] = Field(None, description="Industry of the business (e.g., 'Retail', 'Technology')")
    subindustry: Optional[str] = Field(None, description="Subindustry of the business")
    business_structure: Optional[str] = Field(None, description="Legal structure of the business (e.g., 'LLC', 'Sole Proprietorship')")
    product_list: Optional[List[str]] = Field(default_factory=list, description="List of products the account has (e.g., ['General Liability', 'Workers Compensation'])")
    avg_premium: Optional[float] = Field(None, description="Average premium for the account") # Notebook expects this

app = FastAPI(
    title="Account Value Prediction API",
    description="Predicts account value based on business characteristics using an XGBoost model.",
    version="1.0.0"
)

@app.post("/predict")
async def predict_account_value(data: PredictionInput):
    """
    Receives raw business data, preprocesses it, and returns a predicted account value.
    """
    try:
        # Convert input to DataFrame, matching notebook logic
        input_data_dict = data.dict()
        input_df = pd.DataFrame([input_data_dict])

        # Preprocess using notebook-matching logic
        processed_features = preprocess_new_data(input_df, preprocessing_objects)

        # Predict (on log scale, as in notebook)
        log_prediction = model.predict(processed_features)

        # Inverse transform to original scale and clip negatives to zero (notebook logic)
        final_prediction = float(np.expm1(log_prediction)[0])
        if final_prediction < 0:
            final_prediction = 0.0

        return {"predicted_account_value": final_prediction}

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during prediction. Please check logs for more details.")

@app.get("/")
async def read_root():
    return {"message": "Account Value Prediction API is running!"}