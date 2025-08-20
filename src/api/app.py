# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load('models/random_forest_model.pkl')

app = FastAPI(title="SuperKart Sales Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    product_weight: float
    product_sugar_content: str
    product_allocated_area: float
    product_type: str
    product_mrp: float
    store_establishment_year: int
    store_size: str
    store_location_city_type: str
    store_type: str

@app.get("/")
def root():
    return {"message": "SuperKart Sales Prediction API"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Create the dataframe with ALL columns (including text ones)
        data = pd.DataFrame([{
            'Product_Weight': request.product_weight,
            'Product_Sugar_Content': request.product_sugar_content,  # Keep as string
            'Product_Allocated_Area': request.product_allocated_area,
            'Product_Type': request.product_type,  # Keep as string
            'Product_MRP': request.product_mrp,
            'Store_Establishment_Year': request.store_establishment_year,
            'Store_Size': request.store_size,  # Keep as string
            'Store_Location_City_Type': request.store_location_city_type,  # Keep as string
            'Store_Type': request.store_type,  # Keep as string
            'Store_Age': 2024 - request.store_establishment_year,
            'Is_New_Store': 1 if (2024 - request.store_establishment_year) < 5 else 0,
            'Price_Per_Weight': request.product_mrp / (request.product_weight + 0.01),
            'Price_Per_Area': request.product_mrp / (request.product_allocated_area + 0.01),
            'Is_Tier1': 1 if request.store_location_city_type == 'Tier 1' else 0,
            'Is_Large_Store': 1 if request.store_size == 'High' else 0,
            'Premium_in_Tier1': 1 if (request.store_location_city_type == 'Tier 1' and request.product_mrp > 100) else 0,
            'Product_Sugar_Content_Encoded': 0 if request.product_sugar_content in ['Low', 'low'] else 1,
            'Product_Type_Encoded': abs(hash(request.product_type)) % 20,
            'Store_Type_Encoded': 0 if 'Type1' in request.store_type else (1 if 'Type2' in request.store_type else 2)
        }])
        
        # Select ONLY numeric columns for prediction (like your training did)
        numeric_columns = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP',
                          'Store_Establishment_Year', 'Store_Age', 'Is_New_Store',
                          'Price_Per_Weight', 'Price_Per_Area', 'Is_Tier1', 'Is_Large_Store',
                          'Premium_in_Tier1', 'Product_Sugar_Content_Encoded',
                          'Product_Type_Encoded', 'Store_Type_Encoded']
        
        X = data[numeric_columns]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return {
            "predicted_sales": float(prediction),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)