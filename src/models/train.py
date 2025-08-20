"""
Model training with MLflow tracking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import joblib

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("superkart-sales")

def train_model():
    # Load features
    print("Loading data...")
    df = pd.read_csv('data/features/training_features.csv')
    
    # Handle categorical columns
    print("Encoding categorical features...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target and ID columns from categorical list if present
    exclude_cols = ['Product_Store_Sales_Total', 'Product_Id', 'Store_Id']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Prepare X and y
    feature_cols = [col for col in df.columns 
                   if col not in ['Product_Store_Sales_Total', 'Product_Id', 'Store_Id']]
    
    # Select only numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['Product_Store_Sales_Total']
    
    # Check for any remaining non-numeric columns
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()[:10]}...")  # Show first 10
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_v1"):
        # Parameters
        params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        print("Training model...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.4f}")
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally too
        joblib.dump(model, 'models/random_forest_model.pkl')
        joblib.dump(label_encoders, 'models/label_encoders.pkl')
        print("✅ Model saved")
        
        return model, rmse, r2

if __name__ == "__main__":
    model, rmse, r2 = train_model()
    print(f"\n✅ Training complete! Check MLflow UI at http://localhost:5000")