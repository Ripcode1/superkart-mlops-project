"""
Prefect pipeline for automated training
"""

from prefect import flow, task
import pandas as pd
from pathlib import Path
import sys
sys.path.append('.')

from src.features.feature_engineering import FeatureEngineer
from src.models.train import train_model

@task
def load_raw_data():
    print("Loading raw data...")
    df = pd.read_csv('data/raw/SuperKart.csv')
    print(f"Loaded {len(df)} rows")
    return df

@task
def clean_data(df):
    print("Cleaning data...")
    df = df[df['Product_Store_Sales_Total'] >= 0]
    df.loc[df['Store_Establishment_Year'] > 2024, 'Store_Establishment_Year'] = 2024
    return df

@task
def create_features(df):
    print("Engineering features...")
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    df_features.to_csv('data/features/training_features.csv', index=False)
    return df_features

@task
def train_model_task():
    print("Training model...")
    model, rmse, r2 = train_model()
    return {"rmse": rmse, "r2": r2}

@flow(name="SuperKart Training Pipeline")
def training_pipeline():
    raw_data = load_raw_data()
    clean_data_df = clean_data(raw_data)
    features_df = create_features(clean_data_df)
    metrics = train_model_task()
    
    print(f"\nPipeline complete! RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}")
    return metrics

if __name__ == "__main__":
    training_pipeline()