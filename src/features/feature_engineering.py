"""
Feature engineering for SuperKart
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
    
    def create_features(self, df):
        """Create all features"""
        print("Creating features...")
        
        # Make a copy
        df = df.copy()
        
        # 1. Temporal features
        df['Store_Age'] = 2024 - df['Store_Establishment_Year']
        df['Is_New_Store'] = (df['Store_Age'] < 5).astype(int)
        
        # 2. Product features
        df['Price_Per_Weight'] = df['Product_MRP'] / (df['Product_Weight'] + 0.01)
        df['Price_Per_Area'] = df['Product_MRP'] / (df['Product_Allocated_Area'] + 0.01)
        
        # 3. Store features
        df['Is_Tier1'] = (df['Store_Location_City_Type'] == 'Tier 1').astype(int)
        df['Is_Large_Store'] = (df['Store_Size'] == 'High').astype(int)
        
        # 4. Interaction features
        df['Premium_in_Tier1'] = df['Is_Tier1'] * (df['Product_MRP'] > df['Product_MRP'].median())
        
        # 5. Encode categoricals
        categorical_cols = ['Product_Sugar_Content', 'Product_Type', 'Store_Type']
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"Created {len(df.columns) - 12} new features")
        return df
    
    def save_encoders(self, path='models/'):
        """Save label encoders"""
        Path(path).mkdir(exist_ok=True)
        joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')
        print(f"Encoders saved to {path}")

# Run feature engineering
if __name__ == "__main__":
    # Load clean data
    df = pd.read_csv('data/processed/superkart_clean.csv')
    
    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    
    # Save
    df_features.to_csv('data/features/training_features.csv', index=False)
    fe.save_encoders()
    
    print(f"✅ Features saved: {df_features.shape}")