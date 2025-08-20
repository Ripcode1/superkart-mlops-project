"""
Data validation script for SuperKart dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import traceback

def validate_superkart_data():
    try:
        # Print working directory for debugging
        working_dir = os.getcwd()
        print(f"Working directory: {working_dir}")

        # Define input and output paths
        input_path = os.path.join('data', 'raw', 'SuperKart.csv')
        output_dir = os.path.join('data', 'processed')
        output_path = os.path.join(output_dir, 'superkart_clean.csv')

        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"ERROR: Input file not found at {input_path}")
            print(f"Files in data/raw: {os.listdir('data/raw') if os.path.exists('data/raw') else 'data/raw folder not found'}")
            return None
        print(f"Input file found at: {os.path.abspath(input_path)}")

        # Load data
        print("Loading data...")
        df = pd.read_csv(input_path)
        print(f"✅ Loaded {len(df)} rows")

        # Check for issues
        print("Checking for issues...")
        issues_found = []

        # 1. Check for negative sales
        negative_sales = df[df['Product_Store_Sales_Total'] < 0]
        if not negative_sales.empty:
            print(f"   Found {len(negative_sales)} rows with negative sales")
            issues_found.append("negative_sales")
            df.loc[negative_sales.index, 'Product_Store_Sales_Total'] = 0  # Replace with 0 or handle as needed

        # 2. Check establishment years
        current_year = datetime.now().year  # 2025 based on today's date
        future_years = df[df['Store_Establishment_Year'] > current_year]
        if not future_years.empty:
            print(f"   Found {len(future_years)} stores with future years (>{current_year})")
            df.loc[future_years.index, 'Store_Establishment_Year'] = current_year
            issues_found.append("future_years")

        # 3. Handle missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"   Found {missing_count} missing values, filling with forward fill...")
            df = df.ffill()
            issues_found.append("missing_values")
        else:
            print("   No missing values found")

        # Create processed directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processed directory ready at: {os.path.abspath(output_dir)}")

        # Save cleaned data
        print(f"Saving cleaned data to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned data. File size: {os.path.getsize(output_path):,} bytes")

        # Verify save
        if os.path.exists(output_path):
            print(f"✅ Verification successful. File located at: {os.path.abspath(output_path)}")
        else:
            print("❌ Verification failed: File not found after save")
            print(f"Files in {output_dir}: {os.listdir(output_dir)}")

        if issues_found:
            print(f"Summary of issues found: {', '.join(issues_found)}")
        else:
            print("✅ No issues found during validation")

        return df

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING SUPERKART DATA VALIDATION SCRIPT")
    print("=" * 60)

    result = validate_superkart_data()

    if result is not None:
        print("\n✅ Script completed successfully")
    else:
        print("\n❌ Script failed - review errors above")

    print("=" * 60)
    input