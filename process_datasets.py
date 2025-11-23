"""
Process Dataset 2 and Dataset 3 from Excel files
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("="*70)
print("PROCESSING DATASET 2 AND DATASET 3")
print("="*70)

# ============================================================================
# Process Dataset 2
# ============================================================================

print("\n" + "="*70)
print("DATASET 2: All strikes for specific maturities")
print("="*70)

dataset2_files = os.listdir('dataset2')
print(f"\nFiles found in dataset2/: {dataset2_files}")

dataset2_data = []

for file in dataset2_files:
    if file.endswith('.xlsx'):
        filepath = os.path.join('dataset2', file)
        print(f"\nProcessing: {file}")

        try:
            # Read Excel file
            df = pd.read_excel(filepath)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")

            # Display first few rows
            print(f"\n  First 5 rows:")
            print(df.head())

            dataset2_data.append({
                'filename': file,
                'data': df
            })

        except Exception as e:
            print(f"  Error reading {file}: {e}")

# ============================================================================
# Process Dataset 3
# ============================================================================

print("\n" + "="*70)
print("DATASET 3: ~25 strikes at quarterly maturities")
print("="*70)

dataset3_files = sorted(os.listdir('dataset3'))
print(f"\nFiles found in dataset3/: {dataset3_files}")

dataset3_data = []

for file in dataset3_files:
    if file.endswith('.xlsx'):
        filepath = os.path.join('dataset3', file)
        print(f"\nProcessing: {file}")

        try:
            # Read Excel file
            df = pd.read_excel(filepath)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")

            # Display first few rows
            print(f"\n  First 3 rows:")
            print(df.head(3))

            dataset3_data.append({
                'filename': file,
                'data': df
            })

        except Exception as e:
            print(f"  Error reading {file}: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Dataset 2 files processed: {len(dataset2_data)}")
print(f"Dataset 3 files processed: {len(dataset3_data)}")

# Save processed data
import pickle
with open('processed_datasets.pkl', 'wb') as f:
    pickle.dump({
        'dataset2': dataset2_data,
        'dataset3': dataset3_data
    }, f)

print("\nProcessed data saved to: processed_datasets.pkl")
print("="*70)
