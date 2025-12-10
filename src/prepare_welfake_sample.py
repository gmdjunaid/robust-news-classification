"""
Prepare a sample from the WELFake dataset.

This script assumes each developer has the original large WELFake CSV locally.
It extracts the last 10,000 rows from the CSV and saves the result as a 
smaller CSV in data/test/ to be used as test data.

This script is kept in the repo so reviewers can see exactly how the large CSV was processed.
"""

import pandas as pd
from pathlib import Path

# File paths - assumes script is run from project root
input_file = Path('data/test/WELFake_Dataset.csv')
output_file = Path('data/test/WELFake_Dataset_sample_10000.csv')

# Verify input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_file}\n"
        "Make sure you have the original large WELFake CSV file locally."
    )

# Read the entire dataset and extract the last 10,000 rows
sample_size = 10000
print(f"Reading {input_file} (this may take a moment for large files)...")
df_full = pd.read_csv(input_file)

total_rows = len(df_full)
print(f"Total rows in dataset: {total_rows:,}")

# Extract the last 10,000 rows
print(f"\nExtracting last {min(sample_size, total_rows):,} rows...")
df_sample = df_full.tail(sample_size).copy()

# Display basic information about the sample
print(f"\nExtracted {len(df_sample)} rows")
print(f"Columns: {list(df_sample.columns)}")
print(f"\nLabel distribution:")
if 'label' in df_sample.columns:
    print(df_sample['label'].value_counts())
print(f"\nFirst few rows:")
print(df_sample.head())

# Save to a new CSV file in data/test/
print(f"\nSaving to {output_file}...")
df_sample.to_csv(output_file, index=False)

print(f"Successfully saved {len(df_sample)} rows to {output_file}")


