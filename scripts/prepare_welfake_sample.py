"""
Prepare a sample from the WELFake dataset.

This script assumes each developer has the original large WELFake CSV locally.
It cleans/reduces that CSV by taking (for now) the first 1,000 rows and
saves the result as a smaller CSV in test-data/ to be used as test data.

This script is kept in the repo so reviewers can see exactly how the large CSV was processed.
"""

import pandas as pd
from pathlib import Path

# File paths - assumes script is run from project root
input_file = Path('test-data/WELFake_Dataset.csv')
output_file = Path('test-data/WELFake_Dataset_sample_1000.csv')

# Verify input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_file}\n"
        "Make sure you have the original large WELFake CSV file locally."
    )

# Read the first 1000 rows from the CSV file
print(f"Reading first 1,000 entries from {input_file}...")
df_sample = pd.read_csv(input_file, nrows=1000)

# Display basic information about the sample
print(f"\nExtracted {len(df_sample)} rows")
print(f"Columns: {list(df_sample.columns)}")
print(f"\nFirst few rows:")
print(df_sample.head())

# Save to a new CSV file in test-data/
print(f"\nSaving to {output_file}...")
df_sample.to_csv(output_file, index=False)

print(f"Successfully saved {len(df_sample)} rows to {output_file}")

