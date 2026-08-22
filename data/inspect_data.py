import pandas as pd

# Dataset path
DATA_PATH = "raw/emi_prediction_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

# 1. Dataset dimensions
print(f"\nRows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# 2. Column names
print("\nCOLUMN NAMES")
print("-" * 60)
for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

# 3. Data types
print("\nDATA TYPES")
print("-" * 60)
print(df.dtypes)

# 4. Missing values
print("\nMISSING VALUES")
print("-" * 60)
missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No missing values found.")
else:
    print(missing)

# 5. Duplicate rows
print("\nDUPLICATE ROWS")
print("-" * 60)
print(f"Duplicate rows: {df.duplicated().sum():,}")

# 6. Basic numerical statistics
print("\nNUMERICAL SUMMARY")
print("-" * 60)
print(df.describe().T)

# 7. Unique values for categorical columns
print("\nCATEGORICAL/COLUMN VALUE SUMMARY")
print("-" * 60)

for column in df.select_dtypes(include="object").columns:
    print(f"\n{column}")
    print(df[column].value_counts(dropna=False).head(10))

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)