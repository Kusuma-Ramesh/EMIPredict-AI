import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "emi_prediction_dataset.csv"
)

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n" + "=" * 70)
print("DATA QUALITY INVESTIGATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Mixed-type columns
# ---------------------------------------------------------
print("\n1. MIXED / OBJECT COLUMNS")
print("-" * 70)

object_columns = df.select_dtypes(include="object").columns

for column in object_columns:
    print(f"\n{column}")
    print(f"Data type: {df[column].dtype}")
    print(f"Unique values: {df[column].nunique(dropna=True):,}")
    print("Sample values:")
    print(df[column].dropna().astype(str).unique()[:15])

# ---------------------------------------------------------
# 2. Missing-value analysis
# ---------------------------------------------------------
print("\n\n2. MISSING VALUE ANALYSIS")
print("-" * 70)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values.")
else:
    for column, count in missing.items():
        percentage = (count / len(df)) * 100
        print(f"{column:25} {count:8,} ({percentage:.2f}%)")

# ---------------------------------------------------------
# 3. Numerical range investigation
# ---------------------------------------------------------
print("\n\n3. NUMERICAL RANGE INVESTIGATION")
print("-" * 70)

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    print(
        f"{column:25} "
        f"min={df[column].min():,.2f} | "
        f"max={df[column].max():,.2f}"
    )

# ---------------------------------------------------------
# 4. Target distribution
# ---------------------------------------------------------
print("\n\n4. CLASSIFICATION TARGET DISTRIBUTION")
print("-" * 70)

target_counts = df["emi_eligibility"].value_counts(dropna=False)

for value, count in target_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{str(value):20} {count:8,} ({percentage:.2f}%)")

# ---------------------------------------------------------
# 5. Regression target
# ---------------------------------------------------------
print("\n\n5. REGRESSION TARGET SUMMARY")
print("-" * 70)

print(df["max_monthly_emi"].describe())

# ---------------------------------------------------------
# 6. Potentially suspicious values
# ---------------------------------------------------------
print("\n\n6. POTENTIALLY SUSPICIOUS VALUES")
print("-" * 70)

if "credit_score" in df.columns:
    print("\nCredit score values above 850:")
    print(df.loc[df["credit_score"] > 850, "credit_score"]
          .value_counts()
          .head(20))

if "age" in df.columns:
    print("\nAge values:")
    print(df["age"].dropna().astype(str).value_counts().head(20))

print("\n" + "=" * 70)
print("DATA QUALITY INVESTIGATION COMPLETED")
print("=" * 70)

print("\n\n7. VALUES THAT BECOME INVALID DURING NUMERIC CONVERSION")
print("-" * 70)

for column in ["age", "monthly_salary", "bank_balance"]:
    raw_values = pd.read_csv(
        DATA_PATH,
        usecols=[column],
        low_memory=False
    )

    converted = pd.to_numeric(
        raw_values[column],
        errors="coerce"
    )

    invalid_mask = (
        raw_values[column].notna() &
        converted.isna()
    )

    print(f"\n{column}")
    print(f"Values that cannot be converted: {invalid_mask.sum():,}")

    if invalid_mask.any():
        print("Examples:")
        print(
            raw_values.loc[invalid_mask, column]
            .astype(str)
            .value_counts()
            .head(20)
        )