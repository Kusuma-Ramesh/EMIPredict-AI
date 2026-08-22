from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "emi_prediction_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "emi_prediction_cleaned.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(RAW_PATH, low_memory=False)

original_rows = len(df)
original_columns = len(df.columns)

print("\n" + "=" * 70)
print("EMI DATA CLEANING PIPELINE")
print("=" * 70)

print(f"\nOriginal shape: {df.shape}")


# ============================================================
# 1. NORMALIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ============================================================
# 2. CLEAN CORRUPTED NUMERIC REPRESENTATIONS
# ============================================================

numeric_columns = [
    "age",
    "monthly_salary",
    "bank_balance",
]

for column in numeric_columns:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .replace({
            "nan.0": pd.NA,
            "NaN.0": pd.NA,
            "None.0": pd.NA,
        })
    )

    # Repair values such as:
    # 58.0.0     -> 58.0
    # 18000.0.0  -> 18000.0
    # 183300.0.0 -> 183300.0
    df[column] = df[column].str.replace(
        r"\.0\.0$",
        ".0",
        regex=True
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )
# ============================================================
# 3. NORMALIZE GENDER VALUES
# ============================================================

df["gender"] = (
    df["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female",
    })
)


# ============================================================
# 4. NORMALIZE OTHER CATEGORICAL VALUES
# ============================================================

categorical_columns = [
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
    "emi_eligibility",
]

for column in categorical_columns:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )


# ============================================================
# 5. HANDLE INVALID CREDIT SCORES
# ============================================================
# Standard credit-score range is 300-850.
# Values outside this range are treated as invalid/missing
# instead of being silently deleted.

invalid_credit_score = (
    (df["credit_score"] < 300) |
    (df["credit_score"] > 850)
)

invalid_credit_count = invalid_credit_score.sum()

df.loc[invalid_credit_score, "credit_score"] = pd.NA


# ============================================================
# 6. HANDLE MISSING NUMERICAL VALUES
# ============================================================

numerical_imputation_columns = [
    "age",
    "monthly_salary",
    "monthly_rent",
    "credit_score",
    "bank_balance",
    "emergency_fund",
]

for column in numerical_imputation_columns:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)


# ============================================================
# 7. HANDLE MISSING CATEGORICAL VALUES
# ============================================================

categorical_imputation_columns = [
    "education",
]

for column in categorical_imputation_columns:
    mode_value = df[column].mode(dropna=True)

    if not mode_value.empty:
        df[column] = df[column].fillna(mode_value.iloc[0])


# ============================================================
# 8. VERIFY REMAINING MISSING VALUES
# ============================================================

remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[
    remaining_missing > 0
]


# ============================================================
# 9. REMOVE DUPLICATES
# ============================================================

duplicates_before = df.duplicated().sum()

if duplicates_before > 0:
    df = df.drop_duplicates()


# ============================================================
# 10. SAVE CLEAN DATASET
# ============================================================

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(
    PROCESSED_PATH,
    index=False
)


# ============================================================
# CLEANING REPORT
# ============================================================

print("\n" + "-" * 70)
print("CLEANING SUMMARY")
print("-" * 70)

print(f"Original rows              : {original_rows:,}")
print(f"Original columns           : {original_columns}")
print(f"Final rows                 : {len(df):,}")
print(f"Final columns              : {len(df.columns)}")

print(f"\nDuplicate rows removed     : {duplicates_before:,}")
print(f"Invalid credit scores fixed: {invalid_credit_count:,}")

print("\nFinal data types:")
print(df.dtypes)

print("\nRemaining missing values:")

if remaining_missing.empty:
    print("No missing values remain.")
else:
    print(remaining_missing)

print(f"\nClean dataset saved to:")
print(PROCESSED_PATH)

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED")
print("=" * 70)