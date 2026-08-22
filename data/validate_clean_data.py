import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emi_prediction_cleaned.csv"
)


# ============================================================
# LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(DATA_PATH)


print("=" * 70)
print("CLEANED DATA VALIDATION")
print("=" * 70)

print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 1. DATA TYPES
# ============================================================

print("\n" + "-" * 70)
print("1. DATA TYPES")
print("-" * 70)

print(df.dtypes)


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n" + "-" * 70)
print("2. MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    for column, count in missing.items():
        percentage = (count / len(df)) * 100
        print(f"{column:25} {count:8,} ({percentage:.2f}%)")


# ============================================================
# 3. DUPLICATES
# ============================================================

print("\n" + "-" * 70)
print("3. DUPLICATE ROWS")
print("-" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")


# ============================================================
# 4. NUMERIC CONVERSION CHECK
# ============================================================

print("\n" + "-" * 70)
print("4. NUMERIC COLUMN VALIDATION")
print("-" * 70)

numeric_columns = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
    "max_monthly_emi",
]

for column in numeric_columns:
    invalid = pd.to_numeric(
        df[column],
        errors="coerce"
    ).isna().sum()

    # Ignore legitimate missing values
    actual_missing = df[column].isna().sum()

    conversion_errors = invalid - actual_missing

    print(
        f"{column:25} "
        f"conversion errors: {max(conversion_errors, 0):,}"
    )


# ============================================================
# 5. CATEGORICAL CONSISTENCY
# ============================================================

print("\n" + "-" * 70)
print("5. CATEGORICAL VALUE CHECK")
print("-" * 70)

categorical_columns = [
    "gender",
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

    print(f"\n{column}")
    print(df[column].value_counts(dropna=False))


# ============================================================
# 6. CREDIT SCORE CHECK
# ============================================================

print("\n" + "-" * 70)
print("6. CREDIT SCORE VALIDATION")
print("-" * 70)

if "credit_score" in df.columns:

    invalid_credit = df[
        (df["credit_score"] < 300) |
        (df["credit_score"] > 850)
    ]

    print(
        f"Values outside 300-850 range: "
        f"{len(invalid_credit):,}"
    )

    if len(invalid_credit) > 0:
        print("\nSample invalid values:")
        print(
            invalid_credit["credit_score"]
            .value_counts()
            .head(10)
        )


# ============================================================
# 7. TARGET VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("7. TARGET VALIDATION")
print("-" * 70)

print("\nEMI Eligibility:")
print(df["emi_eligibility"].value_counts())

print("\nMax Monthly EMI:")
print(df["max_monthly_emi"].describe())


# ============================================================
# 8. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)