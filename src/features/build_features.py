from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emi_prediction_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_ready"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("\n" + "=" * 70)
print("EMIPredict AI - FEATURE ENGINEERING")
print("=" * 70)

print(f"\nInput dataset shape: {df.shape}")


# ============================================================
# 1. FINANCIAL FEATURE ENGINEERING
# ============================================================

# Total regular household expenses
df["total_monthly_expenses"] = (
    df["monthly_rent"]
    + df["school_fees"]
    + df["college_fees"]
    + df["travel_expenses"]
    + df["groceries_utilities"]
    + df["other_monthly_expenses"]
)

# Total current debt obligation
df["total_existing_obligations"] = (
    df["current_emi_amount"]
)

# Approximate disposable income before new EMI
df["disposable_income"] = (
    df["monthly_salary"]
    - df["total_monthly_expenses"]
    - df["current_emi_amount"]
)

# Current EMI burden relative to salary
df["current_emi_ratio"] = (
    df["current_emi_amount"]
    / df["monthly_salary"].replace(0, pd.NA)
)

# Total expense burden relative to salary
df["expense_ratio"] = (
    df["total_monthly_expenses"]
    / df["monthly_salary"].replace(0, pd.NA)
)

# Requested amount relative to annual income
df["loan_to_annual_income"] = (
    df["requested_amount"]
    / (df["monthly_salary"] * 12).replace(0, pd.NA)
)

# Emergency fund coverage of monthly expenses
df["emergency_fund_coverage"] = (
    df["emergency_fund"]
    / df["total_monthly_expenses"].replace(0, pd.NA)
)

# Bank balance relative to requested loan amount
df["bank_balance_to_loan_ratio"] = (
    df["bank_balance"]
    / df["requested_amount"].replace(0, pd.NA)
)


# ============================================================
# 2. CLEAN ENGINEERED VALUES
# ============================================================

engineered_columns = [
    "current_emi_ratio",
    "expense_ratio",
    "loan_to_annual_income",
    "emergency_fund_coverage",
    "bank_balance_to_loan_ratio",
]

for column in engineered_columns:
    df[column] = (
        df[column]
        .replace([float("inf"), float("-inf")], pd.NA)
    )


# ============================================================
# 3. IDENTIFY TARGETS
# ============================================================

CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"

print("\nTargets:")
print(f"Classification: {CLASSIFICATION_TARGET}")
print(f"Regression    : {REGRESSION_TARGET}")


# ============================================================
# 4. REMOVE TARGET COLUMNS FROM FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in [
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET,
    ]
]

X = df[feature_columns].copy()

y_classification = df[CLASSIFICATION_TARGET].copy()
y_regression = df[REGRESSION_TARGET].copy()


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

(
    X_train,
    X_test,
    y_class_train,
    y_class_test,
    y_reg_train,
    y_reg_test,
) = train_test_split(
    X,
    y_classification,
    y_regression,
    test_size=0.20,
    random_state=42,
    stratify=y_classification,
)


# ============================================================
# 6. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 7. SAVE TRAINING / TEST DATA
# ============================================================

X_train.to_csv(
    OUTPUT_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    OUTPUT_DIR / "X_test.csv",
    index=False
)

y_class_train.to_csv(
    OUTPUT_DIR / "y_class_train.csv",
    index=False
)

y_class_test.to_csv(
    OUTPUT_DIR / "y_class_test.csv",
    index=False
)

y_reg_train.to_csv(
    OUTPUT_DIR / "y_reg_train.csv",
    index=False
)

y_reg_test.to_csv(
    OUTPUT_DIR / "y_reg_test.csv",
    index=False
)


# ============================================================
# 8. FEATURE INFORMATION
# ============================================================

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()


print("\n" + "-" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("-" * 70)

print(f"\nOriginal rows          : {len(df):,}")
print(f"Original columns       : 27")
print(f"Final feature columns  : {len(feature_columns)}")

print(f"\nNumerical features     : {len(numeric_features)}")
print(f"Categorical features   : {len(categorical_features)}")

print("\nEngineered features:")
for column in [
    "total_monthly_expenses",
    "total_existing_obligations",
    "disposable_income",
    "current_emi_ratio",
    "expense_ratio",
    "loan_to_annual_income",
    "emergency_fund_coverage",
    "bank_balance_to_loan_ratio",
]:
    print(f"  - {column}")


print("\nTrain/Test split:")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

print("\nClassification target:")
print(y_class_train.value_counts())


print("\nRegression target:")
print(y_reg_train.describe().round(2))


# ============================================================
# 9. COMPLETION
# ============================================================

print("\n" + "-" * 70)
print("ML-READY DATA SAVED")
print("-" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("1. X_train.csv")
print("2. X_test.csv")
print("3. y_class_train.csv")
print("4. y_class_test.csv")
print("5. y_reg_train.csv")
print("6. y_reg_test.csv")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 70)