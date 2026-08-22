from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_ready"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "results"
)

MODEL_PATH = (
    MODEL_DIR
    / "emi_amount_regressor.joblib"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "regression_results.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("EMIPredict AI - EMI AMOUNT REGRESSOR")
print("=" * 70)

print("\nLoading training and testing data...")

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "y_reg_train.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    DATA_DIR / "y_reg_test.csv"
).iloc[:, 0]


print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

print("\nRegression target summary:")
print(y_train.describe().round(2))


# ============================================================
# 1. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()


print("\n" + "-" * 70)
print("FEATURE INFORMATION")
print("-" * 70)

print(f"\nNumerical features   : {len(numeric_features)}")
print(f"Categorical features : {len(categorical_features)}")


# ============================================================
# 2. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ============================================================
# 3. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        ),
    ]
)


# ============================================================
# 4. COMBINED PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


# ============================================================
# 5. XGBOOST REGRESSOR
# ============================================================

regressor = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=300,
    max_depth=6,
    learning_rate=0.07,

    subsample=0.85,
    colsample_bytree=0.85,

    min_child_weight=2,

    reg_alpha=0.1,
    reg_lambda=1.0,

    eval_metric="rmse",

    tree_method="hist",

    random_state=42,
    n_jobs=-1,
)


# ============================================================
# 6. COMPLETE REGRESSION PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            regressor
        ),
    ]
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\n" + "-" * 70)
print("MODEL TRAINING")
print("-" * 70)

print("\nTraining XGBoost regression model...")
print(
    "This may take some time because the dataset contains "
    "over 320,000 training records."
)

model_pipeline.fit(
    X_train,
    y_train
)

print("\nModel training completed.")


# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating EMI predictions...")

y_pred = model_pipeline.predict(
    X_test
)


# ============================================================
# 9. SAFETY CHECK
# ============================================================

# EMI cannot be negative.
y_pred = np.maximum(
    y_pred,
    0
)


# ============================================================
# 10. REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION RESULTS")
print("=" * 70)

print(f"\nMAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 12. SAMPLE PREDICTIONS
# ============================================================

prediction_comparison = pd.DataFrame(
    {
        "Actual_EMI": y_test.values,
        "Predicted_EMI": y_pred,
        "Absolute_Error": np.abs(
            y_test.values - y_pred
        ),
    }
)

print("\n" + "-" * 70)
print("SAMPLE PREDICTIONS")
print("-" * 70)

print(
    prediction_comparison
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 13. CREATE OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

joblib.dump(
    model_pipeline,
    MODEL_PATH
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EMIPredict AI - EMI Amount Regression Results\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Training samples : {len(X_train):,}\n"
    )

    file.write(
        f"Testing samples  : {len(X_test):,}\n\n"
    )

    file.write(
        f"Number of features : {X_train.shape[1]}\n"
    )

    file.write(
        f"Numerical features : {len(numeric_features)}\n"
    )

    file.write(
        f"Categorical features : {len(categorical_features)}\n\n"
    )

    file.write(
        f"MAE  : ₹{mae:,.2f}\n"
    )

    file.write(
        f"RMSE : ₹{rmse:,.2f}\n"
    )

    file.write(
        f"R²   : {r2:.4f}\n"
    )


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "-" * 70)
print("MODEL SAVED")
print("-" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nResults:")
print(RESULTS_PATH)

print("\n" + "=" * 70)
print("REGRESSION TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)