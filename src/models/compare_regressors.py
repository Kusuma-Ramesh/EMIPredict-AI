from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "results"
)

COMPARISON_PATH = (
    RESULTS_DIR
    / "regression_model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("EMIPredict AI - REGRESSION MODEL COMPARISON")
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
print(
    y_train.describe().round(2)
)


# ============================================================
# FEATURE TYPES
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

print(
    f"\nNumerical features   : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)


# ============================================================
# PREPROCESSING
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
# MAPE FUNCTION
# ============================================================

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.

    Rows where the actual value is zero are excluded
    to avoid division-by-zero errors.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    non_zero_mask = y_true != 0

    if not np.any(non_zero_mask):
        return np.nan

    return np.mean(
        np.abs(
            (
                y_true[non_zero_mask]
                - y_pred[non_zero_mask]
            )
            / y_true[non_zero_mask]
        )
    ) * 100


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    ),

    "XGBoost": XGBRegressor(
        objective="reg:squarederror",

        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,

        subsample=0.85,
        colsample_bytree=0.85,

        min_child_weight=2,
        gamma=0,

        reg_alpha=0.1,
        reg_lambda=1.0,

        eval_metric="rmse",

        tree_method="hist",

        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []


for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"TRAINING: {model_name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                model
            ),
        ]
    )

    print(
        f"\nTraining {model_name}..."
    )

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    print(
        "Generating EMI predictions..."
    )

    y_pred = pipeline.predict(
        X_test
    )

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

    mape = calculate_mape(
        y_test,
        y_pred
    )

    print(
        f"\nMAE  : ₹{mae:,.2f}"
    )

    print(
        f"RMSE : ₹{rmse:,.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        f"MAPE : {mape:.2f}%"
    )

    results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
        }
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


# Best model:
# lower MAE/RMSE/MAPE are better,
# higher R² is better.
#
# For selection, RMSE is the primary metric
# because it is explicitly used by the project
# acceptance criteria.

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.4f}".format,
            "MAPE": "{:.2f}%".format,
        }
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "-" * 70)
print("BEST REGRESSION MODEL")
print("-" * 70)

print(
    f"\nModel : "
    f"{best_model['Model']}"
)

print(
    f"MAE   : "
    f"₹{best_model['MAE']:,.2f}"
)

print(
    f"RMSE  : "
    f"₹{best_model['RMSE']:,.2f}"
)

print(
    f"R²    : "
    f"{best_model['R2']:.4f}"
)

print(
    f"MAPE  : "
    f"{best_model['MAPE']:.2f}%"
)


# ============================================================
# ACCEPTANCE CRITERION
# ============================================================

print("\n" + "-" * 70)
print("REGRESSION ACCEPTANCE CHECK")
print("-" * 70)

if best_model["RMSE"] < 2000:

    print(
        "\nRMSE < ₹2,000 : PASSED"
    )

else:

    print(
        "\nRMSE < ₹2,000 : FAILED"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    COMPARISON_PATH,
    index=False
)


print("\n" + "-" * 70)
print("COMPARISON RESULTS SAVED")
print("-" * 70)

print(
    f"\nFile:"
)

print(
    COMPARISON_PATH
)

print("\n" + "=" * 70)
print(
    "REGRESSION MODEL COMPARISON "
    "COMPLETED SUCCESSFULLY"
)
print("=" * 70)