from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

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


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "EMIPredict_Regression_Final"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("EMIPredict AI - MLFLOW REGRESSION EXPERIMENT")
print("=" * 70)

print("\nLoading dataset...")

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
# MAPE
# ============================================================

def calculate_mape(y_true, y_pred):

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

    "Linear_Regression": LinearRegression(),

    "Random_Forest": RandomForestRegressor(
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
# TRAIN + MLFLOW TRACKING
# ============================================================

results = []


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

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

    with mlflow.start_run(
        run_name=model_name
    ):

        print("\nTraining model...")

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

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # LOG PARAMETERS
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            model_name
        )

        mlflow.log_param(
            "numerical_features",
            len(numeric_features)
        )

        mlflow.log_param(
            "categorical_features",
            len(categorical_features)
        )

        # Model-specific parameters
        model_params = model.get_params()

        for key, value in model_params.items():

            try:
                mlflow.log_param(
                    key,
                    value
                )
            except Exception:
                pass

        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        mlflow.log_metric(
            "MAE",
            float(mae)
        )

        mlflow.log_metric(
            "RMSE",
            float(rmse)
        )

        mlflow.log_metric(
            "R2",
            float(r2)
        )

        mlflow.log_metric(
            "MAPE",
            float(mape)
        )

        # ----------------------------------------------------
        # ACCEPTANCE CHECK
        # ----------------------------------------------------

        acceptance_passed = rmse < 2000

        mlflow.log_param(
            "rmse_acceptance_threshold",
            2000
        )

        mlflow.log_param(
            "rmse_acceptance_passed",
            acceptance_passed
        )

        # ----------------------------------------------------
        # LOG MODEL
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="pickle"
        )

        print(
            "\nMLflow run logged successfully."
        )

        print(
            "Run ID:",
            mlflow.active_run().info.run_id
        )

        results.append(
            {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape,
                "Run_ID":
                    mlflow.active_run().info.run_id,
            }
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]


print("\n" + "=" * 70)
print("BEST REGRESSION MODEL")
print("=" * 70)

print(
    f"\nModel : {best_model['Model']}"
)

print(
    f"MAE   : ₹{best_model['MAE']:,.2f}"
)

print(
    f"RMSE  : ₹{best_model['RMSE']:,.2f}"
)

print(
    f"R²    : {best_model['R2']:.4f}"
)

print(
    f"MAPE  : {best_model['MAPE']:.2f}%"
)

print(
    f"Run ID: {best_model['Run_ID']}"
)


# ============================================================
# SAVE MLFLOW COMPARISON
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    RESULTS_DIR
    / "mlflow_regression_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


print("\n" + "-" * 70)
print("MLFLOW REGRESSION COMPARISON SAVED")
print("-" * 70)

print(
    f"\nFile: {output_path}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print(
    "MLFLOW REGRESSION EXPERIMENT COMPLETED SUCCESSFULLY"
)
print("=" * 70)