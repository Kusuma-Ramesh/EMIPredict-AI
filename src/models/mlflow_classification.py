
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ml_ready"


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("EMIPredict_Classification_Final")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_class_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_class_test.csv").iloc[:, 0]


target_mapping = {
    "Not_Eligible": 0,
    "Eligible": 1,
    "High_Risk": 2
}

y_train = y_train.map(target_mapping)
y_test = y_test.map(target_mapping)


# ============================================================
# FEATURE DETECTION
# ============================================================

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
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
        )
    ]
)


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic_Regression":

    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random_Forest":

    RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost":

    XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN + LOG
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    with mlflow.start_run(run_name=model_name):

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weights
        )

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_param(
            "training_rows",
            len(X_train)
        )

        mlflow.log_param(
            "feature_count",
            X_train.shape[1]
        )

        if model_name == "Logistic_Regression":

            mlflow.log_param("max_iter",1000)

        elif model_name == "Random_Forest":

            mlflow.log_params({
                "n_estimators":200,
                "max_depth":15,
                "min_samples_split":5
            })

        elif model_name == "XGBoost":

            mlflow.log_params({
                "n_estimators":250,
                "max_depth":6,
                "learning_rate":0.08,
                "subsample":0.85,
                "colsample_bytree":0.85,
                "min_child_weight":2,
                "reg_alpha":0.1,
                "reg_lambda":1.0
            })

        mlflow.log_metrics({

            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "macro_f1":macro_f1,
            "weighted_f1":weighted_f1,
            "roc_auc":roc_auc

        })

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle"
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Macro F1 : {macro_f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification experiments logged successfully.")