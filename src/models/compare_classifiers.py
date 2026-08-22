from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


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
    / "classification_model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("EMIPredict AI - CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

print("\nLoading training and testing data...")

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "y_class_train.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    DATA_DIR / "y_class_test.csv"
).iloc[:, 0]


print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

print("\nTraining target distribution:")
print(y_train.value_counts())


# ============================================================
# TARGET ENCODING
# ============================================================

target_mapping = {
    "Not_Eligible": 0,
    "Eligible": 1,
    "High_Risk": 2,
}

y_train_encoded = y_train.map(target_mapping)
y_test_encoded = y_test.map(target_mapping)

if y_train_encoded.isnull().any():
    raise ValueError(
        "Unknown classification target found in training data."
    )

if y_test_encoded.isnull().any():
    raise ValueError(
        "Unknown classification target found in test data."
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
# CLASS BALANCING
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_encoded
)

print("\n" + "-" * 70)
print("CLASS BALANCING")
print("-" * 70)

print("\nBalanced sample weights calculated.")


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
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
                "classifier",
                model
            ),
        ]
    )

    print(
        f"\nTraining {model_name}..."
    )

    pipeline.fit(
        X_train,
        y_train_encoded,
        classifier__sample_weight=sample_weights
    )

    print("Training completed.")

    print("Generating predictions...")

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test_encoded,
        y_pred
    )

    precision = precision_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test_encoded,
        y_probability,
        multi_class="ovr",
        average="macro"
    )

    print(
        f"\nAccuracy       : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision      : "
        f"{precision:.4f}"
    )

    print(
        f"Recall         : "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"ROC-AUC        : "
        f"{roc_auc:.4f}"
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "ROC_AUC": roc_auc,
        }
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "Macro_F1": "{:.4f}".format,
            "Weighted_F1": "{:.4f}".format,
            "ROC_AUC": "{:.4f}".format,
        }
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "-" * 70)
print("BEST CLASSIFICATION MODEL")
print("-" * 70)

print(
    f"\nModel      : "
    f"{best_model['Model']}"
)

print(
    f"Accuracy   : "
    f"{best_model['Accuracy']:.4f}"
)

print(
    f"Precision  : "
    f"{best_model['Precision']:.4f}"
)

print(
    f"Recall     : "
    f"{best_model['Recall']:.4f}"
)

print(
    f"Macro F1   : "
    f"{best_model['Macro_F1']:.4f}"
)

print(
    f"ROC-AUC    : "
    f"{best_model['ROC_AUC']:.4f}"
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
    "CLASSIFICATION MODEL COMPARISON "
    "COMPLETED SUCCESSFULLY"
)
print("=" * 70)