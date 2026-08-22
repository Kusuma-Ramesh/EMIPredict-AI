from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
    / "emi_eligibility_classifier.joblib"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "classification_results.txt"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "classification_confusion_matrix.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("EMIPredict AI - EMI ELIGIBILITY CLASSIFIER")
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
# 1. ENCODE TARGET
# ============================================================

target_mapping = {
    "Not_Eligible": 0,
    "Eligible": 1,
    "High_Risk": 2,
}

inverse_target_mapping = {
    0: "Not_Eligible",
    1: "Eligible",
    2: "High_Risk",
}

y_train_encoded = y_train.map(target_mapping)
y_test_encoded = y_test.map(target_mapping)


# Safety check
if y_train_encoded.isnull().any():
    raise ValueError(
        "Unknown classification target found in training data."
    )

if y_test_encoded.isnull().any():
    raise ValueError(
        "Unknown classification target found in test data."
    )


# ============================================================
# 2. IDENTIFY FEATURE TYPES
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
# 3. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ============================================================
# 4. CATEGORICAL PREPROCESSING
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
# 5. COMBINED PREPROCESSOR
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
# 6. XGBOOST CLASSIFIER
# ============================================================

classifier = XGBClassifier(
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
)


# ============================================================
# 7. COMPLETE ML PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        ),
    ]
)


# ============================================================
# 8. HANDLE CLASS IMBALANCE
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
# 9. TRAIN MODEL
# ============================================================

print("\n" + "-" * 70)
print("MODEL TRAINING")
print("-" * 70)

print("\nTraining XGBoost classifier...")
print("This may take some time because the dataset contains")
print("over 320,000 training records.")

model_pipeline.fit(
    X_train,
    y_train_encoded,
    classifier__sample_weight=sample_weights
)

print("\nModel training completed.")


# ============================================================
# 10. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred_encoded = model_pipeline.predict(
    X_test
)

y_pred_probability = model_pipeline.predict_proba(
    X_test
)


# ============================================================
# 11. CONVERT PREDICTIONS BACK TO LABELS
# ============================================================

y_pred = pd.Series(
    y_pred_encoded
).map(inverse_target_mapping)

y_test_labels = pd.Series(
    y_test_encoded
).map(inverse_target_mapping)


# ============================================================
# 12. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test_labels,
    y_pred
)

macro_f1 = f1_score(
    y_test_labels,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test_labels,
    y_pred,
    average="weighted"
)

report = classification_report(
    y_test_labels,
    y_pred,
    labels=[
        "Not_Eligible",
        "Eligible",
        "High_Risk",
        ],
    digits=4
)

cm = confusion_matrix(
    y_test_labels,
    y_pred,
    labels=[
        "Not_Eligible",
        "Eligible",
        "High_Risk",
    ]
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)

print(f"\nAccuracy       : {accuracy:.4f}")
print(f"Macro F1 Score : {macro_f1:.4f}")
print(f"Weighted F1    : {weighted_f1:.4f}")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        cm,
        index=[
            "Actual_Not_Eligible",
            "Actual_Eligible",
            "Actual_High_Risk",
        ],
        columns=[
            "Pred_Not_Eligible",
            "Pred_Eligible",
            "Pred_High_Risk",
        ],
    )
)


# ============================================================
# 14. CREATE OUTPUT DIRECTORIES
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
# 15. SAVE MODEL
# ============================================================

joblib.dump(
    model_pipeline,
    MODEL_PATH
)


# ============================================================
# 16. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Not_Eligible",
        "Actual_Eligible",
        "Actual_High_Risk",
    ],
    columns=[
        "Pred_Not_Eligible",
        "Pred_Eligible",
        "Pred_High_Risk",
    ]
)

cm_df.to_csv(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# 17. SAVE MODEL RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EMIPredict AI - EMI Eligibility Classification Results\n"
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
        f"Accuracy       : {accuracy:.4f}\n"
    )

    file.write(
        f"Macro F1 Score : {macro_f1:.4f}\n"
    )

    file.write(
        f"Weighted F1    : {weighted_f1:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write("-" * 70 + "\n")

    file.write(report)

    file.write("\n\nConfusion Matrix\n")
    file.write("-" * 70 + "\n")

    file.write(
        cm_df.to_string()
    )


# ============================================================
# 18. COMPLETION
# ============================================================

print("\n" + "-" * 70)
print("MODEL SAVED")
print("-" * 70)

print(f"\nModel:")
print(MODEL_PATH)

print("\nResults:")
print(RESULTS_PATH)

print("\nConfusion matrix:")
print(CONFUSION_MATRIX_PATH)

print("\n" + "=" * 70)
print("CLASSIFICATION TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)