from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


CLASSIFIER_PATH = (
    MODEL_DIR
    / "emi_eligibility_classifier.joblib"
)

REGRESSOR_PATH = (
    MODEL_DIR
    / "emi_amount_regressor.joblib"
)


# ============================================================
# OUTPUT FILES
# ============================================================

EVALUATION_REPORT = (
    RESULTS_DIR
    / "model_evaluation_report.txt"
)

CLASSIFICATION_CM = (
    FIGURES_DIR
    / "classification_confusion_matrix.png"
)

REGRESSION_ACTUAL_PREDICTED = (
    FIGURES_DIR
    / "regression_actual_vs_predicted.png"
)

REGRESSION_RESIDUALS = (
    FIGURES_DIR
    / "regression_residuals.png"
)

FEATURE_IMPORTANCE = (
    FIGURES_DIR
    / "regression_feature_importance.png"
)


# ============================================================
# DIRECTORIES
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("EMIPredict AI - MODEL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test datasets...")

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_class_test = pd.read_csv(
    DATA_DIR / "y_class_test.csv"
).iloc[:, 0]

y_reg_test = pd.read_csv(
    DATA_DIR / "y_reg_test.csv"
).iloc[:, 0]


print(f"\nX_test shape: {X_test.shape}")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading trained models...")

classifier = joblib.load(
    CLASSIFIER_PATH
)

regressor = joblib.load(
    REGRESSOR_PATH
)

print("Classifier loaded successfully.")
print("Regressor loaded successfully.")


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("1. CLASSIFICATION MODEL EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

class_predictions = classifier.predict(
    X_test
)

class_probabilities = classifier.predict_proba(
    X_test
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

classification_accuracy = accuracy_score(
    y_class_test,
    pd.Series(class_predictions).map(
        {
            0: "Not_Eligible",
            1: "Eligible",
            2: "High_Risk",
        }
    )
)

classification_f1 = f1_score(
    y_class_test,
    pd.Series(class_predictions).map(
        {
            0: "Not_Eligible",
            1: "Eligible",
            2: "High_Risk",
        }
    ),
    average="macro"
)

classification_weighted_f1 = f1_score(
    y_class_test,
    pd.Series(class_predictions).map(
        {
            0: "Not_Eligible",
            1: "Eligible",
            2: "High_Risk",
        }
    ),
    average="weighted"
)


class_predictions_labels = (
    pd.Series(class_predictions)
    .map(
        {
            0: "Not_Eligible",
            1: "Eligible",
            2: "High_Risk",
        }
    )
)


classification_report_text = classification_report(
    y_class_test,
    class_predictions_labels,
    digits=4
)


print(
    f"\nAccuracy       : "
    f"{classification_accuracy:.4f}"
)

print(
    f"Macro F1       : "
    f"{classification_f1:.4f}"
)

print(
    f"Weighted F1    : "
    f"{classification_weighted_f1:.4f}"
)

print("\nClassification Report:")
print(classification_report_text)


# ============================================================
# CONFUSION MATRIX
# ============================================================

class_labels = [
    "Not_Eligible",
    "Eligible",
    "High_Risk",
]

cm = confusion_matrix(
    y_class_test,
    class_predictions_labels,
    labels=class_labels
)

cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
)


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "EMI Eligibility - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.savefig(
    CLASSIFICATION_CM,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nConfusion matrix saved:"
)

print(
    CLASSIFICATION_CM
)


# ============================================================
# REGRESSION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("2. REGRESSION MODEL EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

regression_predictions = regressor.predict(
    X_test
)

regression_predictions = np.maximum(
    regression_predictions,
    0
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_reg_test,
    regression_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        regression_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    regression_predictions
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


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    y_reg_test,
    regression_predictions,
    alpha=0.25,
    s=10
)

min_value = min(
    y_reg_test.min(),
    regression_predictions.min()
)

max_value = max(
    y_reg_test.max(),
    regression_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Maximum Monthly EMI"
)

plt.xlabel(
    "Actual EMI (₹)"
)

plt.ylabel(
    "Predicted EMI (₹)"
)

plt.tight_layout()

plt.savefig(
    REGRESSION_ACTUAL_PREDICTED,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

residuals = (
    y_reg_test.values
    - regression_predictions
)


plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    regression_predictions,
    residuals,
    alpha=0.25,
    s=10
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title(
    "Regression Residual Analysis"
)

plt.xlabel(
    "Predicted EMI (₹)"
)

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.tight_layout()

plt.savefig(
    REGRESSION_RESIDUALS,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "-" * 70)
print("REGRESSION FEATURE IMPORTANCE")
print("-" * 70)


try:

    preprocessing = (
        regressor
        .named_steps["preprocessor"]
    )

    xgb_model = (
        regressor
        .named_steps["regressor"]
    )

    feature_names = (
        preprocessing
        .get_feature_names_out()
    )

    importances = (
        xgb_model
        .feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(
        "\nTop 15 regression features:"
    )

    print(
        importance_df
        .head(15)
        .to_string(index=False)
    )


    # --------------------------------------------------------
    # Plot top 15 features
    # --------------------------------------------------------

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.title(
        "Top 15 Features - EMI Amount Regression"
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nFeature importance saved:"
    )

    print(
        FEATURE_IMPORTANCE
    )

except Exception as error:

    print(
        "\nFeature importance could not be generated."
    )

    print(
        f"Reason: {error}"
    )


# ============================================================
# RESIDUAL SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("RESIDUAL SUMMARY")
print("-" * 70)

print(
    f"\nMean residual       : "
    f"₹{residuals.mean():,.2f}"
)

print(
    f"Median residual     : "
    f"₹{np.median(residuals):,.2f}"
)

print(
    f"Mean absolute error : "
    f"₹{np.mean(np.abs(residuals)):,.2f}"
)

print(
    f"Maximum error       : "
    f"₹{np.max(np.abs(residuals)):,.2f}"
)


# ============================================================
# SAVE EVALUATION REPORT
# ============================================================

with open(
    EVALUATION_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EMIPredict AI - MODEL EVALUATION REPORT\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "CLASSIFICATION MODEL\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Accuracy       : "
        f"{classification_accuracy:.4f}\n"
    )

    file.write(
        f"Macro F1       : "
        f"{classification_f1:.4f}\n"
    )

    file.write(
        f"Weighted F1    : "
        f"{classification_weighted_f1:.4f}\n\n"
    )

    file.write(
        classification_report_text
    )

    file.write(
        "\n\nREGRESSION MODEL\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"MAE  : ₹{mae:,.2f}\n"
    )

    file.write(
        f"RMSE : ₹{rmse:,.2f}\n"
    )

    file.write(
        f"R²   : {r2:.4f}\n\n"
    )

    file.write(
        "RESIDUAL SUMMARY\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Mean residual       : "
        f"₹{residuals.mean():,.2f}\n"
    )

    file.write(
        f"Median residual     : "
        f"₹{np.median(residuals):,.2f}\n"
    )

    file.write(
        f"Mean absolute error : "
        f"₹{np.mean(np.abs(residuals)):,.2f}\n"
    )

    file.write(
        f"Maximum error       : "
        f"₹{np.max(np.abs(residuals)):,.2f}\n"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated evaluation figures:")

print(
    f"1. {CLASSIFICATION_CM}"
)

print(
    f"2. {REGRESSION_ACTUAL_PREDICTED}"
)

print(
    f"3. {REGRESSION_RESIDUALS}"
)

print(
    f"4. {FEATURE_IMPORTANCE}"
)

print("\nEvaluation report:")

print(
    EVALUATION_REPORT
)

print("\n" + "=" * 70)