from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
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
# TARGET MAPPINGS
# ============================================================

CLASS_MAPPING = {
    0: "Not_Eligible",
    1: "Eligible",
    2: "High_Risk",
}


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading EMIPredict-AI models...")

classifier = joblib.load(
    CLASSIFIER_PATH
)

regressor = joblib.load(
    REGRESSOR_PATH
)

print("Classification model loaded.")

print("Regression model loaded.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(customer_data: dict) -> pd.DataFrame:
    """
    Convert raw customer input into the feature structure
    expected by the trained ML pipelines.
    """

    df = pd.DataFrame(
        [customer_data]
    )

    # --------------------------------------------------------
    # Financial features
    # --------------------------------------------------------

    df["total_monthly_expenses"] = (
        df["monthly_rent"]
        + df["school_fees"]
        + df["college_fees"]
        + df["travel_expenses"]
        + df["groceries_utilities"]
        + df["other_monthly_expenses"]
    )

    df["total_existing_obligations"] = (
        df["current_emi_amount"]
    )

    df["disposable_income"] = (
        df["monthly_salary"]
        - df["total_monthly_expenses"]
        - df["current_emi_amount"]
    )

    df["current_emi_ratio"] = (
        df["current_emi_amount"]
        / df["monthly_salary"].replace(0, pd.NA)
    )

    df["expense_ratio"] = (
        df["total_monthly_expenses"]
        / df["monthly_salary"].replace(0, pd.NA)
    )

    df["loan_to_annual_income"] = (
        df["requested_amount"]
        / (
            df["monthly_salary"] * 12
        ).replace(0, pd.NA)
    )

    df["emergency_fund_coverage"] = (
        df["emergency_fund"]
        / df["total_monthly_expenses"].replace(
            0,
            pd.NA
        )
    )

    df["bank_balance_to_loan_ratio"] = (
        df["bank_balance"]
        / df["requested_amount"].replace(
            0,
            pd.NA
        )
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

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
            .replace(
                [float("inf"), float("-inf")],
                pd.NA
            )
        )

    return df


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emi(customer_data: dict) -> dict:
    """
    Generate EMI eligibility and maximum EMI prediction
    for a single customer.
    """

    # --------------------------------------------------------
    # Build ML features
    # --------------------------------------------------------

    features = build_features(
        customer_data
    )

    # --------------------------------------------------------
    # Classification prediction
    # --------------------------------------------------------

    class_prediction = classifier.predict(
        features
    )[0]

    class_probabilities = classifier.predict_proba(
        features
    )[0]

    eligibility = CLASS_MAPPING[
        int(class_prediction)
    ]

    # --------------------------------------------------------
    # Classification probabilities
    # --------------------------------------------------------

    probabilities = {
        CLASS_MAPPING[index]: round(
            float(probability) * 100,
            2
        )
        for index, probability
        in enumerate(class_probabilities)
    }

    # --------------------------------------------------------
    # Regression prediction
    # --------------------------------------------------------

    predicted_emi = regressor.predict(
        features
    )[0]

    # EMI cannot be negative
    predicted_emi = max(
        0,
        float(predicted_emi)
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if eligibility == "Eligible":

        recommendation = (
            "The applicant appears eligible "
            "for the requested EMI based on "
            "the financial profile."
        )

    elif eligibility == "High_Risk":

        recommendation = (
            "The applicant is classified as "
            "high risk. Consider reducing the "
            "requested amount or increasing the "
            "repayment tenure."
        )

    else:

        recommendation = (
            "The applicant is currently classified "
            "as not eligible. Consider reducing "
            "the requested amount or improving "
            "the overall affordability profile."
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {
        "emi_eligibility": eligibility,

        "eligibility_probabilities": probabilities,

        "predicted_max_monthly_emi": round(
            predicted_emi,
            2
        ),

        "recommendation": recommendation,
    }

    return result