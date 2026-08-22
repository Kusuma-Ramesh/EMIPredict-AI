from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PREDICTION SERVICE
# ============================================================

from services.prediction_service import predict_emi


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="EMIPredict-AI",
    description=(
        "AI-powered EMI eligibility and maximum "
        "monthly EMI prediction API."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class EMIPredictionRequest(BaseModel):

    age: float = Field(..., ge=18, le=100)
    gender: str
    marital_status: str
    education: str

    monthly_salary: float = Field(
        ...,
        gt=0
    )

    employment_type: str
    years_of_employment: float = Field(
        ...,
        ge=0
    )

    company_type: str
    house_type: str

    monthly_rent: float = Field(
        ...,
        ge=0
    )

    family_size: int = Field(
        ...,
        ge=1
    )

    dependents: int = Field(
        ...,
        ge=0
    )

    school_fees: float = Field(
        ...,
        ge=0
    )

    college_fees: float = Field(
        ...,
        ge=0
    )

    travel_expenses: float = Field(
        ...,
        ge=0
    )

    groceries_utilities: float = Field(
        ...,
        ge=0
    )

    other_monthly_expenses: float = Field(
        ...,
        ge=0
    )

    existing_loans: str

    current_emi_amount: float = Field(
        ...,
        ge=0
    )

    credit_score: float = Field(
        ...,
        ge=300,
        le=850
    )

    bank_balance: float = Field(
        ...,
        ge=0
    )

    emergency_fund: float = Field(
        ...,
        ge=0
    )

    emi_scenario: str

    requested_amount: float = Field(
        ...,
        gt=0
    )

    requested_tenure: int = Field(
        ...,
        ge=1
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "EMIPredict-AI",
        "status": "running",
        "version": "1.0.0",
        "message": "EMI prediction backend is operational.",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "classifier": "loaded",
        "regressor": "loaded",
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: EMIPredictionRequest):

    try:

        customer_data = request.model_dump()

        result = predict_emi(
            customer_data
        )

        return {
            "success": True,
            "prediction": result,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(error)}"
            ),
        )