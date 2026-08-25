from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# DATABASE / CRUD
# ============================================================

from database.database import Base, engine, get_db
from database.models import Customer, Prediction
from schemas.customer import (
    CustomerCreate,
    CustomerUpdate,
    CustomerResponse,
    PredictionResponse,
)
from services.customer_service import (
    create_customer,
    get_customers,
    get_customer,
    update_customer,
    delete_customer,
)
# ============================================================
# DATABASE INITIALIZATION
# ============================================================

Base.metadata.create_all(bind=engine)

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "classification_model": "online",
        "regression_model": "online",
        "database": "online",
        "mlflow": "online",
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: EMIPredictionRequest,
    db: Session = Depends(get_db),
):

    try:

        customer_data = request.model_dump()

        # ----------------------------------------------------
        # RUN ML PREDICTION
        # ----------------------------------------------------

        result = predict_emi(
            customer_data
        )

        # ----------------------------------------------------
        # SAVE CUSTOMER
        # ----------------------------------------------------

        customer = Customer(
            **customer_data
        )

        db.add(customer)
        db.commit()
        db.refresh(customer)

        # ----------------------------------------------------
        # EXTRACT PREDICTION VALUES
        # ----------------------------------------------------

        probabilities = result.get(
            "eligibility_probabilities",
            {}
        )

        emi_status = result.get(
            "emi_eligibility",
            "Unknown"
        )

        predicted_max_emi = result.get(
            "predicted_max_monthly_emi",
            0
        )

        recommendation = result.get(
            "recommendation",
            ""
        )

        # ----------------------------------------------------
        # CALCULATE REQUESTED MONTHLY EMI
        # ----------------------------------------------------

        requested_amount = request.requested_amount
        requested_tenure = request.requested_tenure

        requested_monthly_emi = (
            requested_amount / requested_tenure
        )

        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        prediction_record = Prediction(
            customer_id=customer.id,

            emi_status=emi_status,

            not_eligible_probability=probabilities.get(
                "Not_Eligible",
                0
            ),

            eligible_probability=probabilities.get(
                "Eligible",
                0
            ),

            high_risk_probability=probabilities.get(
                "High_Risk",
                0
            ),

            predicted_max_monthly_emi=predicted_max_emi,

            requested_monthly_emi=requested_monthly_emi,

            recommendation=recommendation,
        )

        db.add(prediction_record)

        db.commit()

        db.refresh(prediction_record)

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {
            "success": True,
            "customer_id": customer.id,
            "prediction": result,
        }

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(error)}"
            ),
        )
# ============================================================
# PREDICTION HISTORY ENDPOINTS
# ============================================================

@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
)
def get_predictions_endpoint(
    db: Session = Depends(get_db),
):
    predictions = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return [
        PredictionResponse(
            prediction_id=prediction.id,
            customer_id=prediction.customer_id,
            emi_status=prediction.emi_status,
            probability_not_eligible=prediction.not_eligible_probability,
            probability_eligible=prediction.eligible_probability,
            probability_high_risk=prediction.high_risk_probability,
            max_recommended_emi=prediction.predicted_max_monthly_emi,
            requested_emi=prediction.requested_monthly_emi,
            recommendation=prediction.recommendation,
            created_at=prediction.created_at,
        )
        for prediction in predictions
    ]


@app.get(
    "/predictions/{prediction_id}",
    response_model=PredictionResponse,
)
def get_prediction_endpoint(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return PredictionResponse(
        prediction_id=prediction.id,
        customer_id=prediction.customer_id,
        emi_status=prediction.emi_status,
        probability_not_eligible=prediction.not_eligible_probability,
        probability_eligible=prediction.eligible_probability,
        probability_high_risk=prediction.high_risk_probability,
        max_recommended_emi=prediction.predicted_max_monthly_emi,
        requested_emi=prediction.requested_monthly_emi,
        recommendation=prediction.recommendation,
        created_at=prediction.created_at,
    )
# ============================================================
# CUSTOMER CRUD ENDPOINTS
# ============================================================

@app.post(
    "/customers",
    response_model=CustomerResponse,
    status_code=201,
)
def create_customer_endpoint(
    customer_data: CustomerCreate,
    db: Session = Depends(get_db),
):
    return create_customer(
        db,
        customer_data,
    )


@app.get(
    "/customers",
    response_model=list[CustomerResponse],
)
def get_customers_endpoint(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    return get_customers(
        db,
        skip,
        limit,
    )


@app.get(
    "/customers/{customer_id}",
    response_model=CustomerResponse,
)
def get_customer_endpoint(
    customer_id: int,
    db: Session = Depends(get_db),
):
    customer = get_customer(
        db,
        customer_id,
    )

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found",
        )

    return customer


@app.put(
    "/customers/{customer_id}",
    response_model=CustomerResponse,
)
def update_customer_endpoint(
    customer_id: int,
    customer_data: CustomerUpdate,
    db: Session = Depends(get_db),
):
    customer = get_customer(
        db,
        customer_id,
    )

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found",
        )

    return update_customer(
        db,
        customer,
        customer_data,
    )


@app.delete(
    "/customers/{customer_id}",
    status_code=204,
)
def delete_customer_endpoint(
    customer_id: int,
    db: Session = Depends(get_db),
):
    customer = get_customer(
        db,
        customer_id,
    )

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found",
        )

    delete_customer(
        db,
        customer,
    )

    return None