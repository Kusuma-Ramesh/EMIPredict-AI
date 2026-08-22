from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


# ============================================================
# VALID TEST DATA
# ============================================================

VALID_DATA = {
    "age": 32,
    "gender": "Male",
    "marital_status": "Married",
    "education": "Graduate",
    "monthly_salary": 60000,
    "employment_type": "Private",
    "years_of_employment": 6,
    "company_type": "MNC",
    "house_type": "Own",
    "monthly_rent": 0,
    "family_size": 3,
    "dependents": 2,
    "school_fees": 5000,
    "college_fees": 0,
    "travel_expenses": 5000,
    "groceries_utilities": 12000,
    "other_monthly_expenses": 5000,
    "existing_loans": "No",
    "current_emi_amount": 0,
    "credit_score": 750,
    "bank_balance": 250000,
    "emergency_fund": 100000,
    "emi_scenario": "Vehicle EMI",
    "requested_amount": 400000,
    "requested_tenure": 36
}


# ============================================================
# ROOT ENDPOINT TEST
# ============================================================

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


# ============================================================
# HEALTH ENDPOINT TEST
# ============================================================

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["classifier"] == "loaded"
    assert data["regressor"] == "loaded"


# ============================================================
# VALID PREDICTION TEST
# ============================================================

def test_valid_prediction():
    response = client.post(
        "/predict",
        json=VALID_DATA
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "prediction" in data

    prediction = data["prediction"]

    assert "emi_eligibility" in prediction
    assert "eligibility_probabilities" in prediction
    assert "predicted_max_monthly_emi" in prediction
    assert "recommendation" in prediction


# ============================================================
# INVALID AGE TEST
# ============================================================

def test_invalid_age():
    invalid_data = VALID_DATA.copy()

    invalid_data["age"] = 10

    response = client.post(
        "/predict",
        json=invalid_data
    )

    assert response.status_code == 422


# ============================================================
# INVALID CREDIT SCORE TEST
# ============================================================

def test_invalid_credit_score():
    invalid_data = VALID_DATA.copy()

    invalid_data["credit_score"] = 900

    response = client.post(
        "/predict",
        json=invalid_data
    )

    assert response.status_code == 422


# ============================================================
# MISSING REQUIRED FIELD TEST
# ============================================================

def test_missing_required_field():
    invalid_data = VALID_DATA.copy()

    del invalid_data["monthly_salary"]

    response = client.post(
        "/predict",
        json=invalid_data
    )

    assert response.status_code == 422