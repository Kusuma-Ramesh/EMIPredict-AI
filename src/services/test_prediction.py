from src.services.prediction_service import predict_emi


# ============================================================
# TEST CUSTOMER
# ============================================================

customer = {
    "age": 32,
    "gender": "Male",
    "marital_status": "Married",
    "education": "Graduate",
    "monthly_salary": 75000,
    "employment_type": "Private",
    "years_of_employment": 6,
    "company_type": "MNC",
    "house_type": "Own",
    "monthly_rent": 0,
    "family_size": 3,
    "dependents": 1,
    "school_fees": 5000,
    "college_fees": 0,
    "travel_expenses": 5000,
    "groceries_utilities": 12000,
    "other_monthly_expenses": 6000,
    "existing_loans": "No",
    "current_emi_amount": 0,
    "credit_score": 750,
    "bank_balance": 250000,
    "emergency_fund": 100000,
    "emi_scenario": "Vehicle EMI",
    "requested_amount": 500000,
    "requested_tenure": 36,
}


# ============================================================
# RUN PREDICTION
# ============================================================

result = predict_emi(
    customer
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("EMIPredict-AI PREDICTION TEST")
print("=" * 70)

print(
    f"\nEMI Eligibility:"
)

print(
    result["emi_eligibility"]
)

print(
    "\nEligibility Probabilities:"
)

for label, probability in result[
    "eligibility_probabilities"
].items():

    print(
        f"  {label}: {probability}%"
    )

print(
    "\nPredicted Maximum Monthly EMI:"
)

print(
    f"₹{result['predicted_max_monthly_emi']:,.2f}"
)

print(
    "\nRecommendation:"
)

print(
    result["recommendation"]
)

print(
    "\n" + "=" * 70
)