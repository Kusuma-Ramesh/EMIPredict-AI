import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8001"


st.set_page_config(
    page_title="EMIPredict-AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPER — API REQUEST
# ============================================================

def predict_emi(customer_data):

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()

        st.error(
            f"Prediction failed ({response.status_code}): "
            f"{response.text}"
        )

        return None

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the EMIPredict-AI backend. "
            "Make sure FastAPI is running on port 8001."
        )

        return None

    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )

        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("EMIPredict-AI")

    st.caption(
        "AI-powered EMI eligibility and affordability prediction"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "EMI Prediction",
            "Customer Management",
            "Prediction History",
            "System Status",
        ],
    )

    st.divider()

    st.caption("EMIPredict-AI v1.0.0")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("EMIPredict-AI")
    st.subheader("AI-Powered EMI Decision Support System")

    st.write(
        """
        Evaluate a customer's financial profile using machine
        learning to determine EMI eligibility and estimate the
        maximum affordable monthly EMI.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Classification", "XGBoost")

    with col2:
        st.metric("Regression", "XGBoost")

    with col3:
        st.metric("Backend", "FastAPI")

    with col4:
        st.metric("Model Tracking", "MLflow")

    st.divider()

    st.success(
        "EMIPredict-AI backend is ready for predictions."
    )


# ============================================================
# EMI PREDICTION
# ============================================================

elif page == "EMI Prediction":

    st.title("EMI Eligibility Prediction")

    st.write(
        "Enter the applicant's financial profile below."
    )

    st.divider()

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("1. Personal Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"],
        )

    with col3:
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced", "Widowed"],
        )

    with col4:
        education = st.selectbox(
            "Education",
            [
                "High School",
                "Diploma",
                "Graduate",
                "Post Graduate",
                "Doctorate",
            ],
        )

    # --------------------------------------------------------
    # EMPLOYMENT
    # --------------------------------------------------------

    st.subheader("2. Employment Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        monthly_salary = st.number_input(
            "Monthly Salary (₹)",
            min_value=1.0,
            value=60000.0,
            step=1000.0,
        )

    with col2:
        employment_type = st.selectbox(
            "Employment Type",
            [
                "Salaried",
                "Self-Employed",
                "Business",
                "Government",
                "Other",
            ],
        )

    with col3:
        years_of_employment = st.number_input(
            "Years of Employment",
            min_value=0.0,
            value=5.0,
            step=0.5,
        )

    with col4:
        company_type = st.selectbox(
            "Company Type",
            [
                "Private",
                "Government",
                "Public",
                "Self-Employed",
                "Other",
            ],
        )

    # --------------------------------------------------------
    # HOUSING
    # --------------------------------------------------------

    st.subheader("3. Housing & Family")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        house_type = st.selectbox(
            "House Type",
            [
                "Own",
                "Rented",
                "Family",
                "Other",
            ],
        )

    with col2:
        monthly_rent = st.number_input(
            "Monthly Rent (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0,
        )

    with col3:
        family_size = st.number_input(
            "Family Size",
            min_value=1,
            value=4,
            step=1,
        )

    with col4:
        dependents = st.number_input(
            "Dependents",
            min_value=0,
            value=2,
            step=1,
        )

    # --------------------------------------------------------
    # MONTHLY EXPENSES
    # --------------------------------------------------------

    st.subheader("4. Monthly Expenses")

    col1, col2, col3 = st.columns(3)

    with col1:
        school_fees = st.number_input(
            "School Fees (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0,
        )

        travel_expenses = st.number_input(
            "Travel Expenses (₹)",
            min_value=0.0,
            value=3000.0,
            step=500.0,
        )

    with col2:
        college_fees = st.number_input(
            "College Fees (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0,
        )

        groceries_utilities = st.number_input(
            "Groceries & Utilities (₹)",
            min_value=0.0,
            value=8000.0,
            step=500.0,
        )

    with col3:
        other_monthly_expenses = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0.0,
            value=2000.0,
            step=500.0,
        )

        existing_loans = st.selectbox(
            "Existing Loans",
            ["No", "Yes"],
        )

    # --------------------------------------------------------
    # FINANCIAL PROFILE
    # --------------------------------------------------------

    st.subheader("5. Financial Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_emi_amount = st.number_input(
            "Current EMI (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0,
        )

    with col2:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300.0,
            max_value=850.0,
            value=750.0,
            step=1.0,
        )

    with col3:
        bank_balance = st.number_input(
            "Bank Balance (₹)",
            min_value=0.0,
            value=150000.0,
            step=5000.0,
        )

    with col4:
        emergency_fund = st.number_input(
            "Emergency Fund (₹)",
            min_value=0.0,
            value=100000.0,
            step=5000.0,
        )

    # --------------------------------------------------------
    # EMI REQUEST
    # --------------------------------------------------------

    st.subheader("6. Requested EMI")

    col1, col2, col3 = st.columns(3)

    with col1:
        emi_scenario = st.selectbox(
            "EMI Scenario",
            [
                "Home Loan",
                "Personal Loan",
                "Vehicle Loan",
                "Education Loan",
                "Consumer Loan",
                "Other",
            ],
        )

    with col2:
        requested_amount = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=1.0,
            value=500000.0,
            step=10000.0,
        )

    with col3:
        requested_tenure = st.number_input(
            "Requested Tenure (months)",
            min_value=1,
            value=60,
            step=1,
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predict_button = st.button(
        "Predict EMI Eligibility",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        customer_data = {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
        }

        with st.spinner(
            "Running EMIPredict-AI models..."
        ):

            result = predict_emi(customer_data)

        if result and result.get("success"):

            prediction = result["prediction"]

            st.divider()

            st.subheader("Prediction Result")

            eligibility = prediction.get(
                "emi_eligibility",
                "Unknown",
            )

            max_emi = prediction.get(
                "predicted_max_monthly_emi",
                0,
            )

            recommendation = prediction.get(
                "recommendation",
                "",
            )

            probabilities = prediction.get(
                "eligibility_probabilities",
                {},
            )

            if eligibility == "Eligible":

                st.success(
                    f"EMI Status: {eligibility}"
                )

            elif eligibility == "High_Risk":

                st.warning(
                    f"EMI Status: {eligibility}"
                )

            else:

                st.error(
                    f"EMI Status: {eligibility}"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Maximum Predicted Monthly EMI",
                    f"₹{max_emi:,.2f}",
                )

            with col2:

                st.metric(
                    "Requested Monthly EMI",
                    "Based on requested amount & tenure",
                )

            st.subheader(
                "Eligibility Probability"
            )

            for label, probability in probabilities.items():

                st.write(
                    f"**{label}** — {probability:.2f}%"
                )

                st.progress(
                    min(float(probability) / 100, 1.0)
                )

            st.subheader("Recommendation")

            st.info(recommendation)

        elif result:

            st.error(
                "The backend returned an unexpected response."
            )


# ============================================================
# CUSTOMER MANAGEMENT
# ============================================================

elif page == "Customer Management":

    st.title("Customer Management")

    st.info(
        "Customer CRUD integration will be implemented next."
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.title("Prediction History")

    st.info(
        "Prediction history will be implemented after "
        "the customer/database integration."
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

elif page == "System Status":

    st.title("System Status")

    st.subheader("EMIPredict-AI Components")

    col1, col2 = st.columns(2)

    with col1:

        st.success("FastAPI Backend")
        st.write("http://127.0.0.1:8001")

        st.success("XGBoost Classification")
        st.write("EMI eligibility model")

        st.success("XGBoost Regression")
        st.write("Maximum EMI prediction model")

    with col2:

        st.success("SQLite Database")
        st.write("Customer persistence")

        st.success("MLflow")
        st.write("Experiment tracking & model registry")

        st.success("Streamlit")
        st.write("Frontend application")