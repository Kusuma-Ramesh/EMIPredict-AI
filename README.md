# EMIPredict-AI

## AI-Powered EMI Eligibility & Affordability Intelligence

EMIPredict-AI is an end-to-end machine learning decision-support application designed to evaluate a customer's EMI eligibility and estimate the maximum affordable monthly EMI from a structured financial profile.

Instead of producing only a simple eligible/not-eligible result, the system combines **classification and regression** to provide a more informative affordability assessment.

The application evaluates an applicant's financial profile, predicts an EMI eligibility category, estimates the maximum affordable EMI, compares it with the requested EMI, and generates a final affordability recommendation.

---

## 1. Project Overview

EMIPredict-AI combines machine learning, API services, database persistence, experiment tracking, and an interactive web application into a single end-to-end system.

The system provides:

- EMI eligibility classification
- Risk classification
- Maximum affordable EMI prediction
- Requested EMI vs. affordable EMI comparison
- Affordability recommendation
- Prediction history
- Customer and prediction persistence
- Backend health monitoring
- MLflow experiment tracking
- Interactive Streamlit dashboard
- Production deployment

The project is designed as a **decision-support system** for EMI affordability assessment.

---

## 2. Problem Statement

Traditional EMI eligibility decisions may provide only a simple eligible/not-eligible outcome without clearly indicating the applicant's actual affordability level.

A useful EMI decision-support system should consider multiple financial and personal attributes and provide:

1. EMI eligibility classification
2. Risk information
3. Maximum affordable monthly EMI
4. Requested EMI comparison
5. A clear affordability recommendation

EMIPredict-AI addresses this requirement through a dual-model machine learning architecture.

---

## 3. Project Objectives

The main objectives of EMIPredict-AI are:

- Develop an ML-based EMI eligibility prediction system.
- Classify applicants into **Eligible, High Risk, or Not Eligible** categories.
- Estimate the applicant's maximum affordable monthly EMI.
- Compare the requested EMI with the predicted affordable EMI.
- Generate an affordability recommendation.
- Build a REST API for prediction services.
- Persist customer and prediction records.
- Track experiments and models using MLflow.
- Provide an interactive application interface using Streamlit.
- Deploy and verify the integrated application.

---

## 4. System Architecture

The overall workflow is:

```text
Applicant Financial Profile
            |
            v
      Input Validation
            |
            v
     Data Preprocessing
            |
            +----------------------+
            |                      |
            v                      v
 XGBoost Classification    XGBoost Regression
            |                      |
            v                      v
 Eligibility Category      Maximum Affordable EMI
            |                      |
            +----------+-----------+
                       |
                       v
              Requested EMI Comparison
                       |
                       v
                Final Recommendation
                       |
                       v
                Database / API Response
                       |
                       v
                Streamlit Application
```

---

## 5. Machine Learning Approach

### 5.1 Classification

The classification model predicts one of three EMI eligibility outcomes:

- **Eligible**
- **High Risk**
- **Not Eligible**

The classifier also provides class probability information that can be used to understand the prediction.

### 5.2 Regression

The regression model estimates the **maximum affordable monthly EMI** for the applicant.

The predicted affordable EMI is then compared against the requested EMI to support the final recommendation.

---

## 6. Model Selection

The project evaluates machine learning algorithms for both classification and regression tasks.

### Classification Models

- Logistic Regression
- Random Forest
- XGBoost

### Regression Models

- Linear Regression
- Random Forest
- XGBoost

XGBoost was selected as the final model for both classification and regression based on the recorded evaluation results.

---

## 7. Dataset & Feature Information

The final machine-learning experiments recorded:

- **Training Rows:** 323,840
- **Model Features:** 33
- **Final ML Models:** 2 XGBoost models

The applicant information covers several areas.

### Personal Information

- Age
- Gender
- Marital Status
- Education

### Employment Information

- Monthly Salary
- Employment Type
- Years Employed
- Company Type

### Housing & Family Information

- House Type
- Monthly Rent
- Family Size
- Dependents

### Expense Information

- School Fees
- College Fees
- Travel Expenses
- Groceries and Utilities
- Other Expenses

### Financial Information

- Existing Loans
- Current EMI
- Credit Score
- Bank Balance
- Emergency Fund

### Loan Request Information

- EMI Scenario
- Requested Loan Amount
- Requested Tenure

The application converts the applicant inputs into the feature representation required by the trained machine-learning models.

---

## 8. Data Preprocessing

The preprocessing workflow prepares structured applicant information for machine learning.

The main preprocessing operations include:

- Input validation
- Handling missing numerical values
- Median imputation where applicable
- Encoding categorical variables
- Preparing numerical and categorical features
- Constructing the final model feature representation
- Maintaining a consistent feature structure during inference

The classification pipeline uses preprocessing components including:

- `ColumnTransformer`
- `SimpleImputer`
- `OneHotEncoder`
- XGBoost Classifier

Consistent preprocessing is maintained during model inference so that production inputs match the format used during model development.

---

## 9. Feature Engineering

The model uses financial, employment, housing, expense, credit, and loan-related attributes to represent the applicant's repayment capacity and affordability.

The processed inputs are transformed into the final **33-feature model representation** used by the trained machine-learning models.

Two target outputs are generated:

- EMI eligibility/risk category
- Maximum affordable monthly EMI

The predicted affordable EMI is subsequently compared with the requested EMI to generate an affordability recommendation.

---

## 10. Model Performance

### 10.1 Classification — XGBoost

| Metric | Result |
|---|---:|
| Accuracy | **92.03%** |
| ROC-AUC | **98.94%** |
| Recall | **92.18%** |
| Weighted F1 Score | **93.65%** |
| Macro F1 Score | **80.20%** |
| Precision | **77.80%** |

### 10.2 Regression — XGBoost

| Metric | Result |
|---|---:|
| R² Score | **0.9928** |
| RMSE | **658.73** |
| MAE | **277.46** |
| MAPE | **8.80%** |

The classification results demonstrate strong performance in distinguishing the three EMI eligibility outcomes.

The regression results demonstrate a high level of explained variance for the maximum affordable EMI prediction task.

---

## 11. Backend Architecture

The backend is implemented using **FastAPI**.

The backend is responsible for:

- Request validation
- Prediction orchestration
- Classification inference
- Regression inference
- Requested EMI calculation
- Affordability recommendation
- Database persistence
- Prediction history
- Health monitoring

### API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Performs classification and regression in one request |
| `GET /predictions` | Retrieves prediction history |
| `GET /predictions/{id}` | Retrieves a specific prediction |
| `GET /health` | Backend health monitoring |

---

## 12. Database & Persistence

The application uses:

- **SQLAlchemy** for database interaction
- **SQLite** for persistence

The system stores customer and prediction information so that prediction history can be retrieved through the API.

This allows the application to maintain persistent records of prediction activity rather than treating every prediction as an isolated request.

---

## 13. MLOps & Experiment Tracking

**MLflow** is integrated into the project for experiment and model tracking.

The project records machine-learning experiment information including:

- Training scale
- Number of features
- Model metrics
- Classification experiments
- Regression experiments
- Model selection results

MLflow provides a structured record of the experiments used to evaluate and select the final models.

---

## 14. Application Interface

The project includes a **Streamlit application** that provides an interactive interface for the EMIPredict-AI system.

The application allows users to:

1. Enter applicant information.
2. Submit the financial profile.
3. Request an EMI prediction.
4. View the eligibility result.
5. View the affordable EMI estimate.
6. Compare requested and affordable EMI.
7. View the resulting recommendation.
8. Access prediction-related information through the application dashboard.

The Streamlit application communicates with the deployed FastAPI backend.

---

## 15. Application Dashboard

The dashboard provides an overview of prediction activity and system status.

The interface includes:

- Total Customers
- Total Predictions
- Eligible predictions
- High Risk predictions
- Not Eligible predictions
- EMI Eligibility Distribution
- Requested vs. Maximum Recommended EMI
- Recent Prediction Activity

The dashboard is designed to provide a clear operational view of the deployed EMI prediction system.

---

## 16. Deployment

EMIPredict-AI was deployed as an integrated application with the presentation layer communicating with the backend service.

### Deployment Components

| Component | Role |
|---|---|
| Streamlit | Interactive application interface |
| FastAPI | Backend API service |
| XGBoost | Machine-learning models |
| SQLAlchemy | Database interaction |
| SQLite | Data persistence |
| MLflow | Experiment/model tracking |

The deployed application was verified successfully with the frontend showing an active **Backend connected** status.

---

## 17. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | XGBoost, scikit-learn |
| Classification | XGBoost Classifier |
| Regression | XGBoost Regressor |
| Backend | FastAPI |
| Validation | Pydantic |
| Database | SQLite |
| ORM | SQLAlchemy |
| MLOps | MLflow |
| Frontend / UI | Streamlit |
| Version Control | Git / GitHub |
| Deployment | Render / Streamlit deployment |

---

## 18. Project Workflow

```text
1. Applicant enters financial information
                    ↓
2. Input validation
                    ↓
3. Data preprocessing
                    ↓
4. Feature transformation
                    ↓
5. XGBoost classification
                    ↓
6. XGBoost regression
                    ↓
7. Requested EMI calculation
                    ↓
8. Affordable EMI comparison
                    ↓
9. Final recommendation
                    ↓
10. Prediction persistence
                    ↓
11. Result displayed in application
```

---

## 19. Key Features

### Machine Learning

- Dual-model classification and regression
- XGBoost-based prediction
- EMI eligibility classification
- Affordable EMI estimation
- Model evaluation and comparison

### Backend

- FastAPI REST API
- Pydantic validation
- Prediction endpoint
- Prediction history endpoints
- Health monitoring

### Persistence

- SQLite database
- SQLAlchemy ORM
- Customer records
- Prediction records

### MLOps

- MLflow experiment tracking
- Model evaluation
- Performance recording

### Application

- Streamlit interface
- Interactive EMI prediction
- Dashboard
- Prediction monitoring
- Backend connectivity verification

---

## 20. Project Highlights

- **323,840** recorded training rows
- **33** final model features
- **2** final XGBoost models
- **92.03%** classification accuracy
- **98.94%** classification ROC-AUC
- **93.65%** weighted F1 score
- **0.9928** regression R²
- **658.73** regression RMSE
- **277.46** regression MAE
- **8.80%** regression MAPE
- FastAPI backend
- Streamlit application
- SQLite persistence
- SQLAlchemy ORM
- MLflow experiment tracking
- Production deployment

---

## 21. Future Scope

Potential future improvements include:

- SHAP-based model explainability
- Expanded automated testing
- Improved model monitoring
- Data drift detection
- Production-grade database deployment
- Improved financial validation rules
- Additional affordability indicators
- Automated model retraining workflows
- More comprehensive production monitoring

---

## 22. Limitations

The current implementation is intended as a **machine-learning decision-support system** and should not be treated as an independent financial approval system.

Model predictions depend on:

- The quality of the training data
- The representativeness of the dataset
- The quality of applicant-provided information
- The assumptions used during feature engineering and model development

Further validation, monitoring, explainability, security hardening, and production governance would be appropriate before using the system in a regulated financial decision-making environment.

---

## 23. Project Structure

```text
EMIPredict-AI/
│
├── app/
├── data/
├── models/
├── reports/
├── src/
├── tests/
├── emipredict-frontend/
│
├── EMI-Predict-AI.ipynb
├── main.py
├── streamlit_app.py
├── requirements.txt
├── requirements-deploy.txt
├── .gitignore
└── README.md
```

---

## 24. Repository

GitHub Repository:

**https://github.com/Kusuma-Ramesh/EMIPredict-AI**

---

## 25. Project Status

**Completed and Deployed**

The EMIPredict-AI system has been implemented, evaluated, integrated across its machine-learning, backend, database, and application components, and successfully verified in the deployed environment.

---

## 26. Author

**Kusuma R**

Individual Machine Learning Project

### EMIPredict-AI

**AI-Powered EMI Eligibility & Affordability Intelligence**
