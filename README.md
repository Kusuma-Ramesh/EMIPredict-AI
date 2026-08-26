# EMIPredict-AI
## AI-Powered EMI Eligibility & Affordability Intelligence

EMIPredict-AI is an end-to-end machine learning decision-support application designed to evaluate a customer's EMI eligibility and estimate the maximum affordable monthly EMI from a structured financial profile.

Instead of producing only a simple eligible/not-eligible result, the system combines **classification and regression** to provide a more informative affordability assessment. The application evaluates the applicant's financial profile, predicts an EMI eligibility category, estimates the maximum affordable EMI, compares it with the requested EMI, and generates a final recommendation.

The project integrates machine learning models with a production-oriented backend, database persistence, experiment tracking, and an interactive Streamlit application.

---

## Project Overview

EMIPredict-AI addresses the problem of making EMI decisions using only a binary eligibility outcome.

The system answers two complementary questions:

1. **Is the requested EMI acceptable for the applicant?**
2. **What is the maximum monthly EMI the applicant can reasonably afford?**

The solution therefore combines:

- EMI eligibility classification
- Class probability information
- Maximum affordable EMI regression
- Requested EMI calculation
- Requested EMI vs. affordable EMI comparison
- Final affordability recommendation
- Prediction history and persistence
- Backend health monitoring
- ML experiment and model tracking

---

## Problem Statement

Traditional EMI eligibility decisions may provide only a simple eligible/not-eligible outcome without clearly indicating the applicant's actual affordability level.

A useful EMI decision-support system should consider a structured financial profile and provide:

- EMI eligibility classification
- Probability information for the predicted class
- Maximum affordable monthly EMI
- Comparison between requested and affordable EMI
- A clear recommendation

EMIPredict-AI addresses this requirement through a dual-model machine learning architecture.

---

## Objectives

The main objectives of EMIPredict-AI are:

- Develop an ML-based EMI eligibility prediction system.
- Classify applicants into **Eligible, High Risk, or Not Eligible** categories.
- Estimate the applicant's maximum affordable monthly EMI.
- Compare the requested EMI with the predicted affordable EMI.
- Provide an interpretable affordability recommendation.
- Build a REST API for prediction services.
- Persist customer and prediction records.
- Track experiments and models using MLflow.
- Provide an interactive application interface using Streamlit.
- Deploy and verify the integrated application.

---

## System Architecture

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

The system uses a dual-model architecture so that eligibility classification and affordability estimation can be performed together.

Machine Learning Approach
1. Classification

The classification model predicts one of three EMI eligibility outcomes:

Eligible
High Risk
Not Eligible

The classifier also provides class probability information that can be used to understand the prediction.

2. Regression

The regression model estimates the maximum affordable monthly EMI for the applicant.

The predicted affordable EMI is then compared against the requested EMI to support the final recommendation.

Model Selection

Multiple machine learning algorithms were considered during model evaluation.

Classification

The following models were compared:

Logistic Regression
Random Forest
XGBoost
Regression

The following models were compared:

Linear Regression
Random Forest
XGBoost

XGBoost was selected as the final model for both tasks based on the recorded evaluation results.

Dataset & Feature Information

The final tracked machine-learning experiments recorded:

Training Rows: 323,840
Model Features: 33
Final ML Models: 2 XGBoost models

The applicant information covers areas including:

Personal Information
Age
Gender
Marital Status
Education
Employment Information
Monthly Salary
Employment Type
Years Employed
Company Type
Housing & Family Information
House Type
Monthly Rent
Family Size
Dependents
Expense Information
School Fees
College Fees
Travel Expenses
Groceries and Utilities
Other Expenses
Financial Information
Existing Loans
Current EMI
Credit Score
Bank Balance
Emergency Fund
Loan Request Information
EMI Scenario
Requested Loan Amount
Requested Tenure

The user-facing application collects applicant information which is transformed into the final model feature representation.

Data Preprocessing

The preprocessing workflow prepares structured applicant information for machine learning.

The main preprocessing operations include:

Validation of applicant inputs
Handling missing numerical values
Median imputation where applicable
Encoding categorical variables
Preparing numerical and categorical features
Constructing the final model feature representation
Maintaining a consistent feature structure during inference

The classification pipeline uses preprocessing components including:

ColumnTransformer
SimpleImputer
OneHotEncoder
XGBoost Classifier

The same principle of consistent preprocessing is maintained during model inference.

Feature Engineering

The model uses financial, employment, housing, expense, credit, and loan-related attributes to represent the applicant's repayment capacity and affordability.

The processed inputs are transformed into the final 33-feature model representation used by the trained machine-learning models.

Two target outputs are generated:

EMI eligibility/risk category
Maximum affordable monthly EMI
Model Performance
Classification — XGBoost
Metric	Result
Accuracy	92.03%
ROC-AUC	98.94%
Recall	92.18%
Weighted F1 Score	93.65%
Macro F1 Score	80.20%
Precision	77.80%
Regression — XGBoost
Metric	Result
R² Score	0.9928
RMSE	658.73
MAE	277.46
MAPE	8.80%

The classification results demonstrate strong performance in distinguishing the three EMI eligibility outcomes.

The regression results demonstrate a high level of explained variance for the maximum affordable EMI prediction task.

Backend Architecture

The backend is implemented using FastAPI.

The backend is responsible for:

Request validation
Prediction orchestration
Classification inference
Regression inference
Requested EMI calculation
Recommendation generation
Database persistence
Prediction history
Health monitoring
API Endpoints
Endpoint	Purpose
POST /predict	Performs classification and regression in one request
GET /predictions	Retrieves prediction history
GET /predictions/{id}	Retrieves a specific prediction
GET /health	Backend health monitoring
Database & Persistence

The application uses:

SQLAlchemy for database interaction
SQLite for persistence

The system stores customer and prediction information so that prediction history can be retrieved through the API.

This allows the application to maintain a persistent record of prediction activity rather than treating each prediction as an isolated request.

MLOps & Experiment Tracking

MLflow is integrated into the project for experiment and model tracking.

It is used to record information associated with the machine-learning experiments, including:

Training scale
Model features
Model metrics
Classification experiments
Regression experiments
Model selection results

The recorded MLflow experiments provide the basis for the final model-performance results reported in this project.

Application Interface

The project includes a Streamlit presentation layer that provides an interactive interface for the EMIPredict-AI system.

The application allows users to:

Enter applicant information.
Submit the financial profile.
Request an EMI prediction.
View the eligibility result.
View the affordable EMI estimate.
Compare requested and affordable EMI.
View the resulting recommendation.

The interface communicates with the backend prediction service.

Deployment

EMIPredict-AI was deployed as an integrated application with the presentation layer communicating with the backend service.

The deployed application was successfully verified with the frontend displaying a live Backend connected status.

Deployment Components
Streamlit — presentation layer
FastAPI — backend service
XGBoost — machine-learning models
SQLAlchemy + SQLite — persistence
MLflow — experiment/model tracking

Technology Stack
Component	Technology
Programming Language	Python
Machine Learning	XGBoost, scikit-learn
Classification	XGBoost Classifier
Regression	XGBoost Regressor
Backend	FastAPI
Validation	Pydantic
Database	SQLite
ORM	SQLAlchemy
MLOps	MLflow
Frontend / UI	Streamlit
Version Control	Git / GitHub
Deployment	Render / Streamlit deployment
Project Workflow
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
Key Highlights
End-to-end machine learning implementation
Dual-model classification and regression architecture
323,840 recorded training rows
33 final model features
XGBoost selected for both prediction tasks
92.03% classification accuracy
98.94% classification ROC-AUC
0.9928 regression R²
FastAPI REST backend
Streamlit application interface
SQLite persistence
MLflow experiment tracking
Prediction history
Backend health monitoring
Production deployment verification
Future Scope

Potential future improvements include:

SHAP-based model explainability
Expanded automated testing
Improved model monitoring
Data drift detection
Stronger authentication and authorization
Production-grade database deployment
Improved financial validation rules
Additional affordability and repayment indicators
More comprehensive model retraining workflows
Limitations

The current implementation is intended as a machine-learning decision-support system and should not be treated as an independent financial approval system.

Model predictions depend on the quality and representativeness of the training data and the information supplied by the applicant.

Further validation, monitoring, explainability, and production hardening would be appropriate before using the system in a regulated financial decision-making environment.

Project Status

Completed and deployed.

The integrated EMIPredict-AI application has been implemented, evaluated, connected across its frontend and backend components, and verified in the deployed environment.

Author

Kusuma R

Individual Machine Learning Project
