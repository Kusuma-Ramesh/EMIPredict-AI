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
