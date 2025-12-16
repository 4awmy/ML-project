# 🛑 Gap Analysis & Missing Requirements

This document outlines the discrepancies between the current project state, the CAI3101 course requirements, and your specific user requests.

## 1. Course Requirements vs. Current State

| Requirement | Current Status | Gap / Action Needed |
| :--- | :--- | :--- |
| **Dataset:** At least 200 samples & 5 features. | ✅ **Pass** (30,000 samples, 13 features). | None. |
| **Data Understanding:** Describe source, stats, viz, problem type. | ⚠️ **Partial**. | The code has the *logic* (Tab 1), but the *Written Report* (PDF) is missing. The app displays stats/viz correctly. |
| **Data Preprocessing:** Handle missing values, outliers. | ⚠️ **Basic**. | Code uses `dropna()`. We should check if outliers exist or if `dropna` removes too much data (though with 30k rows it's likely fine). |
| **Data Preprocessing:** Encode & Scale. | ✅ **Pass**. | `LabelEncoder` and `StandardScaler` are implemented. |
| **Model Development:** Choose **at least 3 models** (e.g., Naive Bayes, DT, **ANN**). | ⚠️ **Partial**. | We have Random Forest, Naive Bayes, Decision Tree. **Missing:** The prompt suggests "Artificial Neural Network" (ANN). We should replace one or add it as a 4th. |
| **Model Evaluation:** Accuracy & Compare. | ✅ **Pass**. | Tab 2 compares models by accuracy. |
| **Submission:** Report & PowerPoint. | ❌ **Missing**. | You need to write a Report (Word/PDF) and create a PPT. I cannot generate these binary files, but I can provide the text content for them. |

## 2. User Requests vs. Current State

| User Request | Current Status | Gap / Action Needed |
| :--- | :--- | :--- |
| **"Run through GitHub as a domain"** | ❌ **Not Configured**. | You need a `DEPLOYMENT_GUIDE.md` (created) and we need to ensure the repo is ready for Streamlit Cloud. |
| **"New models would be nice"** | ❌ **Missing**. | We should implement `MLPClassifier` (Neural Network) to satisfy both the user request and the course suggestion. |
| **"More unit tests"** | ❌ **Missing**. | No `tests/` directory exists. We need to add `pytest` tests. |
| **"Detailed README"** | ❌ **Missing**. | The current README is generic. We will update it. |

## 3. Missing Files Summary
*   `tests/test_app.py` (Unit Tests)
*   `model_utils.py` (Refactoring - strictly not missing, but highly recommended for code quality "A+" grade)
*   `requirements.txt` (Needs `pytest` added)
