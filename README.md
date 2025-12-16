# 🤖 Job Market Insights: AI Automation Risk Predictor

## CAI3101: Introduction to Artificial Intelligence Project

### 📌 Project Overview
This project is an end-to-end Machine Learning application designed to predict the likelihood of human jobs being replaced by Artificial Intelligence. Using a Random Forest Classifier, the system analyzes various job characteristics—such as industry, salary, education level, and remote work trends—to classify jobs into five distinct risk levels.

The goal is to provide actionable insights for students and professionals to identify "safe" careers in the age of automation.

### 🎯 Objectives
Per the CAI3101 course requirements, this project demonstrates:
*   **End-to-end Workflow:** From data collection to deployment.
*   **Data Preprocessing:** Handling missing values, label encoding, and feature engineering.
*   **Model Development:** Training and evaluating a Supervised Learning model.
*   **Application Deployment:** Creating a user-friendly GUI using Streamlit.

### 📂 Dataset
*   **Source:** Public Dataset (Kaggle) - "AI Impact on Job Market Insights".
*   **Size:** >200 samples with 10+ features.
*   **Target Variable:** Automation Risk (%) (Converted into 5 categorical grades).
*   **Key Features Used:**
    *   Job Title & Industry
    *   Median Salary (USD)
    *   Required Education
    *   Work Experience
    *   Remote Work Ratio
    *   AI Impact Level
    *   Job Openings (Current & Projected)

### 🛠️ Technologies Used
*   **Language:** Python 3.9+
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (RandomForestClassifier, LabelEncoder)
*   **Visualization:** Matplotlib, Seaborn
*   **Graphical User Interface (GUI):** Streamlit

### ⚙️ Methodology

#### 1. Data Preprocessing
*   **Encoding:** Categorical variables (e.g., "Location", "Job Status") were converted into numerical values using LabelEncoder.
*   **Feature Engineering:** The continuous target Automation Risk (%) was binned into 5 distinct classes to create a Multi-Class Classification problem:
    *   0: Very Safe (0-20%)
    *   1: Safe (20-40%)
    *   2: Moderate (40-60%)
    *   3: High Risk (60-80%)
    *   4: Critical Risk (80-100%)

#### 2. Model Selection
We selected the **Random Forest Classifier** for this project because:
*   It handles high-dimensional data (many features) effectively.
*   It is less prone to overfitting compared to a single Decision Tree.
*   It provides high accuracy for classification tasks involving mixed data types (numerical + categorical).

#### 3. Application (GUI)
A web-based interface was built to allow users to:
*   Select existing job profiles from the dataset.
*   Adjust market variables (Salary, Remote Ratio, etc.) via sliders.
*   Receive real-time risk analysis and feedback.

### 🚀 How to Run the Project

#### Prerequisites:
Ensure you have Python installed. Install the required libraries using:

```bash
pip install pandas scikit-learn streamlit
```

#### Steps:
1.  Clone this repository or download the project folder.
2.  Ensure the dataset file named `data.csv` is in the same directory as `app.py`.
3.  Open your terminal/command prompt.
4.  Navigate to the project folder.
5.  Run the application:

```bash
streamlit run app.py
```

### 📊 Results
The model successfully identifies that jobs requiring High Social Intelligence (e.g., Healthcare, Management) and High Creativity are less likely to be automated, regardless of salary. Conversely, roles with high routine tasks and low educational requirements show the highest risk.

### 👥 Contributors
*   [Your Name]
*   [Partner Name 1]
*   [Partner Name 2]

*Submitted for CAI3101 - Term 5*
