# 🤖 AI Job Market Risk Analyzer

## 📖 Overview
The **AI Job Market Risk Analyzer** is an intelligent web application designed to assess the risk of job automation due to Artificial Intelligence. By analyzing key market factors—such as industry, salary, remote work trends, and education requirements—the system predicts the likelihood of a specific job role being automated in the near future.

This project serves as a demonstration of how Machine Learning can be applied to real-world economic data to provide actionable insights for career planning.

---

## 🏗️ System Architecture & Concepts
The system follows a standard **Data Science Pipeline** integrated into a **Interactive Web Application**.

### 1. The Concept
The core logic relies on **Supervised Learning**. We have a dataset of historical or projected job data which includes a "Ground Truth" column (`Automation Risk %`). The model learns the patterns between various features (like Salary, Education, Industry) and this risk percentage.

### 2. The Flow
1.  **Data Ingestion:** The app loads raw job market data from a CSV file.
2.  **Preprocessing:**
    *   **Cleaning:** Handling missing values (if any).
    *   **Encoding:** Converting text data (e.g., "Software Engineer", "Healthcare") into numbers that the machine can understand using **Label Encoding**.
    *   **Target Generation:** Converting the continuous `Automation Risk (%)` into 5 distinct "Risk Grades" (from Very Safe to Critical).
3.  **Model Training:** A **Random Forest Classifier** is trained on-the-fly using the processed data.
4.  **User Interaction:** A Streamlit-based UI allows users to select job parameters.
5.  **Inference:** The trained model predicts the risk grade for the user's inputs.

---

## 🛠️ Technologies Used

*   **Python 3.x:** The core programming language.
*   **Streamlit:** A framework for building the web interface quickly and effectively.
*   **Pandas:** Used for data manipulation and reading the CSV file.
*   **Scikit-Learn (sklearn):** The machine learning library used for:
    *   `RandomForestClassifier`: The predictive model.
    *   `LabelEncoder`: For converting text categories to numbers.
*   **Jupyter Notebook:** Used for the initial Exploratory Data Analysis (EDA) and model selection.

---

## 📂 File Structure

*   **`app.py`**: The main application file. It contains the full logic for the web app, including data loading, model training, and the UI code.
*   **`JobRiskInsights.ipynb`**: A research notebook. It documents the initial steps of the project:
    *   Visualizing the data (histograms, statistics).
    *   Comparing different models (Decision Tree vs. Naive Bayes vs. Random Forest).
    *   Prototyping the logic.
*   **`data.csv`**: The dataset containing job market information (Job Titles, Salaries, Risk percentages, etc.).

---

## 🔍 Code Explanation: `app.py`
Here is a breakdown of the logic inside the main application file:

### 1. Configuration & Imports
We import `streamlit` for the UI, `pandas` for data, and `sklearn` components.
`st.set_page_config` is used to set the browser tab title and layout.

### 2. `get_model()` Function (The Core Logic)
This function is decorated with `@st.cache_data`. This is a crucial optimization: it ensures that data loading and model training happen **only once** when the app starts, rather than every time a user clicks a button.

*   **Data Loading:** Reads `data.csv`. *Note: It handles file not found errors gracefully.*
*   **Preprocessing Loop:** It iterates through categorical columns (like 'Industry', 'Job Status'). It uses `LabelEncoder` to assign a unique number to each category (e.g., "Tech" -> 1, "Finance" -> 2) and saves these encoders to use later for user inputs.
*   **Target Creation:** It converts the `Automation Risk (%)` number into a classification grade (0-4):
    *   **0:** Very Safe (0-20%)
    *   **1:** Safe (20-40%)
    *   **2:** Moderate (40-60%)
    *   **3:** High Risk (60-80%)
    *   **4:** Critical Risk (80-100%)
*   **Training:** It fits a `RandomForestClassifier` on the features to predict this `Risk_Grade`.

### 3. Sidebar Inputs
The app dynamically generates dropdown menus based on the data. For example, the "Industry" dropdown will only show industries present in the dataset. This is done using the `encoders` dictionary created during training.

### 4. Prediction Logic
When the "Analyze" button is pressed:
1.  The app takes the user's inputs (text).
2.  It converts them into numbers using the stored `encoders` (e.g., converts user selection "Remote" to the number the model expects).
3.  The model predicts the Risk Grade (0-4).
4.  The result is displayed with a corresponding message and color (Green for Safe, Red for Risk).

---

## 🚀 Setup Instructions

### Prerequisites
*   Python 3.8 or higher installed.

### 1. Install Dependencies
Open your terminal or command prompt and run:
```bash
pip install streamlit pandas scikit-learn
```

### 2. Configure Data Path
**Important:** The application currently looks for the dataset at a specific location on your machine. You need to update this.

1.  Open `app.py` in a code editor.
2.  Find the line:
    ```python
    FILE_PATH = r"E:\College\Term 5\AI\Dataset\AI_Project\data.csv"
    ```
3.  Change this path to the location of `data.csv` on **your** computer. For example, if it's in the same folder as the code:
    ```python
    FILE_PATH = "data.csv"
    ```

### 3. Run the Application
Navigate to the project folder in your terminal and run:
```bash
streamlit run app.py
```
A browser window should open automatically with the app.

---

## 🔮 Future Improvements
*   **Dynamic Path Handling:** Update the code to automatically detect `data.csv` in the current directory to remove the need for manual path configuration.
*   **Model Persistence:** Save the trained model to a file (`.pkl`) so the app doesn't need to retrain every time it restarts.
*   **Advanced Visualizations:** Add charts to the result page showing how the user's salary or industry compares to others.
*   **More Models:** Allow the user to switch between different algorithms (e.g., SVM, Gradient Boosting) to see if predictions change.
