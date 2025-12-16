# 🤖 AI Job Market Risk Analyzer

## CAI3101: Introduction to Artificial Intelligence Project

### 📌 Project Overview
This project is an end-to-end Machine Learning application designed to predict the likelihood of human jobs being replaced by Artificial Intelligence. It fulfills the CAI3101 course requirements by demonstrating the complete ML lifecycle: data collection, preprocessing, model development (including Neural Networks), and deployment.

### 🌟 Features
*   **Data Analysis:** Interactive dashboard to explore job market trends, correlations, and risk distributions.
*   **Predictive Modeling:** Compares three powerful algorithms:
    *   **Random Forest Classifier** (Ensemble Learning)
    *   **Naive Bayes** (Probabilistic Learning)
    *   **Neural Network (MLP)** (Deep Learning) - *New!*
*   **Interactive Risk Analyzer:** Users can input job details (Salary, Remote Work, Education, etc.) to get a real-time risk assessment.
*   **Actionable Insights:** Provides clear feedback on whether a job is "Safe", "Moderate", or "Critical Risk".

### 🚀 Deployment Guide (How to get a Public URL)

**Note:** Since this is a Python/Streamlit application, it requires a backend server to run the machine learning models. Therefore, it cannot be hosted on GitHub Pages (which only supports static HTML/CSS).

**The Best Approach: Streamlit Community Cloud**
This is the easiest and free way to get a public URL (e.g., `https://your-project.streamlit.app`) that you can share with your professor and friends.

#### Steps to Deploy:
1.  **Push Code to GitHub:**
    *   Create a new repository on GitHub.
    *   Upload all files from this folder (`app.py`, `model_utils.py`, `data.csv`, `requirements.txt`) to the repository.
2.  **Sign up for Streamlit Cloud:**
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Sign in with your GitHub account.
3.  **Deploy:**
    *   Click **"New App"**.
    *   Select your GitHub repository, branch (usually `main`), and the main file path (`app.py`).
    *   Click **"Deploy"**.
4.  **Done!**
    *   Streamlit will install the dependencies from `requirements.txt` and launch your app.
    *   You will get a public URL to share.

### 🛠️ Local Installation & Running

If you want to run the project on your own computer:

1.  **Prerequisites:**
    *   Install Python (3.8 or higher).
    *   Download this project folder.

2.  **Install Dependencies:**
    Open your terminal/command prompt in the project folder and run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
    A browser window will open automatically at `http://localhost:8501`.

4.  **Run Tests:**
    To verify that everything is working correctly:
    ```bash
    python -m pytest tests/
    ```

### 📂 Project Structure

*   `app.py`: The main entry point for the Streamlit application (Frontend & Logic).
*   `model_utils.py`: Contains functions for data loading, preprocessing, and model training.
*   `data.csv`: The dataset used for training and analysis.
*   `requirements.txt`: List of Python libraries required to run the app.
*   `tests/`: Contains unit tests to ensure code quality.

### ⚙️ Technical Details (For Graders/Developers)

#### 1. Data Preprocessing
*   **Cleaning:** Missing values are removed to ensure data quality.
*   **Target Engineering:** The continuous 'Automation Risk (%)' is binned into 5 categorical grades (0-4) for classification.
*   **Encoding:** Categorical features (e.g., Industry, Education) are LabelEncoded.
*   **Scaling:** Numerical features (e.g., Salary, Experience) are StandardScaled for optimal model performance (especially for Neural Networks).

#### 2. Models Used
*   **Random Forest:** Selected for its robustness against overfitting and ability to handle mixed data types.
*   **Naive Bayes:** Used as a baseline probabilistic model.
*   **Neural Network (MLPClassifier):** A Multi-Layer Perceptron (100, 50 hidden units) is used to capture complex non-linear relationships, fulfilling the requirement for advanced algorithms.

#### 3. Evaluation
*   Models are evaluated using **Accuracy** on a held-out test set (20% split).
*   The application displays a comparison chart to highlight the best-performing model.

### 👥 Contributors
*   [Your Name]
*   [Partner Name]
*   [Partner Name]

*Submitted for CAI3101 - Term 5*
