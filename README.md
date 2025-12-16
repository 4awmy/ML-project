# 🤖 AI Job Market Risk Analyzer

## CAI3101: Introduction to Artificial Intelligence Project

### 📌 Project Overview
This project is an end-to-end Machine Learning application designed to predict the likelihood of human jobs being replaced by Artificial Intelligence. It fulfills the CAI3101 course requirements by demonstrating the complete ML lifecycle: data collection, preprocessing, model development (including Neural Networks), and deployment.

### 🌟 Features
*   **Data Analysis:** Interactive dashboard to explore job market trends, correlations, and risk distributions.
*   **Predictive Modeling:** Compares four powerful algorithms:
    *   **Random Forest Classifier** (Ensemble Learning)
    *   **Naive Bayes** (Probabilistic Learning)
    *   **Decision Tree** (Logic-based Learning)
    *   **Neural Network (MLP)** (Deep Learning) - *Advanced Requirement!*
*   **Interactive Risk Analyzer:** Users can input job details (Salary, Remote Work, Education, etc.) to get a real-time risk assessment.
*   **Actionable Insights:** Provides clear feedback on whether a job is "Safe", "Moderate", or "Critical Risk".

### 📚 Documentation
We have prepared detailed documentation for every aspect of this project:

1.  **[Project Requirements Explained](REQUIREMENTS_EXPLAINED.md)** - Read this for the **Written Report** and **PowerPoint**.
2.  **[Installation Guide](INSTALLATION_GUIDE.md)** - Step-by-step instructions to run the project locally.
3.  **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - How to get a public URL using Streamlit Community Cloud (GitHub).
4.  **[Gap Analysis](GAP_ANALYSIS.md)** - A checklist of how we met all course requirements.

### 🚀 Quick Start (Local)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

### ⚙️ Technical Highlights
*   **Refactored Codebase:** Logic is separated into `model_utils.py` for cleaner code and better caching.
*   **Unit Tests:** comprehensive tests in `tests/` ensure data integrity and model stability.
*   **Advanced ML:** Implements Multi-Layer Perceptron (MLP) for non-linear pattern recognition.

### 👥 Contributors
*   [Your Name]
*   [Partner Name]
*   [Partner Name]

*Submitted for CAI3101 - Term 5*
