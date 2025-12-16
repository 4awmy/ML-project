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

1.  **[Project Requirements Explained](docs/REQUIREMENTS_EXPLAINED.md)** - Read this for the **Written Report** and **PowerPoint**.
2.  **[Installation Guide](docs/INSTALLATION_GUIDE.md)** - Step-by-step instructions to run the project locally.
3.  **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - How to get a public URL using Streamlit Community Cloud (GitHub).
4.  **[Gap Analysis](docs/GAP_ANALYSIS.md)** - A checklist of how we met all course requirements.

### 🛠️ Libraries & Imports Explained

The project uses several powerful Python libraries. Here is a breakdown of why each one is used:

*   **`streamlit`**: The core framework used to build the web application. It allows us to create interactive UIs (buttons, sliders, charts) using simple Python code, without needing HTML/CSS/JS.
*   **`pandas`**: The standard library for data manipulation. We use it to load the CSV file (`pd.read_csv`), clean data (`dropna`), and handle the table-like structure of our dataset.
*   **`seaborn` & `matplotlib.pyplot`**: Visualization libraries used to generate static charts, specifically the **Correlation Heatmap**. `seaborn` makes the heatmap look good, while `matplotlib` provides the underlying figure.
*   **`plotly.express`**: An interactive plotting library. We use it for the **Risk Distribution Histogram** because it allows users to hover over bars to see exact numbers.
*   **`sklearn.model_selection.train_test_split`**: A utility to split our data into two parts: Training (80%) and Testing (20%). This ensures we evaluate our models on data they haven't seen before.
*   **`sklearn.metrics.accuracy_score`**: The metric used to grade our models. It simply calculates what percentage of predictions were correct.
*   **`sklearn.ensemble.RandomForestClassifier`**: A powerful model that uses many decision trees to vote on the outcome. Great for high accuracy.
*   **`sklearn.naive_bayes.GaussianNB`**: A probabilistic model based on Bayes' Theorem. Fast and simple, often used as a baseline.
*   **`sklearn.tree.DecisionTreeClassifier`**: A model that learns simple if-then rules. Easy to visualize and explain.
*   **`sklearn.neural_network.MLPClassifier`**: A Multi-Layer Perceptron (Artificial Neural Network). It mimics the human brain's structure to learn complex, non-linear patterns in the data.
*   **`sklearn.preprocessing.LabelEncoder`**: Converts text categories (e.g., "Engineer", "Doctor") into numbers (0, 1) so the models can process them.
*   **`sklearn.preprocessing.StandardScaler`**: Normalizes numerical data (like Salary) so that all features have a mean of 0. This is crucial for the Neural Network to learn effectively.

### 📂 File Structure & Purpose

Here is a detailed explanation of every file in the repository:

*   **`app.py`**: **The Frontend.** This is the main entry point of the application. It handles:
    *   Displaying the Streamlit UI (tabs, sidebar, charts).
    *   Taking user input.
    *   Calling `model_utils` to get predictions.
    *   Displaying results to the user.
*   **`model_utils.py`**: **The Backend Logic.** This file contains the "business logic" to keep `app.py` clean. It handles:
    *   Loading and cleaning the data from CSV.
    *   Training the machine learning models.
    *   Encoding and scaling features.
    *   It uses `@st.cache_data` to ensure we don't re-train models every time the user clicks a button.
*   **`data.csv`**: **The Dataset.** Contains the rows of job data (Salary, Industry, Risk %, etc.) used to train our AI.
*   **`requirements.txt`**: **Dependency List.** Tells the server (or your computer) which Python libraries to install to run the app.
*   **`tests/`**: **Quality Assurance.** A folder containing unit tests (e.g., `test_app.py`) to verify that the data loads correctly and models train without errors.
*   **`docs/`**: **Documentation.** Contains detailed guides for Installation, Deployment, and Course Requirements.
*   **`.gitignore`**: **Configuration.** Tells Git which files to ignore (like temporary `__pycache__` files) to keep the repository clean.

### 🔍 Codebase Deep Dive (How it Works)

This section explains the internal logic of the application for developers and instructors. The code is split into two main files to follow software engineering best practices (Separation of Concerns).

#### 1. The Brain: `model_utils.py`
This file contains all the "heavy lifting" logic for Machine Learning. It is separated from the UI to ensure the code is clean, reusable, and testable.

*   **`load_and_process_data(file_path)`**
    *   **Missing Values:** Automatically removes incomplete rows using `df.dropna()`.
    *   **Target Engineering:** The raw dataset has a percentage risk (0-100%). We convert this into 5 distinct classes (0=Very Safe, 4=Critical) to make it a Classification problem.
    *   **Label Encoding:** Converts text columns (e.g., "Industry: Tech") into numbers (e.g., 0, 1, 2) so the algorithms can understand them.
    *   **Feature Scaling:** Uses `StandardScaler` to normalize numerical data (like Salary). This is **critical** for the Neural Network and Naive Bayes to work correctly.
    *   **Caching:** We use `@st.cache_data` so this expensive process happens only once when the app loads, not every time you click a button.

*   **`train_models(X_train, y_train)`**
    *   Initializes four distinct models:
        *   `RandomForestClassifier`: Robust, good for general purpose.
        *   `GaussianNB`: Fast, probabilistic baseline.
        *   `DecisionTreeClassifier`: Simple, easy to interpret rules.
        *   `MLPClassifier`: A Neural Network with 2 hidden layers (100 and 50 neurons) to capture complex non-linear patterns.
    *   Trains all models on the provided training data.
    *   Returns a dictionary of trained models ready for prediction.

#### 2. The Face: `app.py`
This is the Streamlit application that users interact with.

*   **Page Setup:** Configures the page title, icon, and wide layout.
*   **Tab Structure:**
    *   **Tab 1 (Data):** Displays raw data, summary statistics, and visualizations (Histograms, Heatmaps) using `plotly` and `seaborn`.
    *   **Tab 2 (Model):** Splits the data (80% Train / 20% Test), trains the models, and displays a bar chart comparing their Accuracy scores.
    *   **Tab 3 (Prediction):**
        *   Collects user inputs via Sidebar/Columns (Sliders, Dropdowns).
        *   **Preprocessing consistency:** It ensures user input is encoded and scaled exactly the same way as the training data using the saved `encoders` and `scaler` from `model_utils.py`.
        *   **Inference:** Feeds the processed input into the selected model to get a prediction (0-4).
        *   **Interpretation:** Maps the predicted number (e.g., "3") to a human-readable string (e.g., "High Risk") and displays it with appropriate colors (Red/Green).

### 🚀 Quick Start (Local)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
2.  **Running the app online :**
    ```bash
    [https://xoa-ml.streamlit.app/]
    ```
### ⚙️ Technical Highlights
*   **Refactored Codebase:** Logic is separated into `model_utils.py` for cleaner code and better caching.
*   **Unit Tests:** comprehensive tests in `tests/` ensure data integrity and model stability.
*   **Advanced ML:** Implements Multi-Layer Perceptron (MLP) for non-linear pattern recognition.

### 👥 Contributors
Omar Hossam Eldin  Metwally GAd  
Mohamed Ahmed Ezzat Mohamed Elsayed
Belal Ashraf Sobhy Mohamed Hassan

*Submitted for CAI3101 - Term 5*
