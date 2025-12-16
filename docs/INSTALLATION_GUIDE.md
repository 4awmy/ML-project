# 📖 Beginner's Installation & Run Guide

This guide explains how to set up the environment and run the project from scratch on your local computer.

## 🛠️ Prerequisites
Before you start, ensure you have:
1.  **Python Installed:**
    *   Check if you have it by opening a terminal/command prompt and typing: `python --version` or `python3 --version`.
    *   If not, download it from [python.org](https://www.python.org/downloads/).
2.  **A Code Editor (Optional but recommended):**
    *   VS Code, PyCharm, or even Notepad++.

---

## 🚀 Step-by-Step Installation

### 1. Download the Project
1.  Download the project folder (or unzip it).
2.  Open your **Terminal** (Mac/Linux) or **Command Prompt** (Windows).
3.  Navigate to the project folder:
    ```bash
    cd path/to/your/folder
    # Example: cd Downloads/AI_Project
    ```

### 2. Create a Virtual Environment (Recommended)
A "virtual environment" keeps your project libraries separate from your computer's main system.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
*(You should see `(venv)` appear at the start of your terminal line).*

### 3. Install Dependencies
We need to install libraries like `pandas`, `sklearn`, and `streamlit`.
```bash
pip install -r requirements.txt
```
*Wait for the installation to finish.*

---

## ▶️ How to Run the App

1.  In the same terminal (make sure you are in the project folder), run:
    ```bash
    streamlit run app.py
    ```
2.  A browser window will automatically open (usually at `http://localhost:8501`).
3.  **That's it!** You can now interact with the project.

---

## 🧪 How to Run Tests (Verification)

If you want to check if the code is working correctly internally:
1.  Install the test library:
    ```bash
    pip install pytest
    ```
2.  Run the tests:
    ```bash
    python -m pytest tests/
    ```
    *You should see green text saying "Passed".*
