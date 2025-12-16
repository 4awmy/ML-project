import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 🚨 PATH CONFIGURATION
# ==========================================
FILE_PATH = "data.csv"

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Advanced Job Risk AI", layout="wide")
st.title("🤖 AI Job Market Risk Analyzer")
st.markdown("### CAI3101 Project: End-to-End Machine Learning Workflow")

# --- 2. LOAD DATA & PREPROCESSING ---
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        st.error(f"Error: File not found at {FILE_PATH}")
        return None, None, None, None, None

    # 1. Handle Missing Values (Requirement: Data Preprocessing)
    df = df.dropna()

    # 2. Target Engineering
    if 'Automation Risk (%)' in df.columns:
        def get_risk_grade(risk):
            if risk <= 20: return 0  # Very Safe
            elif risk <= 40: return 1 # Safe
            elif risk <= 60: return 2 # Moderate
            elif risk <= 80: return 3 # High Risk
            else: return 4            # Critical Risk
        df['Risk_Grade'] = df['Automation Risk (%)'].apply(get_risk_grade)
    else:
        st.error("Target column 'Automation Risk (%)' not found.")
        return None, None, None, None, None

    # 3. Encoding & Scaling (Requirement: Encode & Scale)
    encoders = {}
    cat_cols = ['Job Title', 'Industry', 'Job Status', 'AI Impact Level', 'Required Education', 'Location']

    # Store original DF for display, work on df_processed
    df_processed = df.copy()

    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            encoders[col] = le

    # Define Features
    feature_cols = [
        'Job Title', 'Industry', 'Job Status', 'AI Impact Level',
        'Median Salary (USD)', 'Required Education', 'Experience Required (Years)',
        'Job Openings (2024)', 'Projected Openings (2030)', 'Remote Work Ratio (%)', 'Location'
    ]
    final_features = [c for c in feature_cols if c in df_processed.columns]

    # Scaling Numerical Features (Crucial for Naive Bayes)
    scaler = StandardScaler()
    num_cols = ['Median Salary (USD)', 'Experience Required (Years)', 'Job Openings (2024)',
                'Projected Openings (2030)', 'Remote Work Ratio (%)']

    # Only scale columns that exist in final_features
    existing_num_cols = [c for c in num_cols if c in final_features]
    if existing_num_cols:
        df_processed[existing_num_cols] = scaler.fit_transform(df_processed[existing_num_cols])

    return df, df_processed, final_features, encoders, scaler

# Load Data
df_original, df_processed, feature_names, encoders, scaler = load_and_process_data()

if df_original is None:
    st.stop()

# --- TABS FOR PROJECT REQUIREMENTS ---
tab1, tab2, tab3 = st.tabs(["📊 Data Understanding", "⚙️ Model Development & Eval", "🚀 Risk Prediction"])

# ==========================================
# TAB 1: DATA UNDERSTANDING (Requirement: Statistics & Viz)
# ==========================================
with tab1:
    st.header("1. Data Understanding & Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Context")
        st.write(f"**Total Samples:** {df_original.shape[0]}")
        st.write(f"**Total Features:** {df_original.shape[1]}")
        st.dataframe(df_original.head())

    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df_original.describe())

    st.subheader("Data Visualizations")
    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        st.write("**Risk Distribution (Target Variable)**")
        fig = px.histogram(df_original, x='Automation Risk (%)', nbins=20, title="Distribution of Automation Risk")
        st.plotly_chart(fig)

    with col_viz2:
        st.write("**Correlation Heatmap**")
        # Compute correlation on processed numeric data
        corr = df_processed[feature_names + ['Risk_Grade']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ==========================================
# TAB 2: MODEL DEVELOPMENT (Requirement: 3 Models & Split)
# ==========================================
with tab2:
    st.header("2. Model Training & Comparison")
    st.write("Splitting dataset into Training (80%) and Testing (20%) sets.")

    # Prepare X and y
    X = df_processed[feature_names]
    y = df_processed['Risk_Grade']

    # Train/Test Split (Requirement)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define 3 Models (Requirement: At least 3 models)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    # Train and Evaluate
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Accuracy': acc})
        trained_models[name] = model

    # Display Results
    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

    st.subheader("Model Performance")
    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.dataframe(results_df)
        best_model_name = results_df.iloc[0]['Model']
        st.success(f"🏆 Best Model: **{best_model_name}**")

    with col_res2:
        st.bar_chart(results_df.set_index('Model'))

# ==========================================
# TAB 3: PREDICTION (Requirement: Actionable Insights)
# ==========================================
with tab3:
    st.header("3. Interactive Risk Analyzer")

    # Sidebar inputs moved here for clarity
    st.sidebar.header("Job Parameters")

    def get_options(col_name):
        if col_name in encoders:
            return sorted(encoders[col_name].classes_)
        return ["Unknown"]

    # Inputs
    selected_job = st.sidebar.selectbox("Job Title", get_options('Job Title'))
    selected_industry = st.sidebar.selectbox("Industry", get_options('Industry'))
    selected_status = st.sidebar.selectbox("Job Status", get_options('Job Status'))
    selected_edu = st.sidebar.selectbox("Required Education", get_options('Required Education'))
    selected_loc = st.sidebar.selectbox("Location", get_options('Location'))

    salary = st.sidebar.number_input("Median Salary ($)", 5000, 1000000, 60000, 1000)
    experience = st.sidebar.slider("Experience (Years)", 0, 30, 5)
    remote = st.sidebar.slider("Remote Work Ratio (%)", 0, 100, 20)
    selected_ai = st.sidebar.selectbox("AI Impact Level", get_options('AI Impact Level'))
    openings_curr = st.sidebar.number_input("Job Openings (2024)", 0, 100000, 1000)
    openings_proj = st.sidebar.number_input("Projected Openings (2030)", 0, 100000, 1200)

    # User chooses model
    model_choice = st.selectbox("Choose Model for Prediction:", list(trained_models.keys()), index=0)
    active_model = trained_models[model_choice]

    if st.button("🚀 Analyze Risk"):
        try:
            # Encode Categorical Inputs
            input_dict = {
                'Job Title': encoders['Job Title'].transform([selected_job])[0],
                'Industry': encoders['Industry'].transform([selected_industry])[0],
                'Job Status': encoders['Job Status'].transform([selected_status])[0],
                'AI Impact Level': encoders['AI Impact Level'].transform([selected_ai])[0],
                'Required Education': encoders['Required Education'].transform([selected_edu])[0],
                'Location': encoders['Location'].transform([selected_loc])[0],
                'Median Salary (USD)': salary,
                'Experience Required (Years)': experience,
                'Job Openings (2024)': openings_curr,
                'Projected Openings (2030)': openings_proj,
                'Remote Work Ratio (%)': remote
            }

            # Create DataFrame for scaling
            input_df = pd.DataFrame([input_dict])

            # Scale Numerical Inputs using the same scaler fitted on training data
            num_cols_to_scale = ['Median Salary (USD)', 'Experience Required (Years)',
                                 'Job Openings (2024)', 'Projected Openings (2030)',
                                 'Remote Work Ratio (%)']

            # Ensure correct order of features
            final_input_df = input_df[feature_names].copy()
            final_input_df[num_cols_to_scale] = scaler.transform(final_input_df[num_cols_to_scale])

            # Predict
            prediction_grade = active_model.predict(final_input_df)[0]

            st.divider()

            # Interpret Results (Requirement: Interpret results)
            risk_map = {
                0: ("VERY SAFE (0-20%)", "success", "Automation unlikely."),
                1: ("SAFE (20-40%)", "success", "AI will likely be a tool, not a replacement."),
                2: ("MODERATE (40-60%)", "warning", "Hybrid role. Routine tasks automated."),
                3: ("HIGH RISK (60-80%)", "error", "Significant automation expected."),
                4: ("CRITICAL RISK (80-100%)", "error", "Highly likely to be fully automated.")
            }

            label, color, desc = risk_map[prediction_grade]

            if color == "success": st.balloons()
            if color == "error": st.error(f"🚨 {label}")
            elif color == "warning": st.warning(f"⚠️ {label}")
            else: st.success(f"✅ {label}")

            st.write(f"**Insight:** {desc}")
            st.info(f"Prediction made using **{model_choice}** algorithm.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
