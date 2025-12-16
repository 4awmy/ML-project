import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_utils import load_and_process_data, train_models, NUM_COLS_TO_SCALE

# ==========================================
# 🚨 PATH CONFIGURATION
# ==========================================
FILE_PATH = "data.csv"

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Advanced Job Risk AI", layout="wide", page_icon="🤖")

def main():
    st.title("🤖 AI Job Market Risk Analyzer")
    st.markdown("### CAI3101 Project: End-to-End Machine Learning Workflow")
    st.markdown("""
    This application predicts the likelihood of job automation using Machine Learning.
    It demonstrates the full ML lifecycle: **Data Processing -> Model Training -> Evaluation -> Deployment**.
    """)

    # Load Data
    df_original, df_processed, feature_names, encoders, scaler = load_and_process_data(FILE_PATH)

    if df_original is None:
        st.error(f"Error: Could not load data from {FILE_PATH}. Please check if the file exists and has the correct format.")
        st.stop()

    # --- TABS FOR PROJECT REQUIREMENTS ---
    tab1, tab2, tab3 = st.tabs(["📊 Data Understanding", "⚙️ Model Development & Eval", "🚀 Risk Prediction"])

    # ==========================================
    # TAB 1: DATA UNDERSTANDING (Requirement: Statistics & Viz)
    # ==========================================
    with tab1:
        st.header("1. Data Understanding & Statistics")
        st.info("Requirement: Describe dataset source, context, summary statistics, and visualizations.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Context")
            st.write(f"**Total Samples:** {df_original.shape[0]}")
            st.write(f"**Total Features:** {df_original.shape[1]}")
            with st.expander("View Raw Data"):
                st.dataframe(df_original.head(10))

        with col2:
            st.subheader("Summary Statistics")
            st.dataframe(df_original.describe())

        st.subheader("Data Visualizations")
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            st.write("**Risk Distribution (Target Variable)**")
            fig = px.histogram(df_original, x='Automation Risk (%)', nbins=20, title="Distribution of Automation Risk", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)

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
        st.info("Requirement: Train at least 3 models, split data, and evaluate using accuracy.")

        st.write("Splitting dataset into Training (80%) and Testing (20%) sets.")

        # Prepare X and y
        X = df_processed[feature_names]
        y = df_processed['Risk_Grade']

        # Train/Test Split (Requirement)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Models
        with st.spinner("Training models..."):
            trained_models = train_models(X_train, y_train)

        # Evaluate Models
        results = []
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({'Model': name, 'Accuracy': acc})

        # Display Results
        results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

        st.subheader("Model Performance")
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
            best_model_name = results_df.iloc[0]['Model']
            st.success(f"🏆 Best Model: **{best_model_name}**")

        with col_res2:
            fig = px.bar(results_df, x='Model', y='Accuracy', title="Model Accuracy Comparison", color='Accuracy', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # TAB 3: PREDICTION (Requirement: Actionable Insights)
    # ==========================================
    with tab3:
        st.header("3. Interactive Risk Analyzer")
        st.info("Requirement: Interpret results and translate them into actionable insights.")

        # Layout: Sidebar for inputs, Main area for result
        with st.container():
            col_input, col_pred = st.columns([1, 2])

            with col_input:
                st.subheader("Job Parameters")

                def get_options(col_name):
                    if col_name in encoders:
                        return sorted(encoders[col_name].classes_)
                    return ["Unknown"]

                # Inputs
                selected_job = st.selectbox("Job Title", get_options('Job Title'))
                selected_industry = st.selectbox("Industry", get_options('Industry'))
                selected_status = st.selectbox("Job Status", get_options('Job Status'))
                selected_edu = st.selectbox("Required Education", get_options('Required Education'))
                selected_loc = st.selectbox("Location", get_options('Location'))
                selected_ai = st.selectbox("AI Impact Level", get_options('AI Impact Level'))

                salary = st.number_input("Median Salary ($)", 5000, 1000000, 60000, 1000)
                experience = st.slider("Experience (Years)", 0, 30, 5)
                remote = st.slider("Remote Work Ratio (%)", 0, 100, 20)
                openings_curr = st.number_input("Job Openings (2024)", 0, 100000, 1000)
                openings_proj = st.number_input("Projected Openings (2030)", 0, 100000, 1200)

                # User chooses model
                model_choice = st.selectbox("Choose Model:", list(trained_models.keys()), index=0)
                active_model = trained_models[model_choice]

                analyze_btn = st.button("🚀 Analyze Risk", type="primary")

            with col_pred:
                st.subheader("Prediction Result")
                if analyze_btn:
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

                        # Ensure correct order of features
                        final_input_df = input_df[feature_names].copy()

                        # Identify which columns to scale that actually exist in the input
                        cols_to_scale = [c for c in NUM_COLS_TO_SCALE if c in final_input_df.columns]

                        # Scale using the imported constant to ensure consistency
                        if cols_to_scale:
                            final_input_df[cols_to_scale] = scaler.transform(final_input_df[cols_to_scale])

                        # Predict
                        prediction_grade = active_model.predict(final_input_df)[0]

                        # Probabilities (if supported)
                        probs = None
                        if hasattr(active_model, "predict_proba"):
                            probs = active_model.predict_proba(final_input_df)[0]

                        # Interpret Results (Requirement: Interpret results)
                        risk_map = {
                            0: ("VERY SAFE (0-20%)", "success", "Automation is unlikely. This role requires high human intelligence/creativity."),
                            1: ("SAFE (20-40%)", "success", "AI will likely be a productivity tool, not a replacement."),
                            2: ("MODERATE (40-60%)", "warning", "Hybrid role. Routine tasks may be automated, but human oversight is needed."),
                            3: ("HIGH RISK (60-80%)", "error", "Significant automation expected. Upskilling is recommended."),
                            4: ("CRITICAL RISK (80-100%)", "error", "Highly likely to be fully automated. Consider pivoting to adjacent roles.")
                        }

                        label, color, desc = risk_map[prediction_grade]

                        st.divider()

                        if color == "success":
                            st.balloons()
                            st.success(f"### {label}")
                        elif color == "warning":
                            st.warning(f"### {label}")
                        else:
                            st.error(f"### {label}")

                        st.markdown(f"**Insight:** {desc}")

                        if probs is not None:
                            st.write("---")
                            st.write("**Confidence Scores:**")
                            prob_df = pd.DataFrame(probs, index=["Very Safe", "Safe", "Moderate", "High Risk", "Critical Risk"], columns=["Probability"])
                            st.bar_chart(prob_df)

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                else:
                    st.info("👈 Adjust parameters and click 'Analyze Risk' to see the prediction.")

if __name__ == "__main__":
    main()
