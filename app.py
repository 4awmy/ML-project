import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 🚨 PATH CONFIGURATION
# ==========================================
FILE_PATH = r"E:\College\Term 5\AI\Dataset\AI_Project\data.csv"

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Advanced Job Risk AI", layout="wide")
st.title("🤖 AI Job Market Risk Analyzer")
st.markdown("### Comprehensive Analysis based on Industry, Location, and Market Trends")

# --- 2. LOAD DATA & TRAIN MODEL ---
@st.cache_data
def get_model():
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None

    # --- PREPROCESSING ---
    encoders = {}
    
    # Categorical Columns to Encode
    cat_cols = ['Job Title', 'Industry', 'Job Status', 'AI Impact Level', 'Required Education', 'Location']
    
    for col in cat_cols:
        if col in df.columns:
            # Force everything to string to avoid errors
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            encoders[col] = LabelEncoder() # Empty dummy

    # Create Target (5 Grades of Risk)
    if 'Automation Risk (%)' in df.columns:
        def get_risk_grade(risk):
            if risk <= 20: return 0  # Very Safe
            elif risk <= 40: return 1 # Safe
            elif risk <= 60: return 2 # Moderate
            elif risk <= 80: return 3 # High Risk
            else: return 4            # Critical Risk
        df['Risk_Grade'] = df['Automation Risk (%)'].apply(get_risk_grade)
    else:
        st.error("Error: Target column 'Automation Risk (%)' not found.")
        return None, None, None
    
    # Define Features
    feature_cols = [
        'Job Title', 'Industry', 'Job Status', 'AI Impact Level', 
        'Median Salary (USD)', 'Required Education', 'Experience Required (Years)', 
        'Job Openings (2024)', 'Projected Openings (2030)', 'Remote Work Ratio (%)', 'Location'
    ]
    
    # Ensure we only use columns that exist
    final_features = [c for c in feature_cols if c in df.columns]
    
    X = df[final_features]
    y = df['Risk_Grade']
    
    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, encoders, final_features

model, encoders, feature_names = get_model()

# --- 3. SIDEBAR INPUTS ---
if model is None:
    st.error(f"🚨 CRITICAL ERROR: Could not find 'data.csv' at path: {FILE_PATH}")
    st.stop()

st.sidebar.header("1. Job Profile")

# Helper function to get sorted options safely
def get_options(col_name):
    if col_name in encoders and hasattr(encoders[col_name], 'classes_'):
        return sorted(encoders[col_name].classes_)
    return ["Unknown"]

# Inputs
selected_job = st.sidebar.selectbox("Job Title", get_options('Job Title'))
selected_industry = st.sidebar.selectbox("Industry", get_options('Industry'))
selected_status = st.sidebar.selectbox("Job Status", get_options('Job Status'))
selected_edu = st.sidebar.selectbox("Required Education", get_options('Required Education'))
selected_loc = st.sidebar.selectbox("Location", get_options('Location'))

st.sidebar.header("2. Market Data")
salary = st.sidebar.number_input("Median Salary ($)", min_value=5000, max_value=1000000, value=60000, step=1000)
experience = st.sidebar.slider("Experience (Years)", 0, 30, 5)
remote = st.sidebar.slider("Remote Work Ratio (%)", 0, 100, 20)
selected_ai = st.sidebar.selectbox("AI Impact Level", get_options('AI Impact Level'))
openings_curr = st.sidebar.number_input("Job Openings (2024)", min_value=0, value=1000, step=50)
openings_proj = st.sidebar.number_input("Projected Openings (2030)", min_value=0, value=1200, step=50)

# --- 4. PREDICTION LOGIC ---
st.write(f"## Analyzing Risk for: **{selected_job}**")
st.write(f"**Industry:** {selected_industry} | **Location:** {selected_loc}")

if st.button("🚀 Analyze Comprehensive Risk"):
    
    try:
        # Prepare inputs
        input_data = {
            'Job Title': encoders['Job Title'].transform([selected_job])[0],
            'Industry': encoders['Industry'].transform([selected_industry])[0],
            'Job Status': encoders['Job Status'].transform([selected_status])[0],
            'AI Impact Level': encoders['AI Impact Level'].transform([selected_ai])[0],
            'Median Salary (USD)': salary,
            'Required Education': encoders['Required Education'].transform([selected_edu])[0],
            'Experience Required (Years)': experience,
            'Job Openings (2024)': openings_curr,
            'Projected Openings (2030)': openings_proj,
            'Remote Work Ratio (%)': remote,
            'Location': encoders['Location'].transform([selected_loc])[0]
        }
        
        # Organize inputs in the correct order
        final_input_array = []
        for feature in feature_names:
            final_input_array.append(input_data[feature])
        
        # Predict
        prediction_grade = model.predict([final_input_array])[0]
        
        st.markdown("---")
        
        # Display Results
        if prediction_grade == 0:
            st.balloons()
            st.success(f"🌟 **VERY SAFE (0-20% Risk)**")
            st.write(f"**{selected_job}** is highly secure. Automation is very unlikely.")
            
        elif prediction_grade == 1:
            st.success(f"✅ **SAFE (20-40% Risk)**")
            st.write(f"**{selected_job}** is a secure role. AI will likely be a tool, not a replacement.")
            
        elif prediction_grade == 2:
            st.warning(f"⚠️ **MODERATE (40-60% Risk)**")
            st.write(f"**{selected_job}** is a hybrid role. Routine tasks will be automated.")
            
        elif prediction_grade == 3:
            st.error(f"🚨 **HIGH RISK (60-80% Risk)**")
            st.write(f"**{selected_job}** is vulnerable. Significant automation is expected.")
            
        elif prediction_grade == 4:
            st.error(f"🤖 **CRITICAL RISK (80-100% Risk)**")
            st.write(f"**{selected_job}** is highly likely to be fully automated soon.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")