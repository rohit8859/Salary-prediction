import streamlit as st
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered",
)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists("artifacts.pkl"):
        st.error("⚠️ artifacts.pkl not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load("artifacts.pkl")

artifacts   = load_artifacts()
model       = artifacts["model"]
le_degree   = artifacts["le_degree"]
le_jobtitle = artifacts["le_jobtitle"]
le_gender   = artifacts["le_gender"]
age_mean    = artifacts["age_mean"]
age_std     = artifacts["age_std"]
exp_mean    = artifacts["exp_mean"]
exp_std     = artifacts["exp_std"]
job_titles  = sorted(artifacts["job_titles"])
degrees     = sorted(artifacts["degrees"])
genders     = sorted(artifacts["genders"])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center;'>💼 Employee Salary Predictor</h1>
    <p style='text-align:center; color:gray;'>Linear Regression · Built by Rohit</p>
    <hr>
""", unsafe_allow_html=True)

# ── Sidebar – About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app predicts an employee's salary based on:
    - **Age**
    - **Years of Experience**
    - **Education Level**
    - **Job Title**
    - **Gender**

    **Model:** Linear Regression  
    **Framework:** Scikit-learn  
    **UI:** Streamlit
    """)
    st.markdown("---")
    st.write("📁 [GitHub Repository](#)")  # Replace # with your repo URL

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("🔢 Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age        = st.slider("Age", min_value=18, max_value=65, value=30, step=1)
    experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5, step=1)
    gender     = st.selectbox("Gender", genders)

with col2:
    degree    = st.selectbox("Education Level", degrees)
    job_title = st.selectbox("Job Title", job_titles)

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("💡 Predict Salary", use_container_width=True):

    # Scale age & experience the same way as training
    age_scaled = (age - age_mean) / age_std
    exp_scaled = (experience - exp_mean) / exp_std

    # Encode categorical features
    try:
        degree_enc    = le_degree.transform([degree])[0]
        jobtitle_enc  = le_jobtitle.transform([job_title])[0]
        gender_enc    = le_gender.transform([gender])[0]
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    features = np.array([[age_scaled, exp_scaled, degree_enc, jobtitle_enc, gender_enc]])
    salary   = model.predict(features)[0]

    st.markdown("---")
    st.markdown(f"""
        <div style='background:#1e3a5f;padding:20px;border-radius:12px;text-align:center;'>
            <h2 style='color:#4fc3f7;'>Predicted Salary</h2>
            <h1 style='color:white;font-size:3rem;'>${salary:,.0f}</h1>
            <p style='color:#aaa;'>Based on the provided employee profile</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Input Summary ─────────────────────────────────────────────────────────
    st.markdown("#### 📋 Input Summary")
    summary_cols = st.columns(5)
    labels  = ["Age", "Experience", "Gender", "Degree", "Job Title"]
    values  = [age, f"{experience} yrs", gender, degree, job_title]
    for col, lbl, val in zip(summary_cols, labels, values):
        col.metric(lbl, val)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;font-size:0.8rem;'>Made with ❤️ using Streamlit · Raj Kumar Goel Institute of Technology</p>", unsafe_allow_html=True)
