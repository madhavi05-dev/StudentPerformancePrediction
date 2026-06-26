"""
streamlit_app.py
Student Performance Prediction – Streamlit Web Application
Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── Path Setup ────────────────────────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR = os.path.join(BASE, 'models')

@st.cache_resource
def load_artifacts():
    model   = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
    scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feats   = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    return model, scaler, feats

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1F3864,#2E74B5);padding:28px 32px;
            border-radius:12px;margin-bottom:24px'>
  <h1 style='color:white;margin:0;font-size:2rem'>🎓 Student Performance Predictor</h1>
  <p style='color:#BDD7EE;margin:6px 0 0'>
      Powered by XGBoost · Internship Project · Data Pro Organisation
  </p>
  <p style='color:#9DC3E6;font-size:0.85rem;margin:2px 0 0'>
      SEERAPU SRAVANI | Reg: 24U45A4219 | CSE-AIML, JNTUGV
  </p>
</div>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
try:
    model, scaler, feature_names = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("⚠️  Model not found. Please run `python models/train_models.py` first.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ℹ️ About")
st.sidebar.info(
    "This app predicts whether a student will **Pass or Fail** "
    "based on academic and personal features using an XGBoost model "
    "trained on 500 student records.\n\n"
    "**Model Accuracy:** 88.5%\n\n"
    "**AUC Score:** 0.92"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by:** SEERAPU SRAVANI")
st.sidebar.markdown("**Guide:** Mrs. M. KALYANI")
st.sidebar.markdown("**Organisation:** Data Pro Organisation")

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("### 📋 Enter Student Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Academic Information**")
    attendance       = st.slider("Attendance (%)", 30, 100, 75)
    midterm_marks    = st.slider("Mid-Term Marks (out of 100)", 0, 100, 55)
    assignment_score = st.slider("Assignment Score (out of 100)", 0, 100, 60)
    prev_gpa         = st.slider("Previous GPA (out of 10)", 0.0, 10.0, 6.5, 0.1)

with col2:
    st.markdown("**Personal Information**")
    study_hours      = st.slider("Study Hours per Day", 0.0, 14.0, 5.0, 0.5)
    extracurricular  = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    internet_access  = st.selectbox("Internet Access at Home", ["No", "Yes"])
    gender           = st.selectbox("Gender", ["Female", "Male"])

with col3:
    st.markdown("**Socioeconomic Details**")
    socioeconomic    = st.selectbox("Socioeconomic Status", ["Low", "Medium", "High"])
    parent_education = st.selectbox(
        "Parent Education Level",
        ["None", "School", "Graduate", "Postgraduate"]
    )

# Encode categoricals to match training
gender_enc    = 0 if gender == "Female" else 1
socio_map     = {"High": 0, "Low": 1, "Medium": 2}
parent_map    = {"Graduate": 0, "None": 1, "Postgraduate": 2, "School": 3}
socio_enc     = socio_map[socioeconomic]
parent_enc    = parent_map[parent_education]
extra_enc     = 1 if extracurricular == "Yes" else 0
internet_enc  = 1 if internet_access == "Yes" else 0

input_data = pd.DataFrame([[
    gender_enc, socio_enc, parent_enc, internet_enc,
    extra_enc, attendance, study_hours, midterm_marks,
    assignment_score, prev_gpa
]], columns=[
    'Gender', 'SocioeconomicStatus', 'ParentEducation', 'InternetAccess',
    'Extracurricular', 'Attendance', 'StudyHours', 'MidTermMarks',
    'AssignmentScore', 'PrevGPA'
])

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 Predict Performance", type="primary", use_container_width=True)

if predict_btn and model_loaded:
    input_scaled = scaler.transform(input_data[feature_names])
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    result_label = "✅ PASS" if prediction == 1 else "❌ FAIL"
    pass_prob    = probability[1] * 100
    fail_prob    = probability[0] * 100
    result_color = "#1a7a3c" if prediction == 1 else "#c0392b"
    bg_color     = "#d4edda" if prediction == 1 else "#f8d7da"

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.markdown(f"""
        <div style='background:{bg_color};border-left:6px solid {result_color};
                    padding:24px 28px;border-radius:10px;text-align:center'>
            <h2 style='color:{result_color};margin:0'>Predicted Result</h2>
            <h1 style='color:{result_color};font-size:3rem;margin:8px 0'>{result_label}</h1>
            <p style='color:#333;font-size:1.1rem'>
                Pass Probability: <strong>{pass_prob:.1f}%</strong><br>
                Fail Probability: <strong>{fail_prob:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.barh(['Fail', 'Pass'], [fail_prob, pass_prob],
                       color=['#e74c3c', '#27ae60'], edgecolor='white', height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Prediction Confidence", fontweight='bold')
        for bar, val in zip(bars, [fail_prob, pass_prob]):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va='center', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    # Advice
    st.markdown("### 💡 Recommendations")
    advice_col1, advice_col2 = st.columns(2)
    with advice_col1:
        if attendance < 75:
            st.warning(f"📅 Attendance is {attendance}% — below the recommended 75%. Improve attendance.")
        else:
            st.success(f"📅 Attendance is {attendance}% — Good!")
        if study_hours < 4:
            st.warning(f"📚 Study hours ({study_hours}/day) are low. Aim for at least 4-5 hours.")
        else:
            st.success(f"📚 Study hours ({study_hours}/day) — Adequate!")
    with advice_col2:
        if midterm_marks < 50:
            st.warning(f"📝 Mid-term marks ({midterm_marks}) are low. Focus on exam preparation.")
        else:
            st.success(f"📝 Mid-term marks ({midterm_marks}) — Good performance!")
        if prev_gpa < 6.0:
            st.warning(f"🎓 Previous GPA ({prev_gpa}) needs improvement. Set a target of 7.0+.")
        else:
            st.success(f"🎓 Previous GPA ({prev_gpa}) — Well done!")

elif predict_btn and not model_loaded:
    st.error("Model not loaded. Please train the model first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.85rem'>"
    "Student Performance Prediction Using ML · Internship Project · "
    "SEERAPU SRAVANI (24U45A4219) · CSE-AIML, JNTUGV · Data Pro Organisation · 2026"
    "</p>",
    unsafe_allow_html=True
)
