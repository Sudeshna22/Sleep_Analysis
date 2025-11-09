# ============================================================
# 💤 Sleep Quality Prediction – Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# 1️⃣ Load Models & Scaler
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    rf_model = joblib.load(os.path.join(BASE_DIR, "sleep_rf_model.pkl"))
    lr_model = joblib.load(os.path.join(BASE_DIR, "sleep_lr_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "sleep_scaler.pkl"))
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}")

# ------------------------------------------------------------
# 2️⃣ Page Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Sleep Quality Predictor 😴", page_icon="💤", layout="centered")

st.title("💤 Sleep Quality Prediction App")
st.markdown("Predict your sleep quality based on your **lifestyle and daily habits.**")
st.divider()

# ------------------------------------------------------------
# 3️⃣ User Inputs
# ------------------------------------------------------------
st.header("🧍 Enter Your Lifestyle Details")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 25)
occupation = st.selectbox(
    "Occupation",
    ["Software Engineer", "Doctor", "Teacher", "Nurse", "Lawyer", "Accountant", "Salesperson", "Scientist"],
)
activity = st.slider("Physical Activity (minutes/day)", 0, 120, 45)
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 120, 80)
steps = st.slider("Daily Steps", 1000, 15000, 5000)
sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea"])
systolic = st.slider("Systolic BP", 90, 160, 120)
diastolic = st.slider("Diastolic BP", 60, 100, 80)
caffeine = st.slider("☕ Caffeine Intake (mg/day)", 0, 400, 150)
screen_time = st.slider("📱 Screen Time (hours/day)", 1.0, 10.0, 5.0)

# ------------------------------------------------------------
# 4️⃣ Feature Encoding
# ------------------------------------------------------------
gender_encoded = 1 if gender == "Male" else 0
occupation_map = {
    "Software Engineer": 0,
    "Doctor": 1,
    "Teacher": 2,
    "Nurse": 3,
    "Lawyer": 4,
    "Accountant": 5,
    "Salesperson": 6,
    "Scientist": 7,
}
bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
sleep_disorder_map = {"None": 0, "Insomnia": 1, "Sleep Apnea": 2}

input_data = pd.DataFrame(
    {
        "Gender": [gender_encoded],
        "Age": [age],
        "Occupation": [occupation_map[occupation]],
        "Physical_Activity_Level": [activity],
        "BMI_Category": [bmi_map[bmi_category]],
        "Heart_Rate": [heart_rate],
        "Daily_Steps": [steps],
        "Sleep_Disorder": [sleep_disorder_map[sleep_disorder]],
        "Systolic_BP": [systolic],
        "Diastolic_BP": [diastolic],
        "Caffeine_Intake": [caffeine],
        "ScreenTime": [screen_time],
    }
)

# ------------------------------------------------------------
# 5️⃣ Scale and Predict
# ------------------------------------------------------------
try:
    input_scaled = scaler.transform(input_data)
    pred = rf_model.predict(input_data)[0]

    label_map = {0: "Good", 1: "Average", 2: "Poor"}
    result = label_map.get(pred, "Unknown")

    st.divider()
    st.subheader("🧠 Predicted Sleep Quality")

    if result == "Good":
        st.success("💤 Your sleep quality is **Good**! Keep up the healthy routine 😌")
    elif result == "Average":
        st.warning("😴 Your sleep quality is **Average** — maybe reduce caffeine or screen time.")
    else:
        st.error("⚠️ Your sleep quality is **Poor**. Try improving stress and sleep hygiene.")

except Exception as e:
    st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------
# 6️⃣ Footer
# ------------------------------------------------------------
st.divider()
st.caption("Developed with 💜 by **Sudeshna Acharyya** | Powered by Streamlit & Scikit-learn")
