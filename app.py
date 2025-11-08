# =========================================================
# 💤 Sleep Quality Predictor - Streamlit App
# =========================================================

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# =========================================================
# Load Model and Scaler Safely
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'sleep_rf_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'sleep_scaler.pkl')

try:
    rf_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"❌ Missing model or scaler file: {e.filename}")
    st.stop()

# =========================================================
# Streamlit App Config
# =========================================================
st.set_page_config(page_title="Sleep Quality Predictor 😴", page_icon="💤")

st.title("💤 Sleep Quality Prediction App")
st.write("Predict your sleep quality based on your lifestyle and environment!")

# =========================================================
# User Inputs
# =========================================================
age = st.slider("Age", 18, 70, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
screen_time = st.slider("Daily Screen Time (hrs)", 0.5, 6.0, 3.5)
exercise = st.slider("Exercise per day (mins)", 0, 120, 30)
stress = st.slider("Stress level (1-10)", 1.0, 10.0, 5.0)
caffeine = st.selectbox("Caffeine intake (mg/day)", [0, 50, 100, 150, 200, 250])
noise = st.slider("Noise level (dB)", 20, 60, 35)

# =========================================================
# Feature Engineering
# =========================================================
sleep_hours = np.clip(8 - (screen_time * 0.2) - (stress * 0.1) + (exercise / 100) - (caffeine / 500), 3, 10)
caffeine_cat = 0 if caffeine <= 50 else (1 if caffeine <= 150 else 2)
stress_lvl = 0 if stress <= 4 else (1 if stress <= 7 else 2)
active = 1 if exercise > 60 else 0
noise_lvl = 1 if noise > 40 else 0
gender_num = 1 if gender == 'Male' else 0

# Prepare DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_num],
    'ScreenTime': [screen_time],
    'Exercise': [exercise],
    'Stress': [stress],
    'Caffeine': [caffeine],
    'Noise': [noise],
    'SleepHours': [sleep_hours],
    'CaffeineCategory': [caffeine_cat],
    'StressLevel': [stress_lvl],
    'Active': [active],
    'NoiseLevel': [noise_lvl]
})

# =========================================================
# Scale + Predict
# =========================================================
try:
    input_scaled = scaler.transform(input_df)
except Exception:
    # if your model was trained without scaling, fall back gracefully
    input_scaled = input_df.values

pred = rf_model.predict(input_scaled)

label_map = {0: 'Good', 1: 'Moderate', 2: 'Poor'}
result = label_map.get(pred[0], "Unknown")

# =========================================================
# Display Result
# =========================================================
st.subheader("🧠 Predicted Sleep Quality:")
st.markdown(f"### 💤 **{result}**")

st.caption("Model trained on synthetic behavioral and environmental data.")
#to run please type: streamlit run "C:\Users\Sudeshna\OneDrive\Desktop\python_datasc\sleep_analysis\app.py"