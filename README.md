# Sleep_Analysis
# 💤 Sleep Quality Prediction using Machine Learning

## 🧠 Overview
Good sleep = good life.  
This project leverages machine learning to predict a person's sleep quality based on lifestyle, stress levels, and activity data. Using real-world style synthetic data, the model identifies patterns between daily habits and overall sleep quality.

The goal? Help users understand what factors lead to a good night’s sleep — and what habits might be sabotaging it.

---

## 🎯 Project Objective
- **Business Problem:** Predict whether an individual’s sleep quality is *Good* or *Poor* based on health and lifestyle metrics.  
- **Goal:** Build and evaluate models that can classify sleep quality efficiently.  
- **Type:** Binary Classification  
- **Success Metrics:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  

---

## 📂 Project Structure
SleepQualityPrediction/
│
├── sleep_quality.csv # Dataset used for analysis and training
├── sleep_lr_model.pkl # Logistic Regression model
├── sleep_rf_model.pkl # Random Forest model
├── sleep_scaler.pkl # StandardScaler for feature scaling
├── sleep_quality_analysis.ipynb # Jupyter Notebook with full workflow
├── README.md # You’re reading it ;)
└── requirements.txt # Python dependencies


---

## 🔍 Workflow Summary

### 1️⃣ Problem Understanding
Identified the key question:  
*Can lifestyle and health factors predict sleep quality?*

### 2️⃣ Data Understanding & Exploration
- Loaded and analyzed dataset with **Pandas** and **Seaborn**
- Checked data types, missing values, distributions
- Visualized correlations and class balance

### 3️⃣ Feature Engineering
- Encoded categorical variables  
- Scaled numeric features  
- Engineered new columns like **Activity Index**, **Stress Ratio**, etc.

### 4️⃣ Model Building
Trained two models for comparison:
- **Logistic Regression** – interpretable baseline model  
- **Random Forest** – for higher accuracy and non-linear patterns  

### 5️⃣ Evaluation
Used metrics like:
- Accuracy  
- Precision / Recall / F1-score  
- Confusion Matrix  
- ROC-AUC Curve  

### 6️⃣ Prediction
Generated predictions on new data to test model performance in real-world-like conditions.

---

## 📊 Insights
- Higher stress and caffeine intake correlate with poor sleep quality.  
- Regular exercise and consistent sleep hours improve quality significantly.  
- Random Forest performed better overall, but Logistic Regression remains more interpretable.

---

## 🚀 How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/Sudeshna22/SleepQualityPrediction.git
   cd SleepQualityPrediction


🧩 Tech Stack

Language: Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

Environment: Jupyter Notebook

💡 Future Enhancements

Add deep learning models for time-series sleep data

Integrate smartwatch or fitness tracker data

Deploy model as a Streamlit app

✨ Author

Sudeshna
Data Engineer turned ML Explorer 💻
Always caffeinated ☕, occasionally sleep-deprived 😴
Let’s connect on LinkedIn: https://www.linkedin.com/in/sudeshna-acharyya-14182b1ba/
 🚀

