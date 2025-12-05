import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load saved model
model = pickle.load(open("best_heart_model.pkl", "rb"))

st.title("❤️ Heart Disease Prediction Web App")

st.write("Enter patient details to check heart disease risk")


age = st.number_input("Age", min_value=1, max_value=120, value=50)
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)

sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
fbs = st.selectbox("Fasting Blood Sugar", ["True", "False"])
restecg = st.selectbox("ECG Result", ["normal", "ST-T abnormality", "left ventricular hypertrophy"])
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
thal = st.selectbox("Thal", ["normal", "fixed defect", "reversable defect"])
ca = st.number_input("Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)

# Convert into dataframe shape
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]],
                          columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"])

# Predict
if st.button("Predict"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.error("❌ High Chance of Heart Disease")
    else:
        st.success("✔ Low Chance of Heart Disease")
