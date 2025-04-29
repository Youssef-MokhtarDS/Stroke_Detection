import streamlit as st
import numpy as np
import pickle
import os

# Change the current working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the classification model and regression model
with open("stroke_risk_classification_model.pkl", "rb") as f:
    classification_model = pickle.load(f)

with open("stroke_risk_regression_model.pkl", "rb") as f:
    regression_model = pickle.load(f)

# Define the app title
st.title("Stroke Risk Prediction App")

# Collect user inputs via Streamlit widgets
age = st.slider("Age", 0, 100, 30)
gender = st.selectbox("Gender", ["Female", "Male"])
chest_pain = st.checkbox("Chest Pain")
high_bp = st.checkbox("High Blood Pressure")
irregular_heartbeat = st.checkbox("Irregular Heartbeat")
sob = st.checkbox("Shortness of Breath")
fatigue = st.checkbox("Fatigue/Weakness")
dizziness = st.checkbox("Dizziness")
swelling = st.checkbox("Swelling/Edema")
jaw_pain = st.checkbox("Neck/Jaw Pain")
sweating = st.checkbox("Excessive Sweating")
cough = st.checkbox("Persistent Cough")
nausea = st.checkbox("Nausea/Vomiting")
discomfort = st.checkbox("Chest Discomfort")
cold_feet = st.checkbox("Cold Hands/Feet")
snoring = st.checkbox("Snoring/Sleep Apnea")
anxiety = st.checkbox("Anxiety/Doom")

# Button to trigger prediction
if st.button("Predict Stroke Risk"):
    # Prepare input data for the models (ensure it has exactly 22 features)
    input_data = np.array([[
        age,
        1 if gender == "Male" else 0,
        int(chest_pain),
        int(high_bp),
        int(irregular_heartbeat),
        int(sob),
        int(fatigue),
        int(dizziness),
        int(swelling),
        int(jaw_pain),
        int(sweating),
        int(cough),
        int(nausea),
        int(discomfort),
        int(cold_feet),
        int(snoring),
        int(anxiety)
    ]])

    # Check if the input data shape is correct
    st.write(f"Input data shape: {input_data.shape}")

    # Classification Model: Predict if the person is at risk
    risk_classification = classification_model.predict(input_data)[0]

    # Regression Model: Predict the stroke risk percentage
    stroke_risk_percentage = regression_model.predict(input_data)[0]

    # Show the results
    if risk_classification == 1:
        st.success(f"Prediction: At Risk (Risk Percentage: {stroke_risk_percentage:.2f}%)")
    else:
        st.success(f"Prediction: Not at Risk (Risk Percentage: {stroke_risk_percentage:.2f}%)")
