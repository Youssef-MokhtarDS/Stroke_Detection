import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd

# Set the working directory to the current script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the classification model and regression model
with open("stroke_risk_classification_model.pkl", "rb") as f:
    classification_model = pickle.load(f)

with open("stroke_risk_regression_model.pkl", "rb") as f:
    regression_model = pickle.load(f)

# Define the app title
st.title("Stroke Risk Prediction App")

# Define age categories based on detailed ranges
def categorize_age(age):
    if age >= 0 and age <= 1:
        return 'New Born'
    elif age > 1 and age <= 3:
        return 'Toddler'
    elif age > 3 and age <= 6:
        return 'Preschooler'
    elif age > 6 and age <= 12:
        return 'School Age'
    elif age > 12 and age < 20:
        return 'Teenager'
    elif age >= 20 and age <= 24:
        return 'Adolescence'
    elif age > 24 and age <= 39:
        return 'Adult'
    elif age > 39 and age <= 59:
        return 'Middle Aged'
    else:
        return 'Senior'

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

# Get the age category using the function
age_category = categorize_age(age)

# Button to trigger prediction
if st.button("Predict Stroke Risk"):
    # Prepare input data for the models
    input_data = np.array([[ 
        age,
        1 if gender == "Male" else 0,  # Gender encoding
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
        int(anxiety),
        1 if age_category == "Adult" else 0,  # One-hot encoding for 'Adult'
        1 if age_category == "Middle Aged" else 0,  # One-hot encoding for 'Middle Aged'
        1 if age_category == "Senior" else 0,  # One-hot encoding for 'Senior'
    ]])

    # Classification Model: Predict if the person is at risk
    risk_classification = classification_model.predict(input_data)[0]

    # Regression Model: Predict the stroke risk percentage
    stroke_risk_percentage = regression_model.predict(input_data)[0]

    # Show the results
    if risk_classification == 1:
        st.success(f"Prediction: At Risk (Risk Percentage: {stroke_risk_percentage:.2f}%)")
    else:
        st.success(f"Prediction: Not at Risk (Risk Percentage: {stroke_risk_percentage:.2f}%)")
