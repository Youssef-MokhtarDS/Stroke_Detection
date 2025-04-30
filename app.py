import streamlit as st
import numpy as np
import pickle

# Load the trained models
with open('stroke_risk_classification_model.pkl', 'rb') as f:
    model_classification = pickle.load(f)

with open('stroke_risk_regression_model.pkl', 'rb') as f:
    model_regression = pickle.load(f)

# Define age category function
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

# One-hot encoding for age category (4 categories used in model)
def one_hot_encode_age(age_category):
    return [
        1 if age_category == 'Adult' else 0,
        1 if age_category == 'Middle Aged' else 0,
        1 if age_category == 'Senior' else 0,
        1 if age_category == 'Teenager' else 0,
        1 if age_category == 'Adolescence' else 0,

    ]

# Streamlit UI
st.title("Stroke Risk Prediction App")

# User input
age = st.slider("Age", 0, 100, 30)
gender_input = st.selectbox("Gender", ['Male', 'Female'])
gender_encoded = 1 if gender_input == 'Male' else 0  # manual encoding

# Symptoms
chest_pain = st.checkbox("Chest Pain")
high_blood_pressure = st.checkbox("High Blood Pressure")
irregular_heartbeat = st.checkbox("Irregular Heartbeat")
shortness_of_breath = st.checkbox("Shortness of Breath")
fatigue_weakness = st.checkbox("Fatigue or Weakness")
dizziness = st.checkbox("Dizziness")
swelling_edema = st.checkbox("Swelling or Edema")
neck_jaw_pain = st.checkbox("Neck or Jaw Pain")
excessive_sweating = st.checkbox("Excessive Sweating")
persistent_cough = st.checkbox("Persistent Cough")
nausea_vomiting = st.checkbox("Nausea or Vomiting")
chest_discomfort = st.checkbox("Chest Discomfort")
cold_hands_feet = st.checkbox("Cold Hands or Feet")
snoring_sleep_apnea = st.checkbox("Snoring or Sleep Apnea")
anxiety_doom = st.checkbox("Anxiety or Sense of Doom")

# Predict
if st.button("Predict Stroke Risk"):
    age_cat = categorize_age(age)
    age_encoded = one_hot_encode_age(age_cat)

    input_data = np.array([[ 
        age,
        gender_encoded,
        int(chest_pain),
        int(high_blood_pressure),
        int(irregular_heartbeat),
        int(shortness_of_breath),
        int(fatigue_weakness),
        int(dizziness),
        int(swelling_edema),
        int(neck_jaw_pain),
        int(excessive_sweating),
        int(persistent_cough),
        int(nausea_vomiting),
        int(chest_discomfort),
        int(cold_hands_feet),
        int(snoring_sleep_apnea),
        int(anxiety_doom),
        *age_encoded
    ]])

    if input_data.shape[1] != 21:
        st.error(f"Feature mismatch: model expects 21 features, but got {input_data.shape[1]}")
    else:
        class_pred = model_classification.predict(input_data)[0]
        reg_pred = model_regression.predict(input_data)[0]

        st.success(f"Risk Category: {'At Risk' if class_pred == 1 else 'Not At Risk'}")
        st.info(f"Estimated Stroke Risk Percentage: {reg_pred:.2f}%")
