from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained models (ensure you have saved them)
model_classification = joblib.load('random_forest_classification_model.pkl')  # Replace with your actual model file
model_regression = joblib.load('random_forest_regression_model.pkl')  # Replace with your actual model file

# Load label encoder for Gender and AgeCategory encoding
le_gender = joblib.load('label_encoder_gender.pkl')  # Replace with your actual label encoder file

# Define a function to categorize age
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

# Define a function to preprocess the data
def preprocess_input_data(age, gender, chest_pain, high_blood_pressure, irregular_heartbeat, 
                          shortness_of_breath, fatigue_weakness, dizziness, swelling_edema, 
                          neck_jaw_pain, excessive_sweating, persistent_cough, nausea_vomiting, 
                          chest_discomfort, cold_hands_feet, snoring_sleep_apnea, anxiety_doom):
    
    # Age Category (One-hot encode)
    age_category = categorize_age(age)
    age_category_dict = {
        'New Born': [1, 0, 0, 0],
        'Toddler': [0, 1, 0, 0],
        'Preschooler': [0, 0, 1, 0],
        'School Age': [0, 0, 0, 1],
        'Teenager': [0, 0, 0, 1],
        'Adolescence': [0, 0, 0, 1],
        'Adult': [1, 0, 0, 0],
        'Middle Aged': [0, 1, 0, 0],
        'Senior': [0, 0, 1, 0],
    }
    
    # Age Category One-hot encoding
    age_category_encoded = age_category_dict.get(age_category, [0, 0, 0, 0])

    # Gender Encoding
    gender_encoded = le_gender.transform([gender])[0]

    # Prepare input data array for prediction
    input_data = np.array([[ 
        age,  # Age
        gender_encoded,  # Gender
        chest_pain,  # Chest Pain
        high_blood_pressure,  # High Blood Pressure
        irregular_heartbeat,  # Irregular Heartbeat
        shortness_of_breath,  # Shortness of Breath
        fatigue_weakness,  # Fatigue/Weakness
        dizziness,  # Dizziness
        swelling_edema,  # Swelling Edema
        neck_jaw_pain,  # Neck/Jaw Pain
        excessive_sweating,  # Excessive Sweating
        persistent_cough,  # Persistent Cough
        nausea_vomiting,  # Nausea/Vomiting
        chest_discomfort,  # Chest Discomfort
        cold_hands_feet,  # Cold Hands/Feet
        snoring_sleep_apnea,  # Snoring/Sleep Apnea
        anxiety_doom,  # Anxiety/Doom
        age_category_encoded[0],  # AgeCategory_Adult
        age_category_encoded[1],  # AgeCategory_Middle Aged
        age_category_encoded[2],  # AgeCategory_Senior
        age_category_encoded[3],  # AgeCategory_Teenager
    ]])

    return input_data

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    age = float(request.form['age'])
    gender = request.form['gender']
    chest_pain = int(request.form['chest_pain'])
    high_blood_pressure = int(request.form['high_blood_pressure'])
    irregular_heartbeat = int(request.form['irregular_heartbeat'])
    shortness_of_breath = int(request.form['shortness_of_breath'])
    fatigue_weakness = int(request.form['fatigue_weakness'])
    dizziness = int(request.form['dizziness'])
    swelling_edema = int(request.form['swelling_edema'])
    neck_jaw_pain = int(request.form['neck_jaw_pain'])
    excessive_sweating = int(request.form['excessive_sweating'])
    persistent_cough = int(request.form['persistent_cough'])
    nausea_vomiting = int(request.form['nausea_vomiting'])
    chest_discomfort = int(request.form['chest_discomfort'])
    cold_hands_feet = int(request.form['cold_hands_feet'])
    snoring_sleep_apnea = int(request.form['snoring_sleep_apnea'])
    anxiety_doom = int(request.form['anxiety_doom'])
    
    # Preprocess input data
    input_data = preprocess_input_data(
        age, gender, chest_pain, high_blood_pressure, irregular_heartbeat, 
        shortness_of_breath, fatigue_weakness, dizziness, swelling_edema, 
        neck_jaw_pain, excessive_sweating, persistent_cough, nausea_vomiting, 
        chest_discomfort, cold_hands_feet, snoring_sleep_apnea, anxiety_doom
    )

    # Make predictions (classification and regression)
    classification_prediction = model_classification.predict(input_data)
    regression_prediction = model_regression.predict(input_data)

    # Return the result
    return jsonify({
        'classification_result': classification_prediction[0],
        'regression_result': regression_prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
