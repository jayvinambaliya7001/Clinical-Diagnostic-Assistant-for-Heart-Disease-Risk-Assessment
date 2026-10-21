import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'heart_disease_rf_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Load the saved model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# The exact 30 features expected by the model
FEATURE_COLUMNS = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
    'sex_0', 'sex_1',
    'cp_0', 'cp_1', 'cp_2', 'cp_3',
    'fbs_0', 'fbs_1',
    'restecg_0', 'restecg_1', 'restecg_2',
    'exang_0', 'exang_1',
    'slope_0', 'slope_1', 'slope_2',
    'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4',
    'thal_0', 'thal_1', 'thal_2', 'thal_3'
]

COLUMNS_TO_SCALE = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

def preprocess_input(data):
    # Initialize all columns with 0
    row = {col: 0 for col in FEATURE_COLUMNS}
    
    # Continuous values
    row['age'] = float(data.get('age', 50))
    row['trestbps'] = float(data.get('trestbps', 120))
    row['chol'] = float(data.get('chol', 200))
    row['thalach'] = float(data.get('thalach', 150))
    row['oldpeak'] = float(data.get('oldpeak', 1.0))
    
    # Categorical one-hot encoding
    sex = int(data.get('sex', 1))
    cp = int(data.get('cp', 0))
    fbs = int(data.get('fbs', 0))
    restecg = int(data.get('restecg', 0))
    exang = int(data.get('exang', 0))
    slope = int(data.get('slope', 1))
    ca = int(data.get('ca', 0))
    thal = int(data.get('thal', 2))
    
    if f'sex_{sex}' in row: row[f'sex_{sex}'] = 1
    if f'cp_{cp}' in row: row[f'cp_{cp}'] = 1
    if f'fbs_{fbs}' in row: row[f'fbs_{fbs}'] = 1
    if f'restecg_{restecg}' in row: row[f'restecg_{restecg}'] = 1
    if f'exang_{exang}' in row: row[f'exang_{exang}'] = 1
    if f'slope_{slope}' in row: row[f'slope_{slope}'] = 1
    if f'ca_{ca}' in row: row[f'ca_{ca}'] = 1
    if f'thal_{thal}' in row: row[f'thal_{thal}'] = 1
    
    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    # Scale continuous columns
    input_df[COLUMNS_TO_SCALE] = scaler.transform(input_df[COLUMNS_TO_SCALE])
    return input_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form if request.form else request.get_json()
        input_df = preprocess_input(data)
        
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        prob_disease = round(float(probabilities[1]) * 100, 2)
        prob_healthy = round(float(probabilities[0]) * 100, 2)
        
        result = {
            "prediction": int(prediction),
            "status": "High Risk of Heart Disease" if prediction == 1 else "Low Risk / Normal",
            "probability_disease": prob_disease,
            "probability_healthy": prob_healthy,
            "success": True
        }
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify(result)
        return render_template('index.html', result=result, form_data=data)
    except Exception as e:
        error_res = {"success": False, "error": str(e)}
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify(error_res), 400
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    print("Heart Disease Prediction UI running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
