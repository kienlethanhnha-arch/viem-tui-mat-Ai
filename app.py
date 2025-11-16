import streamlit as st
import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify
# Load model
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    print("⚠️ MODEL_PATH không tồn tại:", MODEL_PATH)
print(f"WARNING: model file not found at {MODEL_PATH}")




@app.route('/')
def index():
return "Model prediction API is running. POST JSON to /predict"




@app.route('/predict', methods=['POST'])
def predict():
if model is None:
return jsonify({'error': 'Model not loaded on server'}), 500


data = request.get_json()
if data is None:
return jsonify({'error': 'Request body must be JSON'}), 400


# Accept either dict of named features or an array in the correct order
# If user sends names, we will extract values according to FEATURE_ORDER.
if isinstance(data, dict):
try:
row = [data[name] for name in FEATURE_ORDER]
except KeyError as e:
return jsonify({'error': f'Missing feature: {str(e)}', 'expected_features': FEATURE_ORDER}), 400
elif isinstance(data, list):
if len(data) != len(FEATURE_ORDER):
return jsonify({'error': f'Expected list of length {len(FEATURE_ORDER)}'}), 400
row = data
else:
return jsonify({'error': 'JSON must be an object (named features) or list (ordered features)'}), 400


# Build dataframe with correct column names (to be safe)
X = pd.DataFrame([row], columns=FEATURE_ORDER)


# Convert types (attempt)
X = X.apply(pd.to_numeric, errors='coerce')


if X.isnull().any(axis=None):
return jsonify({'error': 'One or more feature values are missing or not numeric', 'row': X.to_dict(orient='records')[0]}), 400


# Predict probability
try:
if hasattr(model, 'predict_proba'):
prob = model.predict_proba(X)[:, 1][0]
else:
# fallback: use predict (0/1) — we will return 0% or 100% which is less ideal
pred = model.predict(X)[0]
prob = float(pred)
except Exception as e:
return jsonify({'error': f'Model prediction error: {str(e)}'}), 500


# Return as percent
return jsonify({
'probability': float(prob),
'probability_percent': float(prob) * 100.0,
'features_used': FEATURE_ORDER
})




if __name__ == '__main__':
app.run(host='0.0.0.0', port=5000, debug=True)
