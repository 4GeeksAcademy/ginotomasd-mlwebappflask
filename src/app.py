from utils import db_connect
engine = db_connect()

# your code here
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '../models/ridge_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Amazon Stock Close Price Predictor 🚀'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get values from JSON
    try:
        open_val = float(data['Open'])
        high_val = float(data['High'])
        low_val = float(data['Low'])
        volume_val = float(data['Volume'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Please provide valid Open, High, Low, and Volume values'}), 400

    # Prepare input for model
    input_features = np.array([[open_val, high_val, low_val, volume_val]])

    # Make prediction
    predicted_close = model.predict(input_features)[0]

    # Return result
    return jsonify({'predicted_close': round(predicted_close, 2)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')