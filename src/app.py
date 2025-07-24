from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../models/ridge_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        open_val = float(request.form['Open'])
        high_val = float(request.form['High'])
        low_val = float(request.form['Low'])
        volume_val = float(request.form['Volume'])

        input_features = np.array([[open_val, high_val, low_val, volume_val]])
        predicted_close = model.predict(input_features)[0]

        return render_template('index.html', prediction=round(predicted_close, 2))
    except:
        return render_template('index.html', prediction="Error in input. Please enter valid numbers.")
    
    
# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)