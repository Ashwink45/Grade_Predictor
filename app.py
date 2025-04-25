import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = 'grade_predictor_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURES_TO_NORMALIZE = ["failures", "G1", "G2", "absences"]

# Load model and scaler with error handling
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"Error loading model/scaler: {str(e)}")
        return None, None

model, scaler = load_assets()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Early return if model/scaler not loaded
    if model is None or scaler is None:
        return render_template('index.html', 
                            prediction="System error: Model not loaded properly",
                            error=True)

    try:
        # Validate and convert form inputs
        form_data = {
            'G1': float(request.form.get('G1', 0)),
            'G2': float(request.form.get('G2', 0)),
            'studytime': float(request.form.get('studytime', 0)),
            'failures': float(request.form.get('failures', 0)),
            'absences': float(request.form.get('absences', 0))
        }

        # Create DataFrame and normalize features
        input_df = pd.DataFrame([form_data])
        input_df[FEATURES_TO_NORMALIZE] = scaler.transform(input_df[FEATURES_TO_NORMALIZE])

        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return render_template('index.html', 
                            prediction=f'Predicted Grade: {prediction:.2f}',
                            error=False)

    except ValueError as e:
        return render_template('index.html',
                            prediction=f'Invalid input: {str(e)}',
                            error=True)
    except Exception as e:
        return render_template('index.html',
                            prediction=f'Prediction failed: {str(e)}',
                            error=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', 
           port=int(os.environ.get('PORT', 5000)), 
           debug=os.environ.get('DEBUG', True))