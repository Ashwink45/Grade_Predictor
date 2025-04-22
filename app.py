from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model and scaler
model = joblib.load('grade_predictor_model.pkl')  # Assuming you've saved your model as model.pkl
scaler = joblib.load('scaler.pkl')  # Assuming you've saved your scaler as scaler.pkl

# Define features to normalize
features_to_normalize = ["failures", "G1", "G2", "absences"]

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        G1 = int(request.form['G1'])
        G2 = int(request.form['G2'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        absences = int(request.form['absences'])

        # Prepare input data for prediction
        input_data = pd.DataFrame([{
            "G1": G1,
            "G2": G2,
            "studytime": studytime,
            "failures": failures,
            "absences": absences
        }])

        # Normalize the input data
        input_data[features_to_normalize] = scaler.transform(input_data[features_to_normalize])

        # Predict the grade using the loaded model
        predicted_grade = model.predict(input_data)[0]
        
        # Display the result
        return render_template('index.html', prediction=f'{predicted_grade:.2f}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
