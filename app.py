import os
from flask import Flask, render_template, request
import pandas as pd
import joblib

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Configuration
# ------------------------------
LINEAR_MODEL_PATH = 'linear_model.pkl'
LASSO_MODEL_PATH = 'lasso_model.pkl'
SCALER_PATH = 'scaler.pkl'

FEATURES = ["G1", "G2", "studytime", "failures", "absences"]
FEATURES_TO_NORMALIZE = ["failures", "G1", "G2", "absences"]

# ------------------------------
# Load assets
# ------------------------------
def load_assets():
    try:
        linear_model = joblib.load(LINEAR_MODEL_PATH)
        lasso_model = joblib.load(LASSO_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Models and scaler loaded successfully!")
        return linear_model, lasso_model, scaler
    except Exception as e:
        print(f"Error loading assets: {e}")
        return None, None, None

linear_model, lasso_model, scaler = load_assets()

# ------------------------------
# Prediction (Ensemble)
# ------------------------------
def predict_grade(input_df):
    df = input_df.copy()
    df[FEATURES_TO_NORMALIZE] = scaler.transform(df[FEATURES_TO_NORMALIZE])

    pred_linear = linear_model.predict(df)[0]
    pred_lasso = lasso_model.predict(df)[0]

    return (pred_linear + pred_lasso) / 2


# ------------------------------
# What-If Logic
# ------------------------------
def generate_what_if_scenarios(base_input):
    scenarios = []

    for delta in [5, 10]:
        if base_input["absences"] - delta >= 0:
            s = base_input.copy()
            s["scenario"] = f"Absences reduced by {delta}"
            s["absences"] -= delta
            scenarios.append(s)

    if base_input["studytime"] < 4:
        s = base_input.copy()
        s["scenario"] = "Study time increased by 1 level"
        s["studytime"] += 1
        scenarios.append(s)

    for delta in [3, 5]:
        if base_input["G2"] + delta <= 20:
            s = base_input.copy()
            s["scenario"] = f"G2 improved by {delta} points"
            s["G2"] += delta
            scenarios.append(s)

    return scenarios


def run_what_if_analysis(base_input):
    base_df = pd.DataFrame([base_input])
    baseline = predict_grade(base_df)

    scenarios = generate_what_if_scenarios(base_input)
    results = []

    for s in scenarios:
        scenario_name = s.pop("scenario")
        df = pd.DataFrame([s])
        new_grade = predict_grade(df)

        results.append({
            "scenario": scenario_name,
            "new_grade": round(new_grade, 2),
            "improvement": round(new_grade - baseline, 2)
        })

    return round(baseline, 2), results


# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if linear_model is None or lasso_model is None or scaler is None:
        return render_template(
            'index.html',
            error=True,
            message="Model loading failed"
        )

    try:
        # Collect form input
        base_input = {
            "G1": float(request.form.get("G1", 0)),
            "G2": float(request.form.get("G2", 0)),
            "studytime": float(request.form.get("studytime", 1)),
            "failures": float(request.form.get("failures", 0)),
            "absences": float(request.form.get("absences", 0))
        }

        baseline, what_if_results = run_what_if_analysis(base_input)

        return render_template(
            'index.html',
            error=False,
            baseline=baseline,
            what_if_results=what_if_results
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=True,
            message=str(e)
        )


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
