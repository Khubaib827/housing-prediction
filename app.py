from flask import Flask, render_template, request, redirect, url_for, flash
from joblib import load
import pandas as pd
from pathlib import Path

app = Flask(__name__)
app.secret_key = "dev-secret"

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"

FEATURES = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

def load_model():
    if not MODEL_PATH.exists():
        return None
    return load(MODEL_PATH)

model = load_model()

@app.route("/", methods=["GET"])
def form():
    if model is None:
        flash("Model not found. Please run `python train_and_save.py` first.", "error")
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        flash("Model not loaded. Train & save it first.", "error")
        return redirect(url_for("form"))

    try:
        values = []
        for f in FEATURES:
            raw = request.form.get(f, "").strip()
            if raw == "":
                raise ValueError(f"Missing value for {f}")
            values.append(float(raw))
        df = pd.DataFrame([values], columns=FEATURES)
        pred_thousands = model.predict(df)[0]
        price = float(pred_thousands) * 1000.0  # MEDV is in $1000s
        return render_template("result.html", price=price, inputs=dict(zip(FEATURES, values)))
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("form"))

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)