import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model once at startup
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature order the model expects
FEATURES = ["quantity", "unit_price", "discount"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = [float(request.form[feat]) for feat in FEATURES]
            X = np.array(values).reshape(1, -1)
            result = model.predict(X)[0]
            prediction = round(float(result), 2)
        except (KeyError, ValueError):
            error = "Please enter valid numeric values for all fields."

    return render_template("index.html", prediction=prediction, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint: POST {"quantity":.., "unit_price":.., "discount":..}"""
    data = request.get_json(force=True)
    try:
        values = [float(data[feat]) for feat in FEATURES]
    except (KeyError, TypeError, ValueError):
        return {"error": "Missing or invalid feature values"}, 400

    X = np.array(values).reshape(1, -1)
    result = model.predict(X)[0]
    return {"prediction": round(float(result), 2)}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)