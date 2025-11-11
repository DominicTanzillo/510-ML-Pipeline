from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("rf_model.pkl")
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = np.array([data[key] for key in sorted(data.keys())]).reshape(1, -1)
    pred = model.predict(X)[0]
    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    app.run()
