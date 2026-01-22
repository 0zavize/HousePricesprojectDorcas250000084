from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model (relative path)
model = joblib.load("model/best_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        size = float(request.form["size"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])

        features = np.array([[size, bedrooms, bathrooms]])
        prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)

