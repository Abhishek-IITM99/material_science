from flask import Flask, render_template, request
import numpy as np
import joblib

# Load model
model = joblib.load("gradient_boosting_model.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            size = float(request.form["size"])
            number = float(request.form["number"])
            input_features = np.array([[size, number]])
            prediction = model.predict(input_features)[0]
        except ValueError:
            prediction = "Invalid input. Please enter numbers only."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)