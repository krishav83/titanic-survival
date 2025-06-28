from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model/titanic_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Extract data from form using correct lowercase keys
    pclass = int(request.form["pclass"])
    sex = 1 if request.form["sex"] == "male" else 0
    age = float(request.form["age"])
    sibsp = int(request.form["sibsp"])
    parch = int(request.form["parch"])
    fare = float(request.form["fare"])
    embarked = request.form.get("embarked", "S")  # default to S if missing

    # Map embarked to numeric
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    embarked_val = embarked_map.get(embarked, 2)

    # Prepare input
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_val]])

    # Predict and get probability
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Probability of survival

    result = "Survived" if prediction == 1 else "Did Not Survive"

    return render_template("result.html", prediction=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
