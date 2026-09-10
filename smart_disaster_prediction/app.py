from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("flood_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    rainfall = float(request.form["rainfall"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    wind_speed = float(request.form["wind_speed"])
    water_level = float(request.form["water_level"])

    data = [[
        rainfall,
        temperature,
        humidity,
        wind_speed,
        water_level
    ]]

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    risk_score = round(probability * 100, 2)

    if prediction == 1:
        risk_level = "HIGH"
        message = "⚠️ High flood risk detected!"
    else:
        risk_level = "LOW"
        message = "✅ Low flood risk detected."

    return render_template(
        "result.html",
        risk_level=risk_level,
        risk_score=risk_score,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)