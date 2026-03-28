import os

from flask import Flask, request, jsonify
from src.data_ingestion import load_data
from src.data_preprocessing import split_data
from src.model_training import train_model

app = Flask(__name__)

# Train model once when app starts
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)


@app.route("/")
def home():
    return "MLOps Model API is Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    prediction = model.predict([data])
    return jsonify({"prediction": int(prediction[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)