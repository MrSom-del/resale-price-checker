from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load trained models and encoders once, at startup
models = joblib.load("trained_models.pkl")
encoders = joblib.load("encoders.pkl")

df = pd.read_csv("listings_cleaned.csv")  # used for comparable listings lookup

FEATURE_COLS = [
    "brand_enc", "city_enc", "state_enc",
    "condition_score", "condition_score_missing",
    "age_months", "age_months_missing",
    "gear_count", "gear_count_missing",
    "is_elite_seller", "image_count",
]


def encode_input(data):
    """Convert raw user input into the model's expected feature format."""
    row = {}

    for col in ["brand", "city", "state"]:
        value = data.get(col, "Unknown")
        le = encoders[col]
        if value in le.classes_:
            row[col + "_enc"] = le.transform([value])[0]
        else:
            row[col + "_enc"] = le.transform(["__UNSEEN__"])[0]

    row["condition_score"] = data.get("condition_score", 0)
    row["condition_score_missing"] = 1 if "condition_score" not in data else 0

    row["age_months"] = data.get("age_months", 0)
    row["age_months_missing"] = 1 if "age_months" not in data else 0

    row["gear_count"] = data.get("gear_count", 0)
    row["gear_count_missing"] = 1 if "gear_count" not in data else 0

    row["is_elite_seller"] = int(data.get("is_elite_seller", False))
    row["image_count"] = data.get("image_count", 0)

    return pd.DataFrame([row])[FEATURE_COLS]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "brand" not in data:
            return jsonify({"error": "brand is required"}), 400

        X = encode_input(data)

        pred_low = float(models["low"].predict(X)[0])
        pred_median = float(models["median"].predict(X)[0])
        pred_high = float(models["high"].predict(X)[0])

        pred_low = min(pred_low, pred_median)
        pred_high = max(pred_high, pred_median)

        return jsonify({
            "predicted_price": round(pred_median, 2),
            "price_range_low": round(pred_low, 2),
            "price_range_high": round(pred_high, 2),
            "confidence": "80%",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/comparables", methods=["POST"])
def comparables():
    try:
        data = request.get_json()
        brand = data.get("brand", "")
        city = data.get("city", "")

        matches = df[df["brand"] == brand]
        if city:
            city_matches = matches[matches["city"] == city]
            if len(city_matches) >= 3:
                matches = city_matches

        matches = matches.head(5)

        result = matches[["title", "price", "city", "condition_score"]].to_dict(orient="records")
        return jsonify({"comparables": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
