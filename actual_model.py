import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# ---------- Encode categorical features ----------
# Fit encoders on TRAIN only, then apply to both — avoids leakage and handles unseen categories safely

categorical_cols = ["brand", "city", "state"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = train_df[col].fillna("Unknown")
    test_df[col] = test_df[col].fillna("Unknown")

    le.fit(list(train_df[col]) + ["__UNSEEN__"])
    train_df[col + "_enc"] = le.transform(train_df[col])

    # Handle categories in test set not seen during training
    test_df[col + "_enc"] = test_df[col].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["__UNSEEN__"])[0]
    )
    encoders[col] = le

train_df["is_elite_seller"] = train_df["is_elite_seller"].astype(int)
test_df["is_elite_seller"] = test_df["is_elite_seller"].astype(int)

# ---------- Feature set ----------
feature_cols = [
    "brand_enc", "city_enc", "state_enc",
    "condition_score", "condition_score_missing",
    "age_months", "age_months_missing",
    "gear_count", "gear_count_missing",
    "is_elite_seller", "image_count",
]

X_train = train_df[feature_cols]
y_train = train_df["price"]
X_test = test_df[feature_cols]
y_test = test_df["price"]

# ---------- Train three quantile models: 10th, 50th (median), 90th percentile ----------
# This gives you a price RANGE with confidence, not just a single number

quantiles = {"low": 0.10, "median": 0.50, "high": 0.90}
models = {}

for name, q in quantiles.items():
    model = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    models[name] = model

# ---------- Predict on test set ----------
test_df["pred_low"] = models["low"].predict(X_test)
test_df["pred_median"] = models["median"].predict(X_test)
test_df["pred_high"] = models["high"].predict(X_test)

# Ensure low <= median <= high (quantile crossing can occasionally happen)
test_df["pred_low"] = np.minimum(test_df["pred_low"], test_df["pred_median"])
test_df["pred_high"] = np.maximum(test_df["pred_high"], test_df["pred_median"])

# ---------- Evaluate ----------
model_mae = mean_absolute_error(y_test, test_df["pred_median"])

# Calibration check: what % of true prices actually fall within the predicted 10-90 range?
within_range = ((y_test >= test_df["pred_low"]) & (y_test <= test_df["pred_high"])).mean() * 100

print(f"--- Real Model (Gradient Boosting, quantile regression) ---")
print(f"Model MAE (median prediction): ₹{model_mae:.2f}")
print(f"Baseline MAE (brand average):  ₹4301.32")
improvement = (4301.32 - model_mae) / 4301.32 * 100
print(f"Improvement over baseline: {improvement:.1f}%")
print(f"\nCalibration: {within_range:.1f}% of actual prices fell within the predicted 80% confidence range")
print("(Well-calibrated would be close to 80%)")

# Show a few sample predictions
print("\n--- Sample predictions ---")
sample = test_df[["title", "price", "pred_low", "pred_median", "pred_high"]].head(10)
print(sample.to_string(index=False))

test_df.to_csv("test_predictions.csv", index=False)
print("\nSaved full predictions to test_predictions.csv")

joblib.dump(models, "trained_models.pkl")
joblib.dump(encoders, "encoders.pkl")
print("\nSaved trained_models.pkl and encoders.pkl for API use")
