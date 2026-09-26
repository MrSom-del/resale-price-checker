import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("listings_cleaned.csv")

# Split into train/test BEFORE computing any averages — using test data to compute the baseline would leak information
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# ---------- Baseline: predict the brand's average price from training data ----------

brand_avg_price = train_df.groupby("brand")["price"].mean()
overall_avg_price = train_df["price"].mean()  # fallback for brands not seen in training

def baseline_predict(brand):
    return brand_avg_price.get(brand, overall_avg_price)

test_df = test_df.copy()
test_df["baseline_prediction"] = test_df["brand"].apply(baseline_predict)

baseline_mae = mean_absolute_error(test_df["price"], test_df["baseline_prediction"])

print(f"\n--- Baseline Model (brand-average price) ---")
print(f"Baseline MAE: ₹{baseline_mae:.2f}")
print(f"(For reference, average price in dataset: ₹{df['price'].mean():.2f})")

# Save train/test split so the real model uses the EXACT same split for fair comparison
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
print("\nSaved train_data.csv and test_data.csv for consistent comparison with the real model")