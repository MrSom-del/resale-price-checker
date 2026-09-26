import pandas as pd
import numpy as np

df = pd.read_csv("listings.csv")

print(f"Starting with {len(df)} listings")

# ---------- 1. Filter out shop/dealer listings ----------
df = df[df["is_shop_listing"] == False].copy()
print(f"After removing shop listings: {len(df)}")

# ---------- 2. Normalize text categoricals ----------
df["brand"] = df["brand"].str.strip().str.title()
df["city"] = df["city"].str.strip().str.title()
df["state"] = df["state"].str.strip().str.title()

# ---------- 3. Handle missing values with indicator flags ----------

# condition_score: missing -> 0 (unknown), add flag
df["condition_score_missing"] = df["condition_score"].isna().astype(int)
df["condition_score"] = df["condition_score"].fillna(0)

# age_months: missing -> use brand-level median, add flag
df["age_months_missing"] = df["age_months"].isna().astype(int)
brand_median_age = df.groupby("brand")["age_months"].transform("median")
df["age_months"] = df["age_months"].fillna(brand_median_age)
df["age_months"] = df["age_months"].fillna(df["age_months"].median())  # fallback if whole brand group is empty

# gear_count: missing -> 0 (treat as "not specified / likely single-speed"), add flag
df["gear_count_missing"] = df["gear_count"].isna().astype(int)
df["gear_count"] = df["gear_count"].fillna(0)

# area: small amount missing, fill with "Unknown"
df["area"] = df["area"].fillna("Unknown")

# ---------- 4. Basic sanity filtering on price ----------
# Remove obvious junk/outlier prices (e.g., listings priced at 1 or absurdly high)
before = len(df)
df = df[(df["price"] > 100) & (df["price"] < 50000)]
print(f"After price sanity filter: {len(df)} (removed {before - len(df)})")

# ---------- 5. Save cleaned dataset ----------
df.to_csv("listings_cleaned.csv", index=False)
print(f"\nFinal cleaned dataset: {len(df)} listings")
print("Saved to listings_cleaned.csv")

print("\n--- Final feature summary ---")
print(df[["brand", "condition_score", "condition_score_missing",
          "age_months", "age_months_missing",
          "gear_count", "gear_count_missing",
          "price", "city", "is_elite_seller", "image_count"]].describe(include="all").T)