import json
import csv
import re

# ---------- Text extraction helpers ----------

def extract_age_months(text):
    """Extract approximate age in months from free text, avoiding 'suitable for age X' false positives."""
    text = text.lower()
    
    # Patterns that indicate the number refers to a RIDER's age, not the bike's age — skip these
    rider_age_signals = [
        r'for\s+\d+', r'suitable\s+for', r'age\s+of', r'kids?', r'child', r'childre',
        r'boy', r'girl', r'teen', r'person\s+is', r'\d+\s*-\s*\d+\s*year',  # ranges like "10-16 year" = rider age range
    ]
    if any(re.search(pattern, text) for pattern in rider_age_signals):
        return None  # can't reliably tell, don't guess
    
    year_match = re.search(r'(\d+)\s*year', text)
    if year_match:
        age = int(year_match.group(1))
        if age > 15:  # sanity cap: a bicycle described as ">15 years old" is almost certainly a misfire
            return None
        return age * 12
    
    month_match = re.search(r'(\d+)\s*month', text)
    if month_match:
        return int(month_match.group(1))
    
    day_match = re.search(r'(\d+)\s*day', text)
    if day_match:
        return round(int(day_match.group(1)) / 30, 1)
    
    return None


def extract_condition_score(text):
    """Rough condition score: 3=excellent/new, 2=good, 1=fair/issues, None=unknown."""
    text = text.lower()

    excellent_keywords = ["new", "excellent", "as good as new", "mint", "sparingly used"]
    good_keywords = ["good condition", "running condition", "working condition", "no replaces"]
    fair_keywords = ["scratch", "issue", "problem", "repair needed", "damage"]

    if any(k in text for k in excellent_keywords):
        return 3
    elif any(k in text for k in good_keywords):
        return 2
    elif any(k in text for k in fair_keywords):
        return 1
    else:
        return None


def extract_gear_count(text):
    """Extract number of gears/speeds if mentioned."""
    text = text.lower()
    gear_match = re.search(r'(\d+)\s*(speed|gear)', text)
    if gear_match:
        return int(gear_match.group(1))
    return None


def is_likely_shop_listing(title, description):
    """Flag bulk/dealer listings that aren't a single real item — filter these out later."""
    combined = (title + " " + description).lower()
    shop_signals = ["showroom", "all age group", "wholesale", "bulk", "onwards", "kids :", "adult :"]
    return any(signal in combined for signal in shop_signals)


# ---------- Load merged pagination data ----------

with open("all_listings_raw.json", "r", encoding="utf-8") as f:
    elements = json.load(f)  # flat dict: listing_id -> listing_data

print(f"Loaded {len(elements)} raw listings")

# ---------- Extract each listing into a clean row ----------

rows = []
skipped = 0

for listing_id, item in elements.items():
    try:
        title = item.get("title", "") or ""
        description = item.get("description", "") or ""
        combined_text = f"{title} {description}"

        # Brand from structured parameters
        brand = ""
        for p in item.get("parameters", []) or []:
            if p.get("key") == "make":
                brand = p.get("formatted_value", p.get("value_name", ""))

        # Location
        loc = item.get("locations_resolved") or {}

        # Elite seller flag
        is_elite = any(
            f.get("type") == "ELITE_SELLER"
            for f in (item.get("user", {}) or {}).get("features", []) or []
        )

        row = {
            "id": listing_id,
            "title": title,
            "description": description,
            "price": (item.get("price", {}) or {}).get("value", {}).get("raw", ""),
            "created_at": item.get("created_at", ""),
            "user_type": item.get("user_type", ""),
            "is_elite_seller": is_elite,
            "fair_price_flag": item.get("fair_price", ""),
            "state": loc.get("ADMIN_LEVEL_1_name", ""),
            "city": loc.get("ADMIN_LEVEL_3_name", ""),
            "area": loc.get("SUBLOCALITY_LEVEL_1_name", ""),
            "image_count": len(item.get("images", []) or []),
            "brand": brand,
            "age_months": extract_age_months(combined_text),
            "condition_score": extract_condition_score(combined_text),
            "gear_count": extract_gear_count(combined_text),
            "is_shop_listing": is_likely_shop_listing(title, description),
        }
        rows.append(row)

    except Exception as e:
        skipped += 1
        print(f"Skipped listing {listing_id} due to error: {e}")

print(f"\nExtracted {len(rows)} listings successfully ({skipped} skipped due to errors)")

# ---------- Save to CSV ----------

with open("listings.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved to listings.csv")

# ---------- Quick data quality summary ----------

total = len(rows)
missing_age = sum(1 for r in rows if r["age_months"] is None)
missing_condition = sum(1 for r in rows if r["condition_score"] is None)
missing_gear = sum(1 for r in rows if r["gear_count"] is None)
shop_listings = sum(1 for r in rows if r["is_shop_listing"])
missing_price = sum(1 for r in rows if r["price"] == "")

print("\n--- Data Quality Summary ---")
print(f"Total listings: {total}")
print(f"Missing price: {missing_price} ({missing_price/total*100:.1f}%)")
print(f"Missing age_months: {missing_age} ({missing_age/total*100:.1f}%)")
print(f"Missing condition_score: {missing_condition} ({missing_condition/total*100:.1f}%)")
print(f"Missing gear_count: {missing_gear} ({missing_gear/total*100:.1f}%)")
print(f"Flagged as shop listings: {shop_listings} ({shop_listings/total*100:.1f}%)")