"""
generate_data.py
-----------------
Generates a synthetic sales + feedback dataset for Kyda Kids that matches
the schema described in the project report (Fig 4.2 Dataset Sample).

Run once to (re)create data/textile_product.csv (10,000+ rows).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_ROWS = 10000

PRODUCT_TYPES = ["T-Shirt", "Jacket", "Hoodie", "Sweatpants", "Leggings", "Shorts", "Dress"]
MATERIALS = ["Cotton", "Fleece", "Polyester", "Wool", "Spandex"]
COLORS = ["Red", "Blue", "Green", "White", "Black", "Pink", "Yellow", "Grey"]
SEASONS = ["Winter", "Spring", "Summer", "Fall"]
GENDERS = ["Male", "Female", "Unisex"]

WINTER_MATERIALS = {"Fleece", "Wool"}
SUMMER_MATERIALS = {"Cotton", "Spandex"}

POSITIVE_PHRASES = [
    "Loved the design and comfort.",
    "The fabric is soft and comfortable.",
    "Feels breathable and light.",
    "The color is vibrant and attractive.",
    "Perfect fit and great quality.",
    "Excellent stitching and finish.",
    "My kid loves wearing this every day.",
]
NEUTRAL_PHRASES = [
    "It's okay, nothing special.",
    "Average quality for the price.",
    "Decent but could be better.",
]
NEGATIVE_PHRASES = [
    "The stitching quality could be better.",
    "The product arrived damaged.",
    "Color fades after a few washes.",
    "Size runs smaller than expected.",
    "Fabric feels rough and uncomfortable.",
    "Zipper broke after one use.",
]
MIXED_PHRASES = [
    "Great color but the stitching could be better.",
    "Comfortable fabric, however the color fades quickly.",
    "Nice design but size mismatch was an issue.",
]

FEEDBACK_REASON_MAP = {
    "stitching quality": "Stitching Quality",
    "damaged": "Product Damage",
    "color fades": "Color Fade",
    "fades": "Color Fade",
    "smaller than expected": "Size Mismatch",
    "size mismatch": "Size Mismatch",
    "rough": "Fabric Quality",
    "zipper broke": "Durability",
}


def pick_feedback():
    r = random.random()
    if r < 0.55:
        return random.choice(POSITIVE_PHRASES), "Positive"
    elif r < 0.70:
        return random.choice(NEUTRAL_PHRASES), "Neutral"
    elif r < 0.90:
        return random.choice(NEGATIVE_PHRASES), "Negative"
    else:
        return random.choice(MIXED_PHRASES), "Mixed"


def random_date():
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


rows = []
for i in range(N_ROWS):
    product_id = f"P{str(random.randint(1, 300)).zfill(4)}"
    product_type = random.choice(PRODUCT_TYPES)
    material = random.choice(MATERIALS)
    color = random.choice(COLORS)
    date = random_date()
    month = date.month
    if month in (12, 1, 2):
        season = "Winter"
    elif month in (3, 4, 5):
        season = "Spring"
    elif month in (6, 7, 8):
        season = "Summer"
    else:
        season = "Fall"

    # Seasonal demand bias: winter materials sell more in winter, etc.
    base_sales = np.random.randint(5, 350)
    if season == "Winter" and material in WINTER_MATERIALS:
        base_sales = int(base_sales * 1.4)
    if season == "Summer" and material in SUMMER_MATERIALS:
        base_sales = int(base_sales * 1.3)

    return_rate = round(np.random.uniform(0.02, 0.35), 2)
    production_data = base_sales + np.random.randint(20, 200)
    gender = random.choice(GENDERS)
    season_demand_spike = round(np.random.uniform(1.0, 1.5), 2)
    age = np.random.randint(3, 16)
    feedback_text, sentiment_hint = pick_feedback()

    rows.append({
        "Product_ID": product_id,
        "Product_Type": product_type,
        "Material": material,
        "Color": color,
        "Sales_Data": base_sales,
        "Return_Rate": return_rate,
        "Production_Data": production_data,
        "Season": season,
        "Gender": gender,
        "Season_Demand_Spike": season_demand_spike,
        "Date": date.strftime("%Y-%m-%d"),
        "Age": age,
        "Customer_Feedback": feedback_text,
    })

df = pd.DataFrame(rows)
df.to_csv("data/textile_product.csv", index=False)
print(f"Generated {len(df)} rows -> data/textile_product.csv")
