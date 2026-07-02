"""
sentiment_analysis.py
----------------------
Customer feedback sentiment classification using VADER, plus
reason-extraction for Negative / Mixed reviews (Section 5.4.2 / 5.5.3).
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Keyword -> human readable improvement area
REASON_KEYWORDS = {
    "stitching": "Stitching Quality",
    "damaged": "Product Damage",
    "damage": "Product Damage",
    "fade": "Color Fade",
    "faded": "Color Fade",
    "fades": "Color Fade",
    "small": "Size Mismatch",
    "smaller": "Size Mismatch",
    "size": "Size Mismatch",
    "rough": "Fabric Quality",
    "uncomfortable": "Fabric Quality",
    "zipper": "Durability",
    "broke": "Durability",
    "tear": "Durability",
    "torn": "Durability",
}


def classify_sentiment(text: str) -> str:
    """
    Classify review text into Positive / Neutral / Negative / Mixed
    using VADER's compound score plus pos/neg ratios.
    """
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if scores["pos"] > 0.1 and scores["neg"] > 0.1:
        return "Mixed"
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def extract_reasons(text: str):
    """Return a list of improvement areas detected via keyword matching."""
    text_lower = text.lower()
    found = {label for kw, label in REASON_KEYWORDS.items() if kw in text_lower}
    return sorted(found) if found else ["General Dissatisfaction"]


def analyze_feedback_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with a 'Customer_Feedback' column and returns
    it enriched with Sentiment and Action/Reasons columns.
    """
    df = df.copy()
    df["Sentiment"] = df["Customer_Feedback"].astype(str).apply(classify_sentiment)
    df["Reasons"] = df.apply(
        lambda row: extract_reasons(row["Customer_Feedback"])
        if row["Sentiment"] in ("Negative", "Mixed")
        else [],
        axis=1,
    )
    df["Action"] = df["Sentiment"].apply(
        lambda s: "No Action Needed" if s == "Positive" else "Review Required"
    )
    return df


def sentiment_summary(df: pd.DataFrame):
    """Counts per sentiment category, for the bar chart shown in Fig 5.5."""
    return df["Sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative", "Mixed"]
    ).fillna(0)
