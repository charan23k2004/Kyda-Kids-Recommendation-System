"""
app.py
------
Flask web application for the Kyda Kids Intelligent Recommendation System.

Routes:
  /                    Home page
  /sales-analysis      Seasonal sales pattern (ARIMA/SARIMA) + product recommendations
  /feedback-analysis   Upload CSV, search a Product ID, view sentiment breakdown
"""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

from models.forecasting import seasonal_forecast_by_season, recommend_similar_products
from models.sentiment_analysis import analyze_feedback_dataframe, sentiment_summary

app = Flask(__name__)
app.secret_key = "kyda-kids-secret-key"

DEFAULT_DATA_PATH = os.path.join("data", "textile_product.csv")
UPLOAD_FOLDER = os.path.join("data", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/sales-analysis", methods=["GET", "POST"])
def sales_analysis():
    chart_b64 = None
    recommendations = None
    target_product = None
    product_id = ""

    season_totals = seasonal_forecast_by_season(DEFAULT_DATA_PATH)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(season_totals.index, season_totals.values, marker="o", color="#ff8c42")
    ax.set_title("Seasonal Sales Pattern")
    ax.set_ylabel("Total Units Sold")
    chart_b64 = fig_to_base64(fig)

    if request.method == "POST":
        product_id = request.form.get("product_id", "").strip()
        target_product, recs = recommend_similar_products(DEFAULT_DATA_PATH, product_id)
        if target_product is not None:
            recommendations = recs.to_dict(orient="records")

    return render_template(
        "sales_analysis.html",
        chart_b64=chart_b64,
        product_id=product_id,
        recommendations=recommendations,
        target_product=target_product,
    )


@app.route("/feedback-analysis", methods=["GET", "POST"])
def feedback_analysis():
    if request.method == "POST" and "csv_file" in request.files:
        file = request.files["csv_file"]
        if file and file.filename.endswith(".csv"):
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            session["uploaded_csv"] = save_path
        return redirect(url_for("feedback_analysis"))

    csv_path = session.get("uploaded_csv", DEFAULT_DATA_PATH)
    product_id = request.args.get("product_id", "").strip()

    table_rows = None
    chart_b64 = None

    if product_id:
        df = pd.read_csv(csv_path)
        subset = df[df["Product_ID"] == product_id]
        if not subset.empty:
            analyzed = analyze_feedback_dataframe(subset)
            table_rows = analyzed.to_dict(orient="records")

            counts = sentiment_summary(analyzed)
            fig, ax = plt.subplots(figsize=(6, 4))
            counts.plot(kind="bar", ax=ax, color=["#4CAF50", "#9E9E9E", "#F44336", "#FF9800"])
            ax.set_title(f"Customer Feedback Sentiment for Product {product_id}")
            ax.set_ylabel("Count")
            chart_b64 = fig_to_base64(fig)

    return render_template(
        "feedback_analysis.html",
        product_id=product_id,
        table_rows=table_rows,
        chart_b64=chart_b64,
        using_default=("uploaded_csv" not in session),
    )


if __name__ == "__main__":
    app.run(debug=True)
