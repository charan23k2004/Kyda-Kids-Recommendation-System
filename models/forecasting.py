"""
forecasting.py
---------------
Time-series demand forecasting for Kyda Kids using ARIMA and SARIMA,
as described in Chapter 5 of the project report.

- ARIMA(3,1,3)              -> trend-based forecasting
- SARIMA(2,1,2)x(1,1,1,12)  -> seasonal forecasting
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def load_monthly_sales(csv_path="data/textile_product.csv"):
    """Load raw transaction-level data and aggregate to monthly totals."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    monthly = (
        df.set_index("Date")
        .resample("MS")["Sales_Data"]
        .sum()
        .asfreq("MS")
        .ffill()
    )
    return monthly


def check_stationarity(series):
    """Augmented Dickey-Fuller test. Returns (is_stationary, p_value)."""
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value < 0.05, p_value


def rolling_statistics(series, window=12):
    return series.rolling(window=window).mean(), series.rolling(window=window).std()


def run_arima_forecast(series, order=(3, 1, 3), steps=6):
    """Fit ARIMA(3,1,3) and forecast `steps` months ahead."""
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return fitted, forecast


def run_sarima_forecast(series, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12), steps=6):
    """Fit SARIMA(2,1,2)x(1,1,1,12) and forecast `steps` months ahead."""
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=steps)
    return fitted, forecast.predicted_mean


def seasonal_forecast_by_season(csv_path="data/textile_product.csv"):
    """
    Produces an average predicted-sales figure per season, mirroring
    Fig 5.2 (SARIMA Seasonal Forecast) in the report.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    season_avg = df.groupby("Season")["Sales_Data"].sum()
    order = ["Winter", "Spring", "Summer", "Fall"]
    season_avg = season_avg.reindex(order)
    return season_avg


def recommend_similar_products(csv_path, product_id, top_n=5):
    """
    Given a Product_ID, recommend similar products based on
    Material, Product_Type and Color (as described in Section 5.5.2).
    """
    df = pd.read_csv(csv_path)
    target_rows = df[df["Product_ID"] == product_id]
    if target_rows.empty:
        return None, pd.DataFrame()

    target = target_rows.iloc[0]
    candidates = df[df["Product_ID"] != product_id].copy()

    candidates["match_score"] = (
        (candidates["Material"] == target["Material"]).astype(int) * 2
        + (candidates["Product_Type"] == target["Product_Type"]).astype(int) * 2
        + (candidates["Color"] == target["Color"]).astype(int)
    )
    top = (
        candidates.sort_values("match_score", ascending=False)
        .drop_duplicates(subset="Product_ID")
        .head(top_n)
    )
    return target, top[["Product_ID", "Product_Type", "Material", "Color", "Sales_Data"]]
