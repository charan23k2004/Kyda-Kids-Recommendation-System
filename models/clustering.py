"""
clustering.py
--------------
Customer segmentation using K-Means, with an Elbow Method helper
to pick the optimal number of clusters (Section 5.4.1, Fig 5.4).
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["Sales_Data", "Return_Rate", "Production_Data", "Season_Demand_Spike", "Age"]


def elbow_inertias(df: pd.DataFrame, k_range=range(1, 10)):
    """Compute inertia for a range of k values (for the elbow plot)."""
    X = StandardScaler().fit_transform(df[FEATURES])
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    return list(k_range), inertias


def segment_customers(df: pd.DataFrame, n_clusters=4):
    """Fit K-Means with the chosen k (default 4, per the report's elbow result)."""
    df = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df["Customer_Segment"] = km.fit_predict(X)
    return df, km
