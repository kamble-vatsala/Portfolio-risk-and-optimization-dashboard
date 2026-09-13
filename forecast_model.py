import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

conn = sqlite3.connect("data/portfolio.db")
df = pd.read_sql("SELECT * FROM features", conn)
df["Date"] = pd.to_datetime(df["Date"])

results = []

for ticker, group in df.groupby("Ticker"):
    g = group.sort_values("Date").copy()

    # Lagged features: yesterday's info predicting today's return
    g["Lag1_Return"] = g["Daily_Return"].shift(1)
    g["Lag2_Return"] = g["Daily_Return"].shift(2)
    g["Lag5_Return"] = g["Daily_Return"].shift(5)
    g["Volatility"] = g["Rolling_Volatility_20d"].shift(1)
    g["MA_Ratio"] = (g["MA_50"] / g["MA_200"]).shift(1)  # trend signal

    g = g.dropna(subset=["Lag1_Return", "Lag2_Return", "Lag5_Return", "Volatility", "MA_Ratio", "Daily_Return"])

    features = ["Lag1_Return", "Lag2_Return", "Lag5_Return", "Volatility", "MA_Ratio"]
    X = g[features]
    y = g["Daily_Return"]

    # Time-based split: train on first 80%, test on last 20% (NEVER shuffle time series data)
    split = int(len(g) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Baseline: naive forecast = "tomorrow's return will equal today's return"
    baseline_preds = X_test["Lag1_Return"]

    from sklearn.ensemble import RandomForestRegressor

    # Random Forest — same data, no linear assumption, can capture non-linear patterns
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    results.append({
        "Ticker": ticker,
        "Baseline_RMSE": np.sqrt(mean_squared_error(y_test, baseline_preds)),
        "LinReg_RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "LinReg_R2": r2_score(y_test, preds),
        "RF_RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "RF_R2": r2_score(y_test, rf_preds)
    })

results_df = pd.DataFrame(results).sort_values("LinReg_R2", ascending=False)
results_df.to_sql("model_results", conn, if_exists="replace", index=False)
conn.close()

print(results_df.round(5))
