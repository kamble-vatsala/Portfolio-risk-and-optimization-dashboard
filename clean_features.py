import pandas as pd
import sqlite3

conn = sqlite3.connect("data/portfolio.db")

# Read everything back, sorted properly
df = pd.read_sql("SELECT * FROM daily_prices ORDER BY Ticker, Date", conn)

# Make sure Date is an actual datetime, not text
df["Date"] = pd.to_datetime(df["Date"])

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Compute daily return per ticker (% change in Close price)
df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

# 20-day (roughly 1 month) rolling volatility of returns, per ticker
df["Rolling_Volatility_20d"] = (
    df.groupby("Ticker")["Daily_Return"]
      .rolling(window=20)
      .std()
      .reset_index(level=0, drop=True)
)

# 50-day and 200-day moving averages of Close price, per ticker
df["MA_50"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())
df["MA_200"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(200).mean())

# Save the enriched table back into SQLite as a new table (keeps raw data untouched)
df.to_sql("features", conn, if_exists="replace", index=False)

conn.close()
print("\nDone! Enriched data saved to 'features' table in portfolio.db")
print(df.tail())
