import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

tickers = ["AAPL", "MSFT", "JPM", "JNJ", "KO", "XOM", "PG", "DIS", "CAT", "NVDA", "GOOGL"]

end_date = datetime.today()
start_date = end_date - timedelta(days=365*10)

conn = sqlite3.connect("data/portfolio.db")

for ticker in tickers:
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    # Flatten MultiIndex columns (yfinance 1.6+/1.7 returns these even for one ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Ticker"] = ticker

    df.to_sql("daily_prices", conn, if_exists="append", index=False)

conn.close()
print("Done! Data saved to data/portfolio.db")