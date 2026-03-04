"""
Step 2: Fetch SPY Hourly Price Data

DATA SOURCE: Yahoo Finance via yfinance library (free, no API key)

LIMITATIONS:
- yfinance keeps intraday data for ~730 days only
- For 2024 data: works perfectly
- For 2020 COVID data: intraday NOT available via yfinance
  → Fallback: use daily data for COVID, or use Polygon.io / WRDS

OUTPUT:
- data/raw/spy_hourly_2024.csv
- data/raw/spy_daily_covid.csv  (daily fallback for 2020)

COLUMNS: datetime, open, high, low, close, volume
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import yfinance as yf
import pandas as pd
import numpy as np
import config


def fetch_spy_hourly_2024():
    """
    Fetch SPY 1-hour bars for 2024.
    yfinance allows max 730 days of intraday data.
    We fetch in chunks to be safe.
    """
    print("Fetching SPY hourly data for 2024...")

    # yfinance max period for 1h data is 730 days
    # Fetch 2024 full year
    spy = yf.Ticker("SPY")

    # Fetch in ~60-day chunks (yfinance is more reliable this way)
    chunks = []
    starts = pd.date_range(config.MAIN_START, config.MAIN_END, freq="60D")

    for i, start in enumerate(starts):
        end = min(start + pd.Timedelta(days=60), pd.Timestamp(config.MAIN_END))
        print(f"  Chunk {i+1}: {start.date()} to {end.date()}")
        df = spy.history(start=start, end=end, interval="1h")
        if len(df) > 0:
            chunks.append(df)

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]  # remove overlapping rows
    df = df.sort_index()

    # Clean up columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "datetime"

    output_path = config.RAW_DIR / "spy_hourly_2024.csv"
    df.to_csv(output_path)
    print(f"  Saved {len(df)} hourly bars to {output_path}")
    return df


def fetch_spy_daily_covid():
    """
    Fetch SPY daily data for COVID period.
    Daily data is available for any historical period.

    NOTE: For true hourly analysis of COVID period, you need:
    - Polygon.io (free tier: 5 calls/min, historical intraday available)
    - WRDS TAQ via NYU access
    - Alpha Vantage (25 calls/day free)
    """
    print("Fetching SPY daily data for COVID period...")

    spy = yf.Ticker("SPY")
    df = spy.history(start=config.COVID_START, end=config.COVID_END, interval="1d")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "datetime"

    output_path = config.RAW_DIR / "spy_daily_covid.csv"
    df.to_csv(output_path)
    print(f"  Saved {len(df)} daily bars to {output_path}")
    return df


def fetch_spy_hourly_covid_polygon(api_key):
    """
    OPTIONAL: Fetch SPY hourly data for COVID period via Polygon.io.
    Requires a free Polygon.io API key: https://polygon.io/

    Parameters
    ----------
    api_key : str
        Your Polygon.io API key.
    """
    import requests
    import time

    print("Fetching SPY hourly data for COVID period via Polygon.io...")

    all_results = []
    start = pd.Timestamp(config.COVID_START)
    end = pd.Timestamp(config.COVID_END)

    # Polygon aggregates endpoint
    # GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    current = start
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=30), end)
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/hour/"
            f"{current.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
        )
        resp = requests.get(url)
        data = resp.json()
        if "results" in data:
            all_results.extend(data["results"])
        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(12)  # free tier: 5 calls/min

    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]

    output_path = config.RAW_DIR / "spy_hourly_covid.csv"
    df.to_csv(output_path)
    print(f"  Saved {len(df)} hourly bars to {output_path}")
    return df


def fetch_eth_price():
    """
    Fetch ETH-USD hourly price (optional control variable).
    """
    print("Fetching ETH-USD hourly data for 2024...")
    eth = yf.Ticker("ETH-USD")
    df = eth.history(start=config.MAIN_START, end=config.MAIN_END, interval="1h")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "datetime"

    output_path = config.RAW_DIR / "eth_usd_hourly_2024.csv"
    df.to_csv(output_path)
    print(f"  Saved {len(df)} hourly bars to {output_path}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Fetch SPY Price Data")
    print("=" * 60)

    # Main sample - 2024 hourly
    fetch_spy_hourly_2024()

    # COVID - daily fallback
    fetch_spy_daily_covid()

    # Optional: ETH price as control
    fetch_eth_price()

    # Optional: COVID hourly via Polygon
    # fetch_spy_hourly_covid_polygon(api_key="YOUR_POLYGON_API_KEY")

    print("\nDone.")
