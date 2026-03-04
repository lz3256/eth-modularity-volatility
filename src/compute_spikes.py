"""
Step 4: Compute SPY Returns and Identify Volatility Spikes

INPUT:  data/raw/spy_hourly_2024.csv
OUTPUT: data/processed/spy_spikes_2024.csv

Spike definition:
- Compute hourly log returns: r_t = ln(P_t / P_{t-1})
- Rolling 95th percentile of |r_t| over past 210 trading-hour bars (~30 trading days)
- Spike_t = 1 if |r_t| > rolling 95th percentile

Robustness:
- Also compute realized volatility over 24-hour rolling window
- Also flag spikes at 90th and 99th percentile thresholds
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd
import numpy as np
import config
from src.utils import identify_spikes, compute_realized_volatility


def process_hourly(input_path, output_path):
    """Process hourly SPY data into returns + spike indicators."""
    print(f"  Loading: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["datetime"], index_col="datetime")
    df = df.sort_index()

    # Compute log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Drop first row (no return) and any NaN
    df = df.dropna(subset=["log_return"])

    # --- Primary spike definition: rolling 95th percentile ---
    df["abs_return"] = df["log_return"].abs()
    df["rolling_p95"] = (
        df["abs_return"]
        .rolling(window=config.ROLLING_WINDOW_TRADING_HOURS,
                 min_periods=60)
        .quantile(0.95)
    )
    df["spike_95"] = (df["abs_return"] > df["rolling_p95"]).astype(int)
    df.loc[df["rolling_p95"].isna(), "spike_95"] = np.nan

    # --- Robustness: 90th and 99th percentile ---
    for pct in [90, 99]:
        col_thresh = f"rolling_p{pct}"
        col_spike = f"spike_{pct}"
        df[col_thresh] = (
            df["abs_return"]
            .rolling(window=config.ROLLING_WINDOW_TRADING_HOURS,
                     min_periods=60)
            .quantile(pct / 100.0)
        )
        df[col_spike] = (df["abs_return"] > df[col_thresh]).astype(int)
        df.loc[df[col_thresh].isna(), col_spike] = np.nan

    # --- Robustness: realized volatility spike ---
    df["realized_vol_24h"] = compute_realized_volatility(
        df["log_return"], window=config.REALIZED_VOL_WINDOW
    )
    df["rv_rolling_p95"] = (
        df["realized_vol_24h"]
        .rolling(window=config.ROLLING_WINDOW_TRADING_HOURS,
                 min_periods=60)
        .quantile(0.95)
    )
    df["spike_rv"] = (df["realized_vol_24h"] > df["rv_rolling_p95"]).astype(int)
    df.loc[df["rv_rolling_p95"].isna(), "spike_rv"] = np.nan

    # Truncate datetime to hour for merging with ETH data
    df["hour_utc"] = df.index.floor("h")

    # Save
    output_cols = [
        "hour_utc", "close", "log_return", "abs_return",
        "rolling_p95", "spike_95",
        "rolling_p90", "spike_90",
        "rolling_p99", "spike_99",
        "realized_vol_24h", "rv_rolling_p95", "spike_rv",
    ]
    out = df[output_cols].copy()
    out.to_csv(output_path, index=True)
    print(f"  Saved {len(out)} rows to {output_path}")

    # Summary
    for col in ["spike_95", "spike_90", "spike_99", "spike_rv"]:
        valid = out[col].dropna()
        n_spikes = int(valid.sum())
        pct = n_spikes / len(valid) * 100
        print(f"  {col}: {n_spikes} spikes ({pct:.1f}% of valid hours)")

    return out


def process_daily_covid(input_path, output_path):
    """
    Process daily SPY data for COVID period (fallback when hourly unavailable).
    Same logic but on daily bars.
    """
    print(f"  Loading: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["datetime"], index_col="datetime")
    df = df.sort_index()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"])

    df["abs_return"] = df["log_return"].abs()
    # For daily, use 30 trading days as rolling window
    df["rolling_p95"] = (
        df["abs_return"]
        .rolling(window=30, min_periods=15)
        .quantile(0.95)
    )
    df["spike_95"] = (df["abs_return"] > df["rolling_p95"]).astype(int)
    df.loc[df["rolling_p95"].isna(), "spike_95"] = np.nan

    df.to_csv(output_path, index=True)
    print(f"  Saved {len(df)} rows to {output_path}")
    n_spikes = int(df["spike_95"].dropna().sum())
    print(f"  spike_95: {n_spikes} spike days")
    return df


def main():
    # Main sample: 2024 hourly
    hourly_path = config.RAW_DIR / "spy_hourly_2024.csv"
    if hourly_path.exists():
        process_hourly(hourly_path, config.PROCESSED_DIR / "spy_spikes_2024.csv")
    else:
        print(f"  [SKIP] {hourly_path} not found.")

    # COVID: daily fallback
    daily_path = config.RAW_DIR / "spy_daily_covid.csv"
    if daily_path.exists():
        process_daily_covid(daily_path, config.PROCESSED_DIR / "spy_spikes_covid.csv")
    else:
        print(f"  [SKIP] {daily_path} not found.")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: Compute SPY Returns & Identify Volatility Spikes")
    print("=" * 60)
    main()
    print("\nDone.")
