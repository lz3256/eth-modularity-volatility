"""
Step 5: Merge ETH Network Metrics with SPY Spike Data

INPUT:
  - data/processed/network_metrics_2024.csv
  - data/processed/spy_spikes_2024.csv

OUTPUT:
  - data/processed/panel_2024.csv

Merging logic:
- ETH network has 24 hourly observations per day (continuous)
- SPY only has ~7 hours per trading day (9:30-16:00 ET, roughly 14:30-21:00 UTC)
- Merge on hour_utc
- Non-trading hours: SPY spike = NaN (no return to compute)
- ETH modularity is still available for all 24 hours

For regression: only use rows where spike is not NaN (trading hours only).
For event study: can use ETH modularity from non-trading hours in the lead-up window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd
import numpy as np
import config


def merge_2024():
    """Merge 2024 network metrics with SPY spikes."""
    print("  Merging 2024 data...")

    # Load network metrics (24h continuous)
    net = pd.read_csv(
        config.PROCESSED_DIR / "network_metrics_2024.csv",
        parse_dates=["hour_utc"]
    )

    # Load SPY spikes (trading hours only)
    spy = pd.read_csv(
        config.PROCESSED_DIR / "spy_spikes_2024.csv",
        parse_dates=["datetime"]
    )
    # Ensure hour_utc is parsed
    if "hour_utc" in spy.columns:
        spy["hour_utc"] = pd.to_datetime(spy["hour_utc"])
    else:
        spy["hour_utc"] = spy["datetime"].dt.floor("h")

    # Keep relevant SPY columns
    spy_cols = [
        "hour_utc", "close", "log_return", "abs_return",
        "spike_95", "spike_90", "spike_99", "spike_rv"
    ]
    spy_slim = spy[[c for c in spy_cols if c in spy.columns]].copy()

    # Merge: left join on ETH (keep all ETH hours, SPY will be NaN outside trading)
    panel = net.merge(spy_slim, on="hour_utc", how="left")

    # Add log transforms for controls
    panel["log_n_nodes"] = np.log(panel["n_nodes"].clip(lower=1))

    # Add lagged modularity columns
    panel = panel.sort_values("hour_utc").reset_index(drop=True)
    for lag in range(1, config.MAX_LAGS + 1):
        panel[f"modularity_value_lag{lag}"] = panel["modularity_value"].shift(lag)
        panel[f"modularity_count_lag{lag}"] = panel["modularity_count"].shift(lag)
        panel[f"log_n_nodes_lag{lag}"] = panel["log_n_nodes"].shift(lag)
        panel[f"density_value_lag{lag}"] = panel["density_value"].shift(lag)

    # Add time features
    panel["hour_of_day"] = panel["hour_utc"].dt.hour
    panel["day_of_week"] = panel["hour_utc"].dt.dayofweek
    panel["date"] = panel["hour_utc"].dt.date

    output_path = config.PROCESSED_DIR / "panel_2024.csv"
    panel.to_csv(output_path, index=False)
    print(f"  Saved panel: {len(panel)} rows to {output_path}")

    # Summary
    trading = panel.dropna(subset=["spike_95"])
    print(f"  Total hours: {len(panel)}")
    print(f"  Trading hours (spike defined): {len(trading)}")
    print(f"  Spike hours (95th): {int(trading['spike_95'].sum())}")

    return panel


def merge_covid():
    """
    Merge COVID period. If hourly SPY available, same logic.
    If only daily SPY, expand daily spike to all hours of that day.
    """
    print("  Merging COVID data...")

    net_path = config.PROCESSED_DIR / "network_metrics_covid.csv"
    if not net_path.exists():
        print("  [SKIP] COVID network metrics not found.")
        return None

    net = pd.read_csv(net_path, parse_dates=["hour_utc"])

    # Try hourly SPY first
    spy_hourly_path = config.PROCESSED_DIR / "spy_spikes_covid_hourly.csv"
    spy_daily_path = config.PROCESSED_DIR / "spy_spikes_covid.csv"

    if spy_hourly_path.exists():
        spy = pd.read_csv(spy_hourly_path, parse_dates=["hour_utc"])
        panel = net.merge(spy, on="hour_utc", how="left")
    elif spy_daily_path.exists():
        # Daily fallback: assign each day's spike to all hours of that day
        spy_daily = pd.read_csv(spy_daily_path, parse_dates=["datetime"], index_col="datetime")
        spy_daily["date"] = spy_daily.index.date
        net["date"] = net["hour_utc"].dt.date

        panel = net.merge(
            spy_daily[["date", "log_return", "abs_return", "spike_95"]],
            on="date", how="left"
        )
        # For daily merge, spike is constant within a day
        # This is a cruder approach but works for event study visualization
    else:
        print("  [SKIP] No COVID SPY data found.")
        return None

    panel["log_n_nodes"] = np.log(panel["n_nodes"].clip(lower=1))
    panel = panel.sort_values("hour_utc").reset_index(drop=True)

    for lag in range(1, config.MAX_LAGS + 1):
        panel[f"modularity_value_lag{lag}"] = panel["modularity_value"].shift(lag)

    output_path = config.PROCESSED_DIR / "panel_covid.csv"
    panel.to_csv(output_path, index=False)
    print(f"  Saved panel: {len(panel)} rows to {output_path}")
    return panel


def main():
    merge_2024()
    merge_covid()


if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: Merge ETH Network Metrics + SPY Spikes")
    print("=" * 60)
    main()
    print("\nDone.")
