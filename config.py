"""
Configuration for H2 Research: Network Modularity & Volatility Spikes
"""
import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

for d in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# TIME PERIODS
# ============================================================
# Main regression sample: full year 2024
MAIN_START = "2024-01-01"
MAIN_END = "2024-12-31"

# COVID case study (with buffer for rolling window + event study)
COVID_START = "2020-01-15"
COVID_END = "2020-04-30"

# ============================================================
# BIGQUERY
# ============================================================
GCP_PROJECT_ID = "your-gcp-project-id"  # <-- CHANGE THIS
BQ_ETH_TABLE = "bigquery-public-data.crypto_ethereum.transactions"

# ============================================================
# NETWORK CONSTRUCTION
# ============================================================
# Minimum ETH value to include a transaction (filter dust)
MIN_ETH_VALUE = 0.001  # in ETH (not Wei)
MIN_ETH_VALUE_WEI = int(MIN_ETH_VALUE * 1e18)

# ============================================================
# MODULARITY
# ============================================================
LOUVAIN_RESOLUTION = 1.0  # default resolution parameter
LOUVAIN_RANDOM_STATE = 42

# ============================================================
# VOLATILITY SPIKE DEFINITION
# ============================================================
ROLLING_WINDOW_HOURS = 720  # 30 days * 24 hours (but only trading hours count)
# For SPY (only ~7 trading hours/day), 30 trading days ≈ 210 hourly bars
ROLLING_WINDOW_TRADING_HOURS = 210
SPIKE_PERCENTILE = 95  # 95th percentile threshold

# Robustness: realized volatility window
REALIZED_VOL_WINDOW = 24  # hours (for robustness check)

# ============================================================
# EVENT STUDY
# ============================================================
EVENT_WINDOW_BEFORE = 12  # hours before spike
EVENT_WINDOW_AFTER = 12   # hours after spike
MIN_EVENTS_BETWEEN = 6    # minimum hours between spikes to avoid overlap

# ============================================================
# REGRESSION
# ============================================================
MAX_LAGS = 6  # max lagged modularity terms to consider
NEWEY_WEST_LAGS = 10  # for HAC standard errors

# ============================================================
# ROBUSTNESS
# ============================================================
WINSORIZE_PERCENTILE = 0.1  # top 0.1% of transactions by value
SPIKE_PERCENTILE_ROBUST = [90, 95, 99]  # test multiple thresholds
