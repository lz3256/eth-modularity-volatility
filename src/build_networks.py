"""
Step 3: Build Hourly Networks and Compute Network Metrics

INPUT:  data/raw/eth_edges_2024.csv (or eth_edges_covid.csv)
OUTPUT: data/processed/network_metrics_2024.csv

For each hour:
1. Build weighted undirected graph from edge list
2. Compute Louvain modularity Q_t
3. Compute control variables: N_t (active nodes), Density_t
4. Also compute with count-weighted edges (robustness)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd
import numpy as np
from tqdm import tqdm
import config
from src.utils import build_hourly_graph, compute_modularity, compute_network_stats


def process_period(input_path, output_path, period_name=""):
    """Process one time period: read edge list, compute hourly network metrics."""
    print(f"\nProcessing {period_name}: {input_path}")

    edges = pd.read_csv(input_path, parse_dates=["hour_utc"])
    hours = sorted(edges["hour_utc"].unique())
    print(f"  Total hours: {len(hours)}")
    print(f"  Total edge rows: {len(edges):,}")

    results = []

    for hour in tqdm(hours, desc=f"  Building networks ({period_name})"):
        hour_edges = edges[edges["hour_utc"] == hour]

        # Value-weighted graph (primary)
        G_value = build_hourly_graph(hour_edges, weight_col="total_eth")
        Q_value, _ = compute_modularity(G_value)
        stats_value = compute_network_stats(G_value)

        # Count-weighted graph (robustness)
        G_count = build_hourly_graph(hour_edges, weight_col="tx_count")
        Q_count, _ = compute_modularity(G_count)
        stats_count = compute_network_stats(G_count)

        results.append({
            "hour_utc": hour,
            "modularity_value": Q_value,
            "n_nodes": stats_value["n_nodes"],
            "n_edges_value": stats_value["n_edges"],
            "density_value": stats_value["density"],
            "modularity_count": Q_count,
            "n_edges_count": stats_count["n_edges"],
            "density_count": stats_count["density"],
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} hourly observations to {output_path}")
    return df


def main():
    main_input = config.RAW_DIR / "eth_edges_2024.csv"
    main_output = config.PROCESSED_DIR / "network_metrics_2024.csv"
    if main_input.exists():
        process_period(main_input, main_output, period_name="2024")
    else:
        print(f"  [SKIP] {main_input} not found. Run step1 first.")

    covid_input = config.RAW_DIR / "eth_edges_covid.csv"
    covid_output = config.PROCESSED_DIR / "network_metrics_covid.csv"
    if covid_input.exists():
        process_period(covid_input, covid_output, period_name="COVID")
    else:
        print(f"  [SKIP] {covid_input} not found. Run step1 first.")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Build Hourly Networks & Compute Modularity")
    print("=" * 60)
    main()
    print("\nDone.")
