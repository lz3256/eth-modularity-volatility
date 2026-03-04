"""
Step 6: Event Study Analysis

Examine modularity behavior around volatility spike events.

ANALYSIS:
1. Identify all spike hours (spike_95 == 1)
2. Filter to non-overlapping events (min 6 hours apart)
3. For each spike at time t=0, collect modularity from t-12 to t+12
4. Average across all events
5. Plot with confidence bands

OUTPUT:
  - output/figures/event_study_modularity.png
  - output/figures/event_study_modularity_covid.png
  - output/tables/event_study_stats.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import config


def select_non_overlapping_events(spike_times, min_gap_hours=6):
    """
    From a list of spike timestamps, select non-overlapping events
    with at least min_gap_hours between them.
    """
    if len(spike_times) == 0:
        return []

    spike_times = sorted(spike_times)
    selected = [spike_times[0]]
    for t in spike_times[1:]:
        if (t - selected[-1]) >= pd.Timedelta(hours=min_gap_hours):
            selected.append(t)
    return selected


def compute_event_study_windows(panel, spike_col="spike_95",
                                modularity_col="modularity_value",
                                before=12, after=12, min_gap=6):
    """
    Compute event study windows.

    Returns
    -------
    event_df : DataFrame with columns [event_id, relative_hour, modularity, ...]
    """
    # Get spike hours
    valid = panel.dropna(subset=[spike_col, modularity_col])
    spike_mask = valid[spike_col] == 1
    spike_times = valid.loc[spike_mask, "hour_utc"].tolist()

    # Filter non-overlapping
    events = select_non_overlapping_events(spike_times, min_gap_hours=min_gap)
    print(f"  Total spikes: {len(spike_times)}, Non-overlapping events: {len(events)}")

    if len(events) == 0:
        return pd.DataFrame()

    # Build event windows
    # Use the full panel (including non-trading hours) for modularity
    panel_sorted = panel.sort_values("hour_utc").set_index("hour_utc")

    all_windows = []
    for eid, event_time in enumerate(events):
        window_start = event_time - pd.Timedelta(hours=before)
        window_end = event_time + pd.Timedelta(hours=after)

        window = panel_sorted.loc[window_start:window_end].copy()
        window["event_id"] = eid
        window["event_time"] = event_time
        window["relative_hour"] = (
            (window.index - event_time).total_seconds() / 3600
        ).round().astype(int)

        all_windows.append(window.reset_index())

    event_df = pd.concat(all_windows, ignore_index=True)
    return event_df


def plot_event_study(event_df, modularity_col="modularity_value",
                     title="Event Study: Modularity Around Volatility Spikes",
                     output_path=None):
    """
    Plot average modularity trajectory around spike events with CI.
    """
    if event_df.empty:
        print("  No events to plot.")
        return

    # Compute mean and CI by relative hour
    grouped = event_df.groupby("relative_hour")[modularity_col]
    means = grouped.mean()
    sems = grouped.sem()
    counts = grouped.count()

    # 95% confidence interval
    ci_95 = 1.96 * sems

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(means.index, means.values, "b-o", markersize=4, linewidth=1.5,
            label="Mean Modularity")
    ax.fill_between(means.index, means - ci_95, means + ci_95,
                    alpha=0.2, color="blue", label="95% CI")

    # Mark t=0
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Volatility Spike (t=0)")

    # Mark pre-event region
    ax.axvspan(-12, 0, alpha=0.05, color="red")

    ax.set_xlabel("Hours Relative to Spike", fontsize=12)
    ax.set_ylabel("Network Modularity (Q)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    n_events = event_df["event_id"].nunique()
    ax.text(0.02, 0.98, f"N = {n_events} events",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure: {output_path}")
    plt.close()


def compute_event_study_statistics(event_df, modularity_col="modularity_value"):
    """
    Formal statistical tests for the event study.
    - Compare pre-event mean (t=-12 to t=-1) vs baseline (t=-12 to t=-7)
    - Test if modularity at t=-1 is significantly different from t=-6
    """
    if event_df.empty:
        return {}

    results = {}

    # Pre-event: t=-6 to t=-1
    pre = event_df[event_df["relative_hour"].between(-6, -1)][modularity_col].dropna()
    # Baseline: t=-12 to t=-7
    baseline = event_df[event_df["relative_hour"].between(-12, -7)][modularity_col].dropna()
    # Post: t=1 to t=6
    post = event_df[event_df["relative_hour"].between(1, 6)][modularity_col].dropna()
    # At spike: t=0
    at_spike = event_df[event_df["relative_hour"] == 0][modularity_col].dropna()

    results["baseline_mean"] = baseline.mean()
    results["pre_event_mean"] = pre.mean()
    results["at_spike_mean"] = at_spike.mean()
    results["post_event_mean"] = post.mean()

    # t-test: pre-event vs baseline
    if len(pre) > 5 and len(baseline) > 5:
        t_stat, p_val = stats.ttest_ind(pre, baseline)
        results["pre_vs_baseline_tstat"] = t_stat
        results["pre_vs_baseline_pval"] = p_val

    # t-test: pre-event vs post-event
    if len(pre) > 5 and len(post) > 5:
        t_stat, p_val = stats.ttest_ind(pre, post)
        results["pre_vs_post_tstat"] = t_stat
        results["pre_vs_post_pval"] = p_val

    return results


def plot_modularity_timeseries(panel, spike_col="spike_95",
                                modularity_col="modularity_value",
                                output_path=None):
    """
    Plot full timeseries of modularity with spike events marked.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: Modularity timeseries
    ax1.plot(panel["hour_utc"], panel[modularity_col],
             linewidth=0.5, alpha=0.7, color="steelblue")
    # Add 24h rolling mean
    rolling_q = panel[modularity_col].rolling(24, min_periods=12).mean()
    ax1.plot(panel["hour_utc"], rolling_q,
             linewidth=1.5, color="darkblue", label="24h Rolling Mean")

    ax1.set_ylabel("Modularity (Q)", fontsize=12)
    ax1.set_title("Ethereum Network Modularity and SPY Volatility Spikes", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom: spike indicator
    spikes = panel.dropna(subset=[spike_col])
    spike_hours = spikes[spikes[spike_col] == 1]["hour_utc"]
    ax2.vlines(spike_hours, 0, 1, color="red", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("Spike", fontsize=12)
    ax2.set_xlabel("Date (UTC)", fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure: {output_path}")
    plt.close()


def main():
    # ---- 2024 Main Sample ----
    panel_path = config.PROCESSED_DIR / "panel_2024.csv"
    if panel_path.exists():
        print("\n  Processing 2024 main sample...")
        panel = pd.read_csv(panel_path, parse_dates=["hour_utc"])

        # Full timeseries plot
        plot_modularity_timeseries(
            panel, output_path=config.FIGURES_DIR / "timeseries_modularity_2024.png"
        )

        # Event study
        event_df = compute_event_study_windows(
            panel, before=config.EVENT_WINDOW_BEFORE, after=config.EVENT_WINDOW_AFTER,
            min_gap=config.MIN_EVENTS_BETWEEN
        )

        plot_event_study(
            event_df,
            title="Event Study: ETH Modularity Around SPY Volatility Spikes (2024)",
            output_path=config.FIGURES_DIR / "event_study_modularity_2024.png"
        )

        # Statistics
        es_stats = compute_event_study_statistics(event_df)
        print("  Event Study Statistics:")
        for k, v in es_stats.items():
            print(f"    {k}: {v:.4f}")

        pd.DataFrame([es_stats]).to_csv(
            config.TABLES_DIR / "event_study_stats_2024.csv", index=False
        )

        # Robustness: count-weighted modularity
        event_df_count = compute_event_study_windows(
            panel, modularity_col="modularity_count",
            before=config.EVENT_WINDOW_BEFORE, after=config.EVENT_WINDOW_AFTER,
            min_gap=config.MIN_EVENTS_BETWEEN
        )
        plot_event_study(
            event_df_count, modularity_col="modularity_count",
            title="Event Study: ETH Modularity (Count-Weighted) Around SPY Spikes (2024)",
            output_path=config.FIGURES_DIR / "event_study_modularity_count_2024.png"
        )

    else:
        print(f"  [SKIP] {panel_path} not found.")

    # ---- COVID Case Study ----
    covid_path = config.PROCESSED_DIR / "panel_covid.csv"
    if covid_path.exists():
        print("\n  Processing COVID case study...")
        panel_covid = pd.read_csv(covid_path, parse_dates=["hour_utc"])

        plot_modularity_timeseries(
            panel_covid,
            output_path=config.FIGURES_DIR / "timeseries_modularity_covid.png"
        )

        event_df_covid = compute_event_study_windows(
            panel_covid,
            before=config.EVENT_WINDOW_BEFORE, after=config.EVENT_WINDOW_AFTER,
            min_gap=config.MIN_EVENTS_BETWEEN
        )
        plot_event_study(
            event_df_covid,
            title="Event Study: ETH Modularity Around SPY Spikes (COVID 2020)",
            output_path=config.FIGURES_DIR / "event_study_modularity_covid.png"
        )
    else:
        print(f"  [SKIP] {covid_path} not found.")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 6: Event Study Analysis")
    print("=" * 60)
    main()
    print("\nDone.")
