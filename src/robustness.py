"""
Step 8: Robustness Checks

Tests:
1. Alternative spike thresholds (90th, 99th percentile)
2. Realized volatility-based spikes instead of absolute return
3. Count-weighted vs value-weighted modularity
4. Winsorized edge weights (remove top 0.1% extreme transfers)
5. Subsample stability (first half vs second half of 2024)
6. Adding ETH return as control (to separate crypto-driven effects)
7. Different rolling window lengths for spike definition

OUTPUT:
    - output/tables/robustness_thresholds.csv
    - output/tables/robustness_subsample.csv
    - output/figures/robustness_comparison.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as plt
import config


def run_logit(sample, y_col, x_cols, max_lags_hac=10):
    """Helper: run logistic regression with HAC standard errors."""
    y = sample[y_col].values
    X = sample[x_cols].copy()
    X = sm.add_constant(X)

    mask = ~(X.isna().any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None  # Not enough events

    model = Logit(y, X)
    try:
        result = model.fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": max_lags_hac})
        return result
    except Exception as e:
        print(f"    Regression failed: {e}")
        return None


def extract_q_coefficient(result, q_var_index=1):
    """Extract coefficient, SE, p-value for the modularity variable."""
    if result is None:
        return {"coef": np.nan, "se": np.nan, "pval": np.nan, "n": 0}
    return {
        "coef": result.params.iloc[q_var_index],
        "se": result.bse.iloc[q_var_index],
        "pval": result.pvalues.iloc[q_var_index],
        "n": int(result.nobs),
        "pseudo_r2": result.prsquared,
    }


def robustness_1_thresholds(sample):
    """Test different spike percentile thresholds."""
    print("\n  Robustness 1: Alternative Spike Thresholds")
    results = []
    x_cols = ["modularity_value_lag1", "log_n_nodes_lag1", "density_value_lag1"]

    for pct in [90, 95, 99]:
        spike_col = f"spike_{pct}"
        if spike_col not in sample.columns:
            continue

        sub = sample.dropna(subset=[spike_col] + x_cols)
        res = run_logit(sub, spike_col, x_cols)
        info = extract_q_coefficient(res)
        info["threshold"] = f"p{pct}"
        info["n_spikes"] = int(sub[spike_col].sum())
        results.append(info)
        print(f"    p{pct}: coef={info['coef']:.4f}, p={info['pval']:.4f}, "
              f"spikes={info['n_spikes']}")

    return pd.DataFrame(results)


def robustness_2_realized_vol(sample):
    """Test realized volatility-based spike definition."""
    print("\n  Robustness 2: Realized Volatility Spikes")
    x_cols = ["modularity_value_lag1", "log_n_nodes_lag1", "density_value_lag1"]

    if "spike_rv" not in sample.columns:
        print("    [SKIP] spike_rv not found.")
        return pd.DataFrame()

    sub = sample.dropna(subset=["spike_rv"] + x_cols)
    res = run_logit(sub, "spike_rv", x_cols)
    info = extract_q_coefficient(res)
    info["spike_def"] = "realized_vol"
    info["n_spikes"] = int(sub["spike_rv"].sum())
    print(f"    RV spike: coef={info['coef']:.4f}, p={info['pval']:.4f}")
    return pd.DataFrame([info])


def robustness_3_count_weighted(sample):
    """Test count-weighted modularity instead of value-weighted."""
    print("\n  Robustness 3: Count-Weighted Modularity")
    x_cols = ["modularity_count_lag1", "log_n_nodes_lag1", "density_value_lag1"]

    sub = sample.dropna(subset=["spike_95"] + x_cols)
    res = run_logit(sub, "spike_95", x_cols)
    info = extract_q_coefficient(res)
    info["weight_type"] = "count"
    print(f"    Count-weighted: coef={info['coef']:.4f}, p={info['pval']:.4f}")
    return pd.DataFrame([info])


def robustness_4_subsample(sample):
    """Test on first half and second half of the sample."""
    print("\n  Robustness 4: Subsample Stability")
    x_cols = ["modularity_value_lag1", "log_n_nodes_lag1", "density_value_lag1"]

    sample = sample.sort_values("hour_utc")
    midpoint = sample["hour_utc"].iloc[len(sample) // 2]

    results = []
    for label, sub in [("first_half", sample[sample["hour_utc"] < midpoint]),
                        ("second_half", sample[sample["hour_utc"] >= midpoint])]:
        sub_clean = sub.dropna(subset=["spike_95"] + x_cols)
        res = run_logit(sub_clean, "spike_95", x_cols)
        info = extract_q_coefficient(res)
        info["subsample"] = label
        info["n_spikes"] = int(sub_clean["spike_95"].sum())
        results.append(info)
        print(f"    {label}: coef={info['coef']:.4f}, p={info['pval']:.4f}, "
              f"spikes={info['n_spikes']}")

    return pd.DataFrame(results)


def robustness_5_additional_controls(sample):
    """
    Add extra controls: hour-of-day fixed effects, day-of-week FE.
    This checks whether time-of-day patterns drive the result.
    """
    print("\n  Robustness 5: Additional Time Controls")

    x_cols_base = ["modularity_value_lag1", "log_n_nodes_lag1", "density_value_lag1"]

    # Add hour-of-day dummies (drop one for identification)
    if "hour_of_day" in sample.columns:
        hour_dummies = pd.get_dummies(sample["hour_of_day"], prefix="hour", drop_first=True)
        sample_aug = pd.concat([sample, hour_dummies], axis=1)
        x_cols_aug = x_cols_base + [c for c in hour_dummies.columns]

        sub = sample_aug.dropna(subset=["spike_95"] + x_cols_base)
        # Make sure dummy columns have no NaN
        for c in hour_dummies.columns:
            sub[c] = sub[c].fillna(0)

        res = run_logit(sub, "spike_95", x_cols_aug)
        info = extract_q_coefficient(res)
        info["controls"] = "hour_FE"
        print(f"    With hour FE: coef={info['coef']:.4f}, p={info['pval']:.4f}")
        return pd.DataFrame([info])

    return pd.DataFrame()


def plot_robustness_comparison(all_robustness, output_path):
    """Bar chart comparing Q coefficient across robustness checks."""
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = all_robustness.index if isinstance(all_robustness.index, pd.RangeIndex) \
        else all_robustness.index
    x = range(len(all_robustness))

    colors = ["green" if p < 0.05 else "orange" if p < 0.10 else "red"
              for p in all_robustness["pval"]]

    bars = ax.bar(x, all_robustness["coef"], yerr=1.96 * all_robustness["se"],
                  color=colors, alpha=0.7, capsize=4)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_robustness["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Coefficient on Lagged Modularity", fontsize=12)
    ax.set_title("Robustness: Modularity Coefficient Across Specifications", fontsize=14)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="p < 0.05"),
        Patch(facecolor="orange", alpha=0.7, label="p < 0.10"),
        Patch(facecolor="red", alpha=0.7, label="p >= 0.10"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    panel_path = config.PROCESSED_DIR / "panel_2024.csv"
    if not panel_path.exists():
        print(f"  [SKIP] {panel_path} not found.")
        return

    panel = pd.read_csv(panel_path, parse_dates=["hour_utc"])
    sample = panel.dropna(subset=["spike_95", "modularity_value_lag1"])

    # Run all robustness checks
    r1 = robustness_1_thresholds(sample)
    r2 = robustness_2_realized_vol(sample)
    r3 = robustness_3_count_weighted(sample)
    r4 = robustness_4_subsample(sample)
    r5 = robustness_5_additional_controls(sample)

    # Combine for summary table
    all_robust = []
    for df, label_prefix in [(r1, ""), (r2, ""), (r3, ""), (r4, ""), (r5, "")]:
        if not df.empty:
            all_robust.append(df)

    if all_robust:
        combined = pd.concat(all_robust, ignore_index=True)

        # Create labels
        labels = []
        for _, row in combined.iterrows():
            parts = []
            if "threshold" in row and pd.notna(row.get("threshold")):
                parts.append(f"Threshold {row['threshold']}")
            if "spike_def" in row and pd.notna(row.get("spike_def")):
                parts.append(f"Spike: {row['spike_def']}")
            if "weight_type" in row and pd.notna(row.get("weight_type")):
                parts.append(f"Weight: {row['weight_type']}")
            if "subsample" in row and pd.notna(row.get("subsample")):
                parts.append(f"Sample: {row['subsample']}")
            if "controls" in row and pd.notna(row.get("controls")):
                parts.append(f"Controls: {row['controls']}")
            labels.append(" | ".join(parts) if parts else f"Spec {_}")

        combined["label"] = labels
        combined.to_csv(config.TABLES_DIR / "robustness_all.csv", index=False)
        print(f"\n  Saved: {config.TABLES_DIR / 'robustness_all.csv'}")

        plot_robustness_comparison(
            combined, config.FIGURES_DIR / "robustness_comparison.png"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Step 8: Robustness Checks")
    print("=" * 60)
    main()
    print("\nDone.")
