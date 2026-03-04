"""
Step 7: Predictive Regression

Model:
    Spike_t^{stock} = alpha + beta * Q_{t-1} + gamma1 * log(N_t) + gamma2 * Density_t + epsilon_t

Where:
    Spike_t = 1 if |r_t| > 95th percentile rolling threshold
    Q_{t-1} = lagged modularity (value-weighted)
    N_t = number of active nodes
    Density_t = network density

Estimation:
    - Logistic regression (binary outcome)
    - OLS/Probit as robustness
    - Newey-West (HAC) standard errors for serial correlation
    - Multiple lag specifications

OUTPUT:
    - output/tables/regression_main.csv
    - output/tables/regression_robustness.csv
    - output/figures/roc_curve.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import config


def prepare_regression_sample(panel, spike_col="spike_95"):
    """
    Prepare regression sample: drop NaN spikes (non-trading hours)
    and rows without valid lagged modularity.
    """
    # Only keep trading hours where spike is defined
    sample = panel.dropna(subset=[spike_col, "modularity_value_lag1",
                                   "log_n_nodes", "density_value"]).copy()
    print(f"  Regression sample: {len(sample)} observations")
    print(f"  Spike=1: {int(sample[spike_col].sum())} ({sample[spike_col].mean()*100:.1f}%)")
    return sample


def run_logistic_regression(sample, spike_col="spike_95", lag=1,
                             modularity_col="modularity_value"):
    """
    Run logistic regression: Spike_t ~ Q_{t-lag} + log(N_t) + Density_t

    Returns statsmodels result object.
    """
    y = sample[spike_col].values
    mod_lag_col = f"{modularity_col}_lag{lag}"

    X = sample[[mod_lag_col, f"log_n_nodes_lag{lag}", f"density_value_lag{lag}"]].copy()
    X.columns = [f"Q_lag{lag}", f"logN_lag{lag}", f"Density_lag{lag}"]
    X = sm.add_constant(X)

    # Drop any remaining NaN
    mask = ~(X.isna().any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    model = Logit(y, X)
    result = model.fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": config.NEWEY_WEST_LAGS})
    return result, X, y


def run_ols_linear_probability(sample, spike_col="spike_95", lag=1,
                                modularity_col="modularity_value"):
    """
    OLS Linear Probability Model as alternative.
    Advantage: easier interpretation of coefficients.
    """
    y = sample[spike_col].values
    mod_lag_col = f"{modularity_col}_lag{lag}"

    X = sample[[mod_lag_col, f"log_n_nodes_lag{lag}", f"density_value_lag{lag}"]].copy()
    X.columns = [f"Q_lag{lag}", f"logN_lag{lag}", f"Density_lag{lag}"]
    X = sm.add_constant(X)

    mask = ~(X.isna().any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    model = sm.OLS(y, X)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": config.NEWEY_WEST_LAGS})
    return result, X, y


def run_multi_lag_regression(sample, spike_col="spike_95", max_lags=3):
    """
    Regression with multiple lags of modularity.
    Spike_t ~ Q_{t-1} + Q_{t-2} + ... + Q_{t-k} + controls
    """
    y = sample[spike_col].values

    X_cols = []
    for lag in range(1, max_lags + 1):
        X_cols.append(f"modularity_value_lag{lag}")
    # Controls: use lag 1
    X_cols.extend(["log_n_nodes_lag1", "density_value_lag1"])

    X = sample[X_cols].copy()
    X = sm.add_constant(X)

    mask = ~(X.isna().any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    model = Logit(y, X)
    result = model.fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": config.NEWEY_WEST_LAGS})
    return result, X, y


def compute_predictive_metrics(result, X, y):
    """Compute AUC-ROC and PR-AUC for the logistic regression."""
    y_pred_prob = result.predict(X)

    metrics = {}
    try:
        metrics["auc_roc"] = roc_auc_score(y, y_pred_prob)
    except ValueError:
        metrics["auc_roc"] = np.nan

    precision, recall, _ = precision_recall_curve(y, y_pred_prob)
    metrics["auc_pr"] = auc(recall, precision)
    metrics["n_obs"] = len(y)
    metrics["n_spikes"] = int(y.sum())
    metrics["pseudo_r2"] = result.prsquared

    return metrics


def plot_roc_curve(result, X, y, label="", ax=None):
    """Plot ROC curve."""
    y_pred_prob = result.predict(X)
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    auc_val = roc_auc_score(y, y_pred_prob)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
    return ax


def format_regression_table(results_dict):
    """
    Format multiple regression results into a comparison table.
    results_dict: {model_name: statsmodels_result}
    """
    rows = []
    for name, result in results_dict.items():
        for param_name in result.params.index:
            rows.append({
                "model": name,
                "variable": param_name,
                "coefficient": result.params[param_name],
                "std_error": result.bse[param_name],
                "z_stat": result.tvalues[param_name],
                "p_value": result.pvalues[param_name],
            })
    return pd.DataFrame(rows)


def main():
    panel_path = config.PROCESSED_DIR / "panel_2024.csv"
    if not panel_path.exists():
        print(f"  [SKIP] {panel_path} not found.")
        return

    panel = pd.read_csv(panel_path, parse_dates=["hour_utc"])
    sample = prepare_regression_sample(panel)

    all_results = {}
    all_metrics = {}

    # ============================================================
    # Main specification: Logit, lag 1
    # ============================================================
    print("\n  --- Main: Logit, lag=1 ---")
    result_main, X_main, y_main = run_logistic_regression(sample, lag=1)
    print(result_main.summary())
    all_results["Logit_lag1"] = result_main
    all_metrics["Logit_lag1"] = compute_predictive_metrics(result_main, X_main, y_main)

    # ============================================================
    # Alternative lags
    # ============================================================
    for lag in [2, 3, 6]:
        print(f"\n  --- Logit, lag={lag} ---")
        try:
            res, X, y = run_logistic_regression(sample, lag=lag)
            all_results[f"Logit_lag{lag}"] = res
            all_metrics[f"Logit_lag{lag}"] = compute_predictive_metrics(res, X, y)
            beta = res.params.iloc[1]  # Q coefficient
            pval = res.pvalues.iloc[1]
            print(f"  Q_lag{lag}: coef={beta:.4f}, p={pval:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    # ============================================================
    # Multi-lag specification
    # ============================================================
    print("\n  --- Multi-lag Logit (lags 1-3) ---")
    try:
        res_multi, X_multi, y_multi = run_multi_lag_regression(sample, max_lags=3)
        print(res_multi.summary())
        all_results["Logit_multilags"] = res_multi
        all_metrics["Logit_multilags"] = compute_predictive_metrics(res_multi, X_multi, y_multi)
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # OLS Linear Probability Model
    # ============================================================
    print("\n  --- OLS LPM, lag=1 ---")
    result_ols, X_ols, y_ols = run_ols_linear_probability(sample, lag=1)
    print(result_ols.summary())
    all_results["OLS_LPM_lag1"] = result_ols

    # ============================================================
    # Count-weighted modularity (robustness)
    # ============================================================
    print("\n  --- Robustness: Count-weighted modularity ---")
    try:
        res_count, X_count, y_count = run_logistic_regression(
            sample, lag=1, modularity_col="modularity_count"
        )
        all_results["Logit_count_lag1"] = res_count
        all_metrics["Logit_count_lag1"] = compute_predictive_metrics(res_count, X_count, y_count)
        print(f"  Q_count_lag1: coef={res_count.params.iloc[1]:.4f}, p={res_count.pvalues.iloc[1]:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # Save results
    # ============================================================

    # Regression table
    reg_table = format_regression_table(all_results)
    reg_table.to_csv(config.TABLES_DIR / "regression_results.csv", index=False)
    print(f"\n  Saved regression table: {config.TABLES_DIR / 'regression_results.csv'}")

    # Predictive metrics
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(config.TABLES_DIR / "predictive_metrics.csv")
    print(f"  Saved metrics: {config.TABLES_DIR / 'predictive_metrics.csv'}")
    print("\n  Predictive Metrics:")
    print(metrics_df.to_string())

    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in all_results.items():
        if "Logit" in name and name in all_metrics:
            # Re-run prediction for plotting
            try:
                if "multilags" in name:
                    plot_roc_curve(res, X_multi, y_multi, label=name, ax=ax)
                elif "count" in name:
                    plot_roc_curve(res, X_count, y_count, label=name, ax=ax)
                else:
                    lag = int(name.split("lag")[1])
                    _, X_temp, y_temp = run_logistic_regression(sample, lag=lag)
                    plot_roc_curve(res, X_temp, y_temp, label=name, ax=ax)
            except Exception:
                pass

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: Predicting SPY Volatility Spikes", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    print(f"  Saved ROC figure: {config.FIGURES_DIR / 'roc_curves.png'}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Step 7: Predictive Regression Analysis")
    print("=" * 60)
    main()
    print("\nDone.")
