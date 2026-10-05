# Ethereum Modularity Method Notes

## Question and hypothesis

The project asks whether Ethereum transaction-network community structure is associated with equity volatility events. The original hypothesis expected modularity to increase; recorded figures and the original summary instead describe a decrease.

## Implemented comparison

The configuration selects 2024 and a COVID case-study period. Transfers above the minimum value form weighted undirected hourly graphs. Louvain modularity is computed with resolution 1 and random state 42. Count weights provide a second graph statistic.

SPY close-to-close log returns generate absolute-return and realized-volatility labels. Rolling quantiles include the current bar. The primary threshold uses 210 trading observations with 60 minimum observations. The graph panel contains calendar hours, so `shift(lag)` in the 2024 panel represents rows of that calendar-hour series rather than successive equity trading bars. Coverage gaps can change that interpretation.

The main Logit specification uses lagged modularity, log node count, and density. HAC standard errors use ten lags. Alternatives include linear probability models and additional modularity lags. Event studies summarize network metrics around selected events.

## Interpret results within their design

- `compute_predictive_metrics()` calls `result.predict(X)` on the fitted regression design matrix. ROC and PR statistics therefore describe in-sample fit.
- The tracked repository contains figures but no original numerical regression tables or assembled input panels. Original coefficient, event-count, and significance statements are recorded summaries requiring a complete rerun to verify.
- The COVID default path assigns a daily equity spike label to network hours within that day. These repeated labels are not independent hourly equity events.
- Graph structure can vary with participation, transfer size, address roles, and exchange activity. Modularity changes alone do not identify cross-market contagion or liquidation flows.
- Alternative specifications create multiple comparisons. Isolated p-values do not establish a robust forecasting effect.

## Reproduction checklist

Confirm input coverage, UTC conversion, filtering, graph weights, and period boundaries. Archive provider query settings and input hashes. Produce the numerical event and regression tables alongside figures. For prediction, estimate choices on training data and evaluate on a later untouched period against equity-only and constant-probability baselines.

This documentation update preserves the existing algorithms and figures. It does not query billed data sources or generate new empirical results.
