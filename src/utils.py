"""
Shared utility functions for H2 research.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain
from tqdm import tqdm
import config


def build_hourly_graph(edge_df, weight_col="total_eth"):
    """
    Build a weighted undirected graph from an hourly edge list.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Must have columns: from_address, to_address, and the weight column.
    weight_col : str
        "total_eth" for value-weighted, "tx_count" for count-weighted.

    Returns
    -------
    G : nx.Graph (undirected, weighted)
    """
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        u, v, w = row["from_address"], row["to_address"], row[weight_col]
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def compute_modularity(G, resolution=config.LOUVAIN_RESOLUTION,
                       random_state=config.LOUVAIN_RANDOM_STATE):
    """
    Compute Louvain modularity for a weighted undirected graph.

    Returns
    -------
    Q : float
        Modularity score.
    partition : dict
        Node -> community mapping.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() < 1:
        return np.nan, {}

    partition = community_louvain.best_partition(
        G, weight="weight", resolution=resolution, random_state=random_state
    )
    Q = community_louvain.modularity(partition, G, weight="weight")
    return Q, partition


def compute_network_stats(G):
    """
    Compute network-level statistics for controls.

    Returns
    -------
    dict with keys: n_nodes, n_edges, density
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": density,
    }


def identify_spikes(returns_series, rolling_window, percentile=95):
    """
    Identify volatility spikes as hours where |r_t| exceeds
    the rolling percentile threshold.

    Parameters
    ----------
    returns_series : pd.Series with DatetimeIndex
    rolling_window : int (number of observations, not calendar time)
    percentile : float

    Returns
    -------
    pd.DataFrame with columns: return, abs_return, rolling_threshold, spike
    """
    df = pd.DataFrame({"return": returns_series})
    df["abs_return"] = df["return"].abs()
    df["rolling_threshold"] = (
        df["abs_return"]
        .rolling(window=rolling_window, min_periods=max(30, rolling_window // 2))
        .quantile(percentile / 100.0)
    )
    df["spike"] = (df["abs_return"] > df["rolling_threshold"]).astype(int)
    # NaN out the initial window where we can't compute threshold
    df.loc[df["rolling_threshold"].isna(), "spike"] = np.nan
    return df


def compute_realized_volatility(returns_series, window=24):
    """
    Compute realized volatility as sqrt(sum of squared returns) over a rolling window.
    """
    return (returns_series ** 2).rolling(window=window).sum().apply(np.sqrt)
