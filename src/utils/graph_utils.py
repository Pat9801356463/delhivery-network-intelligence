"""
Graph construction and analysis utilities for logistics network modelling.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def build_corridor_graph(
    df: pd.DataFrame,
    source_col: str = "source_center",
    dest_col: str = "destination_center",
    weight_col: str = "trip_count",
    edge_attrs: Optional[list] = None,
) -> nx.DiGraph:
    """
    Build a directed logistics graph from shipment records.

    Args:
        df: Aggregated corridor-level DataFrame.
        source_col: Column name for source facility.
        dest_col: Column name for destination facility.
        weight_col: Column to use as primary edge weight.
        edge_attrs: Additional columns to attach as edge attributes.

    Returns:
        Directed NetworkX graph with edge attributes.
    """
    G = nx.DiGraph()
    edge_attrs = edge_attrs or []

    for _, row in df.iterrows():
        attrs = {weight_col: row[weight_col]}
        for attr in edge_attrs:
            if attr in row.index:
                attrs[attr] = row[attr]
        G.add_edge(row[source_col], row[dest_col], **attrs)

    return G


def compute_node_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute full centrality suite for all nodes in a directed graph.

    Args:
        G: Directed NetworkX graph.

    Returns:
        DataFrame indexed by node with all centrality measures.
    """
    nodes = list(G.nodes())

    centrality = pd.DataFrame({"facility": nodes})
    centrality["in_degree"]   = centrality["facility"].map(dict(G.in_degree()))
    centrality["out_degree"]  = centrality["facility"].map(dict(G.out_degree()))
    centrality["total_degree"]= centrality["facility"].map(dict(G.degree()))
    centrality["in_degree_w"] = centrality["facility"].map(dict(G.in_degree(weight="weight")))
    centrality["out_degree_w"]= centrality["facility"].map(dict(G.out_degree(weight="weight")))
    centrality["betweenness"] = centrality["facility"].map(
        nx.betweenness_centrality(G, weight="weight", normalized=True)
    )
    centrality["pagerank"]    = centrality["facility"].map(
        nx.pagerank(G, weight="weight", alpha=0.85, max_iter=500)
    )
    centrality["closeness"]   = centrality["facility"].map(
        nx.closeness_centrality(G, distance="weight")
    )
    try:
        hubs, authorities = nx.hits(G, max_iter=1000)
        centrality["hub_score"]       = centrality["facility"].map(hubs)
        centrality["authority_score"] = centrality["facility"].map(authorities)
    except Exception:
        centrality["hub_score"]       = 0.0
        centrality["authority_score"] = 0.0

    return centrality.fillna(0)


def compute_structural_risk(
    node_features: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Compute composite structural risk score for each facility.

    Weights default to the published formula:
        Risk = 0.30 × betweenness + 0.25 × SLA_breach + 0.20 × delay
             + 0.15 × throughput  + 0.10 × pagerank

    Args:
        node_features: DataFrame with normalised risk components.
        weights: Optional custom weight dict.

    Returns:
        Pandas Series of risk scores (0–1 range).
    """
    if weights is None:
        weights = {
            "norm_betweenness": 0.30,
            "norm_sla_breach":  0.25,
            "norm_delay":       0.20,
            "norm_throughput":  0.15,
            "norm_pagerank":    0.10,
        }

    def minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / (rng + 1e-10)

    df = node_features.copy()
    df["norm_betweenness"] = minmax(df["betweenness"])
    df["norm_sla_breach"]  = minmax(df.get("avg_sla_breach",  pd.Series(0, index=df.index)))
    df["norm_delay"]       = minmax(df.get("avg_delay_ratio", pd.Series(0, index=df.index)))
    df["norm_throughput"]  = minmax(df.get("total_trips",     pd.Series(0, index=df.index)))
    df["norm_pagerank"]    = minmax(df["pagerank"])

    risk = sum(w * df[col] for col, w in weights.items())
    return risk


def corridor_entropy(series: pd.Series) -> float:
    """Compute Shannon entropy of a categorical distribution."""
    counts = series.value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-10)).sum())
