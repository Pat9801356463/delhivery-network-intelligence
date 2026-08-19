"""
Evaluation metrics and scoring utilities for ETA prediction models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import pandas as pd


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
    tolerance_pct: float = 0.15,
) -> Dict[str, float]:
    """
    Compute full regression metric suite for ETA prediction evaluation.

    Args:
        y_true: Ground truth delivery times (minutes).
        y_pred: Predicted delivery times (minutes).
        label: Optional model label for printing.
        tolerance_pct: Fractional tolerance for within-N% accuracy (default 0.15 = ±15%).

    Returns:
        Dictionary containing MAE, RMSE, MAPE, R², and within-tolerance accuracy.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    r2   = r2_score(y_true, y_pred)
    tol_15 = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5) < 0.15) * 100
    tol_20 = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5) < 0.20) * 100

    results = {
        "model":        label,
        "mae":          round(mae, 3),
        "rmse":         round(rmse, 3),
        "mape":         round(mape, 3),
        "r2":           round(r2, 4),
        "within_15pct": round(tol_15, 2),
        "within_20pct": round(tol_20, 2),
    }

    if label:
        print(
            f"  {label:<44} "
            f"MAE={mae:6.1f}  RMSE={rmse:7.1f}  "
            f"R²={r2:.4f}  ±15%={tol_15:.1f}%  ±20%={tol_20:.1f}%"
        )

    return results


def compute_graph_advantage(
    baseline_metrics: Dict[str, float],
    graph_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute the quantified lift from graph-enhanced features.

    Args:
        baseline_metrics: Metrics dict from baseline model (no graph features).
        graph_metrics: Metrics dict from graph-enhanced model.

    Returns:
        Dictionary of improvement percentages / percentage points.
    """
    rmse_lift = (baseline_metrics["rmse"] - graph_metrics["rmse"]) / baseline_metrics["rmse"] * 100
    mae_lift  = (baseline_metrics["mae"]  - graph_metrics["mae"])  / baseline_metrics["mae"]  * 100
    w15_lift  = graph_metrics["within_15pct"] - baseline_metrics["within_15pct"]

    return {
        "baseline_rmse":    baseline_metrics["rmse"],
        "graph_rmse":       graph_metrics["rmse"],
        "rmse_improvement_pct": round(rmse_lift, 2),
        "baseline_mae":     baseline_metrics["mae"],
        "graph_mae":        graph_metrics["mae"],
        "mae_improvement_pct":  round(mae_lift, 2),
        "baseline_within15":    baseline_metrics["within_15pct"],
        "graph_within15":       graph_metrics["within_15pct"],
        "within15_improvement_pp": round(w15_lift, 2),
    }


def winsorize_by_group(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorize a target column at per-group quantile bounds.

    Args:
        df: Input DataFrame.
        target_col: Column to winsorize.
        group_col: Column to group by before computing bounds.
        lower_pct: Lower bound quantile (default 1st percentile).
        upper_pct: Upper bound quantile (default 99th percentile).

    Returns:
        DataFrame with target_col winsorized in-place.
    """
    df = df.copy()
    df[target_col] = df[target_col].astype(float)  # ensure float for clip assignment
    for group_val in df[group_col].unique():
        mask  = df[group_col] == group_val
        lower = df.loc[mask, target_col].quantile(lower_pct)
        upper = df.loc[mask, target_col].quantile(upper_pct)
        df.loc[mask, target_col] = df.loc[mask, target_col].clip(lower, upper)
    return df
