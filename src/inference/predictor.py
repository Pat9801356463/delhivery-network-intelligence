"""
Production ETA Prediction Engine.

Provides a clean, stateless inference interface over the trained
per-route HistGradientBoosting models with graph-enhanced features.

Usage:
    from src.inference.predictor import ETAPredictor
    predictor = ETAPredictor.load("models/")
    result = predictor.predict(
        osrm_time=120, osrm_distance=150, actual_distance=148,
        route_type="FTL", hour=10, day_of_week=1, month=6,
        source_center="IND562132AAA", destination_center="IND000000ACB",
    )
"""

import numpy as np
import pandas as pd
import pickle
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ETAPrediction:
    """Result container for a single ETA prediction."""
    predicted_minutes: float
    predicted_hours: float
    osrm_minutes: float
    overrun_minutes: float
    route_type: str
    model_used: str
    corridor_sla_risk: str          # "HIGH" | "MEDIUM" | "LOW"
    corridor_delay_ratio: float
    source_risk_score: float
    destination_risk_score: float
    confidence_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class ETAPredictor:
    """
    Stateless production ETA predictor.

    Wraps the trained per-route HistGBM models and graph feature
    lookup tables into a single callable inference object.
    """

    def __init__(
        self,
        models: Dict,
        node_features: pd.DataFrame,
        corridor_stats: pd.DataFrame,
        all_features: list,
    ):
        self.models         = models
        self.node_features  = node_features.set_index("facility")
        self.corridor_stats = corridor_stats.set_index(["source_center", "destination_center"])
        self.all_features   = all_features

    @classmethod
    def load(cls, models_dir: str, reports_dir: str) -> "ETAPredictor":
        """
        Load predictor from serialised model artefacts.

        Args:
            models_dir: Path to directory containing improved_models.pkl.
            reports_dir: Path to directory containing node_features.csv
                         and corridor_stats_enriched.csv.

        Returns:
            Initialised ETAPredictor instance.
        """
        with open(os.path.join(models_dir, "improved_models.pkl"), "rb") as f:
            models = pickle.load(f)

        node_features  = pd.read_csv(os.path.join(reports_dir, "node_features.csv"))
        corridor_stats = pd.read_csv(os.path.join(reports_dir, "corridor_stats_enriched.csv"))

        return cls(
            models=models,
            node_features=node_features,
            corridor_stats=corridor_stats,
            all_features=models["ALL_FEATURES"],
        )

    def _get_node_feature(self, facility: str, col: str) -> float:
        """Lookup a node feature, falling back to the column median."""
        if facility in self.node_features.index and col in self.node_features.columns:
            return float(self.node_features.loc[facility, col])
        if col in self.node_features.columns:
            return float(self.node_features[col].median())
        return 0.0

    def _get_corridor_feature(self, source: str, dest: str, col: str) -> float:
        """Lookup a corridor-level feature, falling back to 0."""
        key = (source, dest)
        if key in self.corridor_stats.index and col in self.corridor_stats.columns:
            return float(self.corridor_stats.loc[key, col])
        return 0.0

    def _build_feature_vector(
        self,
        osrm_time: float,
        osrm_distance: float,
        actual_distance: float,
        route_type: str,
        hour: int,
        day_of_week: int,
        month: int,
        source_center: str,
        destination_center: str,
        cutoff_factor: float = 180.0,
    ) -> pd.DataFrame:
        """Construct the 49-feature input vector for inference."""
        is_weekend  = int(day_of_week in [5, 6])
        is_rush     = int(hour in [7, 8, 9, 17, 18, 19])

        feature_dict = {
            # Base features
            "route_type_enc":   int(route_type == "FTL"),
            "log_osrm_time":    np.log1p(osrm_time),
            "log_osrm_dist":    np.log1p(osrm_distance),
            "log_distance":     np.log1p(actual_distance),
            "osrm_speed":       osrm_distance / (osrm_time / 60 + 1e-5),
            "dist_time_ratio":  actual_distance / (osrm_time + 1),
            "hour":             hour,
            "dow":              day_of_week,
            "month":            month,
            "is_weekend":       is_weekend,
            "is_rush_hour":     is_rush,
            "hour_sin":  np.sin(2 * np.pi * hour / 24),
            "hour_cos":  np.cos(2 * np.pi * hour / 24),
            "dow_sin":   np.sin(2 * np.pi * day_of_week / 7),
            "dow_cos":   np.cos(2 * np.pi * day_of_week / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "network_load_norm": 0.5,
            "cutoff_factor":     cutoff_factor,
            # Source node graph features
            "src_betweenness":          self._get_node_feature(source_center, "betweenness"),
            "src_pagerank":             self._get_node_feature(source_center, "pagerank"),
            "src_closeness":            self._get_node_feature(source_center, "closeness"),
            "src_structural_risk_score":self._get_node_feature(source_center, "structural_risk_score"),
            "src_in_degree":            self._get_node_feature(source_center, "in_degree"),
            "src_out_degree":           self._get_node_feature(source_center, "out_degree"),
            "src_avg_sla_breach":       self._get_node_feature(source_center, "avg_sla_breach"),
            "src_avg_delay_ratio":      self._get_node_feature(source_center, "avg_delay_ratio"),
            "src_community_id":         self._get_node_feature(source_center, "community_id"),
            "src_hub_score":            self._get_node_feature(source_center, "hub_score"),
            # Destination node graph features
            "dst_betweenness":          self._get_node_feature(destination_center, "betweenness"),
            "dst_pagerank":             self._get_node_feature(destination_center, "pagerank"),
            "dst_closeness":            self._get_node_feature(destination_center, "closeness"),
            "dst_structural_risk_score":self._get_node_feature(destination_center, "structural_risk_score"),
            "dst_in_degree":            self._get_node_feature(destination_center, "in_degree"),
            "dst_out_degree":           self._get_node_feature(destination_center, "out_degree"),
            "dst_avg_sla_breach":       self._get_node_feature(destination_center, "avg_sla_breach"),
            "dst_avg_delay_ratio":      self._get_node_feature(destination_center, "avg_delay_ratio"),
            "dst_community_id":         self._get_node_feature(destination_center, "community_id"),
            # Corridor features
            "corr_median_delay_ratio": self._get_corridor_feature(source_center, destination_center, "median_delay_ratio") or 1.86,
            "corr_sla_breach_rate":    self._get_corridor_feature(source_center, destination_center, "sla_breach_rate")    or 0.82,
            "corr_trip_count":         self._get_corridor_feature(source_center, destination_center, "trip_count")         or 10.0,
            "corr_ftl_share":          self._get_corridor_feature(source_center, destination_center, "ftl_share")          or 0.5,
            "corr_route_type_entropy": self._get_corridor_feature(source_center, destination_center, "route_type_entropy") or 0.5,
            "corr_cross_community":    self._get_corridor_feature(source_center, destination_center, "cross_community")    or 0.0,
            "corr_corridor_risk":      self._get_corridor_feature(source_center, destination_center, "corridor_risk")      or 0.3,
            # Risk flags
            "cross_community_flag": 0,
            "high_risk_src":  int(self._get_node_feature(source_center,      "structural_risk_score") > 0.3),
            "high_risk_dst":  int(self._get_node_feature(destination_center, "structural_risk_score") > 0.3),
            "both_high_risk": int(
                self._get_node_feature(source_center, "structural_risk_score") > 0.3 and
                self._get_node_feature(destination_center, "structural_risk_score") > 0.3
            ),
        }

        row = pd.DataFrame([feature_dict])
        for col in self.all_features:
            if col not in row.columns:
                row[col] = 0.0
        return row[self.all_features].fillna(0)

    def predict(
        self,
        osrm_time: float,
        osrm_distance: float,
        actual_distance: float,
        route_type: str,
        hour: int,
        day_of_week: int,
        month: int,
        source_center: str,
        destination_center: str,
        cutoff_factor: float = 180.0,
    ) -> ETAPrediction:
        """
        Predict delivery ETA using the per-route graph-enhanced model.

        Args:
            osrm_time: OSRM-estimated travel time (minutes).
            osrm_distance: OSRM-estimated distance (km).
            actual_distance: Actual distance to destination (km).
            route_type: "FTL" or "Carting".
            hour: Departure hour (0–23).
            day_of_week: Day of week (0=Monday … 6=Sunday).
            month: Month (1–12).
            source_center: Source facility code.
            destination_center: Destination facility code.
            cutoff_factor: Scheduled cutoff window in minutes.

        Returns:
            ETAPrediction dataclass with full metadata.
        """
        if route_type not in ("FTL", "Carting"):
            raise ValueError(f"route_type must be 'FTL' or 'Carting', got '{route_type}'")

        model      = self.models["hgbm_ftl"] if route_type == "FTL" else self.models["hgbm_cart"]
        model_name = f"{route_type}-specific HistGBM"

        X = self._build_feature_vector(
            osrm_time, osrm_distance, actual_distance,
            route_type, hour, day_of_week, month,
            source_center, destination_center, cutoff_factor,
        )

        predicted = float(model.predict(X)[0])
        sla_rate  = self._get_corridor_feature(source_center, destination_center, "sla_breach_rate")
        sla_risk  = "HIGH" if sla_rate > 0.7 else "MEDIUM" if sla_rate > 0.4 else "LOW"
        delay_r   = self._get_corridor_feature(source_center, destination_center, "median_delay_ratio") or 1.86
        src_risk  = self._get_node_feature(source_center,      "structural_risk_score")
        dst_risk  = self._get_node_feature(destination_center, "structural_risk_score")

        return ETAPrediction(
            predicted_minutes=round(predicted, 1),
            predicted_hours=round(predicted / 60, 2),
            osrm_minutes=osrm_time,
            overrun_minutes=round(predicted - osrm_time, 1),
            route_type=route_type,
            model_used=model_name,
            corridor_sla_risk=sla_risk,
            corridor_delay_ratio=round(delay_r, 3),
            source_risk_score=round(src_risk, 4),
            destination_risk_score=round(dst_risk, 4),
            confidence_note="Per-route HistGBM with 49 graph-enhanced features",
        )
