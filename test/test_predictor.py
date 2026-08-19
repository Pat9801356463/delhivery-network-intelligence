"""
Unit tests for the ETA prediction inference engine.
Uses mock models — no trained artefacts required.

Run: python3 -m unittest discover tests/
"""

import numpy as np
import pandas as pd
import unittest
from unittest.mock import MagicMock
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference.predictor import ETAPredictor, ETAPrediction


def _make_mock_predictor() -> ETAPredictor:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([240.0])

    node_features = pd.DataFrame({
        "facility": ["SRC001", "DST001"],
        "betweenness": [0.05, 0.02], "pagerank": [0.01, 0.008],
        "closeness": [0.3, 0.25], "structural_risk_score": [0.4, 0.2],
        "in_degree": [5, 8], "out_degree": [6, 4],
        "avg_sla_breach": [0.75, 0.60], "avg_delay_ratio": [1.9, 1.7],
        "community_id": [1, 2], "hub_score": [0.03, 0.01],
        "total_trips": [500, 300],
    })
    corridor_stats = pd.DataFrame({
        "source_center": ["SRC001"], "destination_center": ["DST001"],
        "median_delay_ratio": [1.85], "sla_breach_rate": [0.78],
        "trip_count": [45], "ftl_share": [0.65],
        "route_type_entropy": [0.8], "cross_community": [1], "corridor_risk": [0.35],
    })
    all_features = [
        "route_type_enc","log_osrm_time","log_osrm_dist","log_distance",
        "osrm_speed","dist_time_ratio","hour","dow","month","is_weekend",
        "is_rush_hour","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
        "network_load_norm","cutoff_factor",
        "src_betweenness","src_pagerank","src_closeness","src_structural_risk_score",
        "src_in_degree","src_out_degree","src_avg_sla_breach","src_avg_delay_ratio",
        "src_community_id","src_hub_score",
        "dst_betweenness","dst_pagerank","dst_closeness","dst_structural_risk_score",
        "dst_in_degree","dst_out_degree","dst_avg_sla_breach","dst_avg_delay_ratio","dst_community_id",
        "corr_median_delay_ratio","corr_sla_breach_rate","corr_trip_count","corr_ftl_share",
        "corr_route_type_entropy","corr_cross_community","corr_corridor_risk",
        "cross_community_flag","high_risk_src","high_risk_dst","both_high_risk",
    ]
    models = {"hgbm_ftl": mock_model, "hgbm_cart": mock_model, "ALL_FEATURES": all_features}
    return ETAPredictor(models=models, node_features=node_features,
                        corridor_stats=corridor_stats, all_features=all_features)


class TestETAPredictor(unittest.TestCase):

    def _predict(self, route_type="FTL", src="SRC001", dst="DST001"):
        predictor = _make_mock_predictor()
        return predictor, predictor.predict(
            osrm_time=120, osrm_distance=200, actual_distance=198,
            route_type=route_type, hour=10, day_of_week=1, month=6,
            source_center=src, destination_center=dst,
        )

    def test_returns_eta_prediction_type(self):
        _, result = self._predict()
        self.assertIsInstance(result, ETAPrediction)

    def test_predicted_minutes(self):
        _, result = self._predict()
        self.assertAlmostEqual(result.predicted_minutes, 240.0, places=1)

    def test_predicted_hours_conversion(self):
        _, result = self._predict()
        self.assertAlmostEqual(result.predicted_hours, 4.0, places=2)

    def test_overrun_calculation(self):
        _, result = self._predict()
        self.assertAlmostEqual(result.overrun_minutes, 120.0, places=1)

    def test_ftl_uses_ftl_model(self):
        predictor, _ = self._predict("FTL")
        predictor.models["hgbm_ftl"].predict.assert_called()

    def test_carting_uses_carting_model(self):
        predictor = _make_mock_predictor()
        predictor.models["hgbm_ftl"].reset_mock()
        predictor.models["hgbm_cart"].reset_mock()
        predictor.predict(osrm_time=30, osrm_distance=20, actual_distance=19,
                          route_type="Carting", hour=14, day_of_week=3, month=8,
                          source_center="SRC001", destination_center="DST001")
        predictor.models["hgbm_cart"].predict.assert_called_once()

    def test_invalid_route_type_raises_value_error(self):
        predictor = _make_mock_predictor()
        with self.assertRaises(ValueError):
            predictor.predict(osrm_time=120, osrm_distance=200, actual_distance=198,
                              route_type="EXPRESS", hour=10, day_of_week=1, month=6,
                              source_center="SRC001", destination_center="DST001")

    def test_sla_risk_high_for_high_breach_corridor(self):
        _, result = self._predict()
        self.assertEqual(result.corridor_sla_risk, "HIGH")  # sla_rate=0.78 > 0.7

    def test_to_dict_contains_key_fields(self):
        _, result = self._predict()
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        for key in ["predicted_minutes", "corridor_sla_risk", "route_type"]:
            self.assertIn(key, d)

    def test_model_family_defaults_to_histgbm(self):
        predictor, result = self._predict()
        self.assertIn("HistGBM", result.model_used)

    def test_model_family_xgboost_uses_xgb_model_when_loaded(self):
        predictor = _make_mock_predictor()
        mock_xgb = MagicMock()
        mock_xgb.predict.return_value = np.array([200.0])
        predictor.models["xgb_ftl"] = mock_xgb
        predictor.models["xgb_cart"] = mock_xgb
        result = predictor.predict(
            osrm_time=120, osrm_distance=200, actual_distance=198,
            route_type="FTL", hour=10, day_of_week=1, month=6,
            source_center="SRC001", destination_center="DST001",
            model_family="xgboost",
        )
        mock_xgb.predict.assert_called_once()
        self.assertAlmostEqual(result.predicted_minutes, 200.0, places=1)
        self.assertIn("XGBoost", result.model_used)

    def test_model_family_unavailable_raises_value_error(self):
        predictor = _make_mock_predictor()
        with self.assertRaises(ValueError):
            predictor.predict(
                osrm_time=120, osrm_distance=200, actual_distance=198,
                route_type="FTL", hour=10, day_of_week=1, month=6,
                source_center="SRC001", destination_center="DST001",
                model_family="lightgbm",
            )

    def test_invalid_model_family_raises_value_error(self):
        predictor = _make_mock_predictor()
        with self.assertRaises(ValueError):
            predictor.predict(
                osrm_time=120, osrm_distance=200, actual_distance=198,
                route_type="FTL", hour=10, day_of_week=1, month=6,
                source_center="SRC001", destination_center="DST001",
                model_family="prophet",
            )

    def test_unknown_facility_does_not_raise(self):
        predictor = _make_mock_predictor()
        result = predictor.predict(
            osrm_time=60, osrm_distance=80, actual_distance=79,
            route_type="Carting", hour=9, day_of_week=0, month=1,
            source_center="UNKNOWN_HUB_XYZ", destination_center="ANOTHER_UNKNOWN",
        )
        self.assertIsInstance(result, ETAPrediction)
        self.assertGreater(result.predicted_minutes, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
