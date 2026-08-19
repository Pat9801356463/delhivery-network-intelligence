"""
Unit tests for evaluation metrics and utility functions.
Compatible with: pytest (recommended) and unittest

Run: python3 -m unittest discover tests/
"""

import numpy as np
import pandas as pd
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.metrics import (
    compute_regression_metrics,
    compute_graph_advantage,
    winsorize_by_group,
)


class TestComputeRegressionMetrics(unittest.TestCase):

    def test_perfect_predictions(self):
        y = np.array([100.0, 200.0, 300.0, 400.0])
        result = compute_regression_metrics(y, y)
        self.assertAlmostEqual(result["mae"],  0.0, places=4)
        self.assertAlmostEqual(result["rmse"], 0.0, places=4)
        self.assertAlmostEqual(result["r2"],   1.0, places=3)
        self.assertAlmostEqual(result["within_15pct"], 100.0, places=1)

    def test_known_mae(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        result = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["mae"], 10.0, places=2)

    def test_known_rmse(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        result = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["rmse"], np.sqrt(12.5), places=2)

    def test_within_15pct_all_within(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 220.0, 330.0])  # all 10% off
        result = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["within_15pct"], 100.0, places=1)

    def test_output_has_all_keys(self):
        y = np.random.uniform(50, 500, 100)
        result = compute_regression_metrics(y, y * 1.1)
        for key in ["mae", "rmse", "mape", "r2", "within_15pct", "within_20pct"]:
            self.assertIn(key, result)

    def test_label_stored_in_result(self):
        y = np.array([100.0, 200.0])
        result = compute_regression_metrics(y, y, label="test_model")
        self.assertEqual(result["model"], "test_model")


class TestComputeGraphAdvantage(unittest.TestCase):

    def test_improvement_detected(self):
        baseline = {"rmse": 100.0, "mae": 50.0, "within_15pct": 50.0}
        graph    = {"rmse": 90.0,  "mae": 45.0, "within_15pct": 55.0}
        adv = compute_graph_advantage(baseline, graph)
        self.assertAlmostEqual(adv["rmse_improvement_pct"], 10.0, places=2)
        self.assertAlmostEqual(adv["mae_improvement_pct"],  10.0, places=2)
        self.assertAlmostEqual(adv["within15_improvement_pp"], 5.0, places=2)

    def test_no_improvement_gives_zero(self):
        m = {"rmse": 100.0, "mae": 50.0, "within_15pct": 50.0}
        adv = compute_graph_advantage(m, m)
        self.assertAlmostEqual(adv["rmse_improvement_pct"], 0.0, places=2)

    def test_output_contains_required_keys(self):
        m = {"rmse": 80.0, "mae": 35.0, "within_15pct": 60.0}
        adv = compute_graph_advantage(m, m)
        for key in ["baseline_rmse", "graph_rmse", "rmse_improvement_pct",
                    "baseline_within15", "graph_within15", "within15_improvement_pp"]:
            self.assertIn(key, adv)


class TestWinsorizeByGroup(unittest.TestCase):

    def test_caps_extreme_outliers(self):
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 9999], "group": ["A"] * 6})
        result = winsorize_by_group(df, "value", "group", lower_pct=0.0, upper_pct=0.90)
        self.assertLess(result["value"].max(), 9999)

    def test_per_group_independent_bounds(self):
        df = pd.DataFrame({
            "value": [10, 20, 30, 9999, 1, 2, 3, 8888],
            "group": ["FTL"] * 4 + ["Carting"] * 4,
        })
        result = winsorize_by_group(df, "value", "group", upper_pct=0.75)
        self.assertLess(result[result["group"]=="FTL"]["value"].max(), 9999)
        self.assertLess(result[result["group"]=="Carting"]["value"].max(), 8888)

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"value": [1, 2, 3, 9999], "group": ["A"] * 4})
        _ = winsorize_by_group(df, "value", "group")
        self.assertEqual(df["value"].max(), 9999)

    def test_returns_dataframe(self):
        df = pd.DataFrame({"value": [1, 2, 3], "group": ["A"] * 3})
        result = winsorize_by_group(df, "value", "group")
        self.assertIsInstance(result, pd.DataFrame)


class TestGraphUtils(unittest.TestCase):

    def test_build_corridor_graph_returns_digraph(self):
        from src.utils.graph_utils import build_corridor_graph
        import networkx as nx
        df = pd.DataFrame({
            "source_center": ["A", "B", "C"],
            "destination_center": ["B", "C", "A"],
            "trip_count": [10, 20, 5],
        })
        G = build_corridor_graph(df)
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 3)

    def test_corridor_entropy_uniform(self):
        from src.utils.graph_utils import corridor_entropy
        series = pd.Series(["FTL", "Carting", "FTL", "Carting"])
        self.assertAlmostEqual(corridor_entropy(series), 1.0, places=2)

    def test_corridor_entropy_constant_is_zero(self):
        from src.utils.graph_utils import corridor_entropy
        series = pd.Series(["FTL", "FTL", "FTL"])
        self.assertAlmostEqual(corridor_entropy(series), 0.0, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
