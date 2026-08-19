# API Reference

## `src.inference.predictor`

### `ETAPredictor`

Production inference class wrapping per-route HistGBM models with graph feature lookups.

#### `ETAPredictor.load(models_dir, reports_dir) → ETAPredictor`

Load predictor from serialised artefacts.

```python
predictor = ETAPredictor.load(
    models_dir="models/",
    reports_dir="outputs/",
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `models_dir` | `str` | Directory containing `improved_models.pkl` |
| `reports_dir` | `str` | Directory containing `node_features.csv` and `corridor_stats_enriched.csv` |

---

#### `ETAPredictor.predict(...) → ETAPrediction`

Predict delivery ETA using route-type-specific graph-enhanced model.

```python
result = predictor.predict(
    osrm_time=120,           # OSRM estimated time (minutes)
    osrm_distance=200,       # OSRM estimated distance (km)
    actual_distance=198,     # Actual distance to destination (km)
    route_type="FTL",        # "FTL" or "Carting"
    hour=10,                 # Departure hour (0–23)
    day_of_week=1,           # 0=Monday … 6=Sunday
    month=6,                 # 1–12
    source_center="IND562132AAA",
    destination_center="IND000000ACB",
    cutoff_factor=180.0,     # Scheduled cutoff window (minutes), optional
)
```

**Raises:** `ValueError` if `route_type` is not `"FTL"` or `"Carting"`.

---

### `ETAPrediction`

Dataclass returned by `ETAPredictor.predict()`.

| Field | Type | Description |
|-------|------|-------------|
| `predicted_minutes` | `float` | Predicted delivery time in minutes |
| `predicted_hours` | `float` | Predicted delivery time in hours |
| `osrm_minutes` | `float` | OSRM estimate provided as input |
| `overrun_minutes` | `float` | `predicted - osrm` (positive = delayed) |
| `route_type` | `str` | `"FTL"` or `"Carting"` |
| `model_used` | `str` | Human-readable model name |
| `corridor_sla_risk` | `str` | `"HIGH"`, `"MEDIUM"`, or `"LOW"` |
| `corridor_delay_ratio` | `float` | Historical median delay ratio for this corridor |
| `source_risk_score` | `float` | Structural risk score of source hub (0–1) |
| `destination_risk_score` | `float` | Structural risk score of destination hub (0–1) |
| `confidence_note` | `str` | Model description string |

#### `.to_dict() → Dict[str, Any]`

Serialise prediction to plain dictionary (JSON-compatible).

---

## `src.utils.metrics`

### `compute_regression_metrics(y_true, y_pred, label, tolerance_pct) → dict`

Full regression metric suite for ETA model evaluation.

```python
from src.utils.metrics import compute_regression_metrics

metrics = compute_regression_metrics(
    y_true=y_test,
    y_pred=model.predict(X_test),
    label="MyModel",
)
# Returns: mae, rmse, mape, r2, within_15pct, within_20pct
```

### `compute_graph_advantage(baseline_metrics, graph_metrics) → dict`

Quantify the lift from graph-enhanced features over a baseline.

```python
from src.utils.metrics import compute_graph_advantage

adv = compute_graph_advantage(baseline_metrics, graph_metrics)
print(f"RMSE improvement: {adv['rmse_improvement_pct']:+.1f}%")
print(f"±15% accuracy lift: {adv['within15_improvement_pp']:+.1f}pp")
```

### `winsorize_by_group(df, target_col, group_col, lower_pct, upper_pct) → DataFrame`

Per-group winsorization of a numeric column.

```python
from src.utils.metrics import winsorize_by_group

df_clean = winsorize_by_group(
    df, target_col="actual_time", group_col="route_type",
    lower_pct=0.01, upper_pct=0.99,
)
```

---

## `src.utils.graph_utils`

### `build_corridor_graph(df, source_col, dest_col, weight_col, edge_attrs) → nx.DiGraph`

Build a directed logistics graph from aggregated corridor records.

```python
from src.utils.graph_utils import build_corridor_graph

G = build_corridor_graph(
    corridor_stats,
    weight_col="trip_count",
    edge_attrs=["median_delay_ratio", "sla_breach_rate"],
)
```

### `compute_node_centrality(G) → pd.DataFrame`

Compute full centrality suite (betweenness, PageRank, closeness, HITS).

```python
from src.utils.graph_utils import compute_node_centrality

centrality = compute_node_centrality(G_base)
top_hubs = centrality.nlargest(10, "betweenness")
```

### `compute_structural_risk(node_features, weights) → pd.Series`

Composite risk score using customisable weighted combination of centrality metrics.

```python
from src.utils.graph_utils import compute_structural_risk

risk = compute_structural_risk(node_features)
node_features["structural_risk_score"] = risk
```

### `corridor_entropy(series) → float`

Shannon entropy of a categorical series (e.g. route type distribution per corridor).

---

## Configuration

All model hyperparameters live in `configs/model_config.yaml`.

```python
import yaml

with open("configs/model_config.yaml") as f:
    config = yaml.safe_load(f)

lr = config["advanced"]["histgbm_global"]["learning_rate"]  # 0.05
```

Key sections:

| Section | Purpose |
|---------|---------|
| `data` | Raw/processed paths, train/test split key |
| `outliers` | Winsorization strategy and percentiles |
| `graph` | Embedding dim, community method, risk weights |
| `features` | Target column, leakage exclusions, feature lists |
| `baseline` | RF and LR hyperparameters |
| `advanced` | HistGBM FTL/Carting/global hyperparameters, blend weights |
| `evaluation` | CV folds, primary metric, tolerance levels |
| `paths` | All output directories |
