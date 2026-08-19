# Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.9 | 3.10 or 3.11 recommended |
| pip | ≥ 22.0 | `pip install --upgrade pip` |
| RAM | ≥ 8 GB | 16 GB recommended for full dataset |
| Disk | ≥ 2 GB | For data, models, and visuals |

---

## Quick Start (3 steps)

```bash
# 1. Clone and enter
git clone https://github.com/YOUR_USERNAME/delhivery-network-intelligence.git
cd delhivery-network-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data and run pipeline
cp /path/to/delivery_data.csv data/raw/
python3 -m src.data.data_exploration
python3 -m src.features.graph_construction
python3 -m src.models.graph_analytics
python3 -m src.features.feature_engineering
python3 -m src.training.baseline_models
python3 -m src.data.outlier_handler
python3 -m src.training.advanced_models

# Launch dashboard
streamlit run app.py
```

---

## Option A — pip (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python3 -c "import sklearn, networkx; print('OK')"
```

## Option B — Conda

```bash
conda env create -f environment.yml
conda activate delhivery-ni
```

---

## Data Setup

The raw dataset is **not included** in the repository (55 MB). Place it at:

```
data/raw/delivery_data.csv
```

Expected schema (25 columns):

| Column | Type | Description |
|--------|------|-------------|
| `data` | str | `"training"` or `"test"` |
| `trip_uuid` | str | Unique trip identifier |
| `route_type` | str | `"FTL"` or `"Carting"` |
| `source_center` | str | Source facility code |
| `destination_center` | str | Destination facility code |
| `actual_time` | float | Actual delivery time (minutes) — **target** |
| `osrm_time` | float | OSRM estimated time (minutes) |
| `osrm_distance` | float | OSRM estimated distance (km) |
| `actual_distance_to_destination` | float | Actual distance (km) |
| `is_cutoff` | bool | SLA breach flag |
| `cutoff_factor` | float | Scheduled cutoff window (minutes) |
| `od_start_time` | datetime | Trip start timestamp |

A 5,000-row sample is provided at `data/sample/delivery_sample_5k.csv` for testing.

---

## Running the Pipeline

Each script is self-contained and saves outputs to `outputs/` and `models/`:

```bash
# Step 1 — Data exploration & quality report (~30s)
python3 -m src.data.data_exploration

# Step 2 — Graph construction (~60s)
python3 -m src.features.graph_construction

# Step 3 — Graph analytics: centrality, communities, risk scoring (~90s)
python3 -m src.models.graph_analytics

# Step 4 — Feature engineering: build 49-feature store (~120s)
python3 -m src.features.feature_engineering

# Step 5 — Baseline model training (~100s)
python3 -m src.training.baseline_models

# Step 6 — Outlier handling + SLA clarification (~30s)
python3 -m src.data.outlier_handler

# Step 7 — Advanced models: HistGBM + per-route + ensemble (~180s)
python3 -m src.training.advanced_models
```

---

## Running Tests

```bash
# All tests (26 tests, ~0.5s)
python3 -m unittest discover tests/ -v

# Single module
python3 -m unittest tests.test_metrics -v
python3 -m unittest tests.test_predictor -v
```

Expected output:
```
Ran 26 tests in 0.3s
OK
```

---

## Launching the Dashboard

```bash
streamlit run app.py
# Open http://localhost:8501
```

Dashboard pages:
1. 🏠 Network Overview
2. 🔴 Bottleneck Hubs
3. ⚠️ Delay Corridors
4. ⏱️ ETA Prediction
5. 🔀 Route Recommender
6. 💰 Business Impact
7. 📈 Improvement Report

---

## Using the Predictor API

```python
from src.inference.predictor import ETAPredictor

predictor = ETAPredictor.load(
    models_dir="models/",
    reports_dir="outputs/",
)

result = predictor.predict(
    osrm_time=120,
    osrm_distance=200,
    actual_distance=198,
    route_type="FTL",
    hour=10,
    day_of_week=1,
    month=6,
    source_center="IND562132AAA",
    destination_center="IND000000ACB",
)

print(f"Predicted ETA: {result.predicted_minutes:.0f} min")
print(f"SLA Risk: {result.corridor_sla_risk}")
print(result.to_dict())
```

---

## Troubleshooting

See [`docs/troubleshooting.md`](troubleshooting.md) for common issues.
