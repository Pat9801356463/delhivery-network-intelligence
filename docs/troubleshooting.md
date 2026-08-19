# Troubleshooting

## Common Issues

---

### `FileNotFoundError: data/raw/delivery_data.csv`

**Cause:** Raw data not placed in the expected location.

**Fix:**
```bash
cp /path/to/delivery_data.csv data/raw/delivery_data.csv
```

---

### `FileNotFoundError: models/improved_models.pkl`

**Cause:** Pipeline not run before launching the dashboard or calling the predictor.

**Fix:** Run the full pipeline in order:
```bash
python3 -m src.data.data_exploration
python3 -m src.features.graph_construction
python3 -m src.models.graph_analytics
python3 -m src.features.feature_engineering
python3 -m src.training.baseline_models
python3 -m src.data.outlier_handler
python3 -m src.training.advanced_models
```

---

### `ModuleNotFoundError: No module named 'networkx'`

**Fix:**
```bash
pip install -r requirements.txt
```

If pip cannot find the packages (restricted network), install individually:
```bash
pip install networkx scikit-learn pandas numpy matplotlib seaborn streamlit
```

---

### `streamlit: command not found`

**Fix:**
```bash
pip install streamlit
python3 -m streamlit run app.py   # alternative invocation
```

---

### Dashboard shows blank charts / missing images

**Cause:** Visuals not generated yet (pipeline not run), or path mismatch.

**Fix:** Run the full pipeline first. The dashboard reads from `assets/visuals/`. Confirm:
```bash
ls assets/visuals/*.png | wc -l
# Should show 18
```

---

### Training times out or takes >10 minutes

**Cause:** Large dataset with high `n_estimators`.

**Fix:** Reduce estimators in `configs/model_config.yaml`:
```yaml
advanced:
  histgbm_global:
    max_iter: 200   # reduce from 500
```

Or use the sample dataset for development:
```python
# In src/training/advanced_models.py, temporarily use:
df = pd.read_csv("data/sample/delivery_sample_5k.csv")
```

---

### `MemoryError` during graph analytics

**Cause:** Betweenness centrality is O(VE) and can be slow on large graphs.

**Fix:** The pipeline already uses `normalized=True` which is efficient. For very large graphs, reduce the graph to top-N corridors by volume:
```python
# In src/models/graph_analytics.py
top_corridors = corridor_stats.nlargest(1000, "trip_count")
```

---

### `ValueError: route_type must be 'FTL' or 'Carting'`

**Cause:** Passing an unexpected route type string to `ETAPredictor.predict()`.

**Fix:** Ensure `route_type` is exactly `"FTL"` or `"Carting"` (case-sensitive):
```python
result = predictor.predict(..., route_type="FTL")   # correct
result = predictor.predict(..., route_type="ftl")   # will raise ValueError
```

---

### Tests fail with `ModuleNotFoundError`

**Cause:** Test runner not invoked from the repo root.

**Fix:**
```bash
# Always run from repo root
cd /path/to/delhivery-network-intelligence
python3 -m unittest discover tests/ -v
```

---

### Streamlit port already in use

**Fix:**
```bash
streamlit run app.py --server.port 8502
```

---

## Getting Help

- Open an issue: [GitHub Issues](https://github.com/YOUR_USERNAME/delhivery-network-intelligence/issues)
- Review architecture: [`docs/architecture.md`](architecture.md)
- Full API reference: [`docs/api_reference.md`](api_reference.md)
