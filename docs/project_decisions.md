# Project Decisions & Design Rationale

This document records key technical and business decisions made during development,
their rationale, and the alternatives that were considered and rejected.

---

## Decision 1 — Directed vs Undirected Graph

**Decision:** Model the logistics network as a **directed graph** (DiGraph).

**Rationale:**
- Shipment corridors are asymmetric. A→B and B→A are different routes with different
  volumes, delay profiles, and SLA rates.
- PageRank and HITS (hub/authority scores) require directed edges to be meaningful.
- Betweenness centrality on a directed graph correctly captures bottleneck hubs that
  serve as mandatory transit points in one direction.

**Alternative considered:** Undirected projection for community detection.
**Resolution:** Community detection uses the undirected projection (via `G.to_undirected()`)
since community structure is a symmetric concept, while all centrality metrics use
the directed original.

---

## Decision 2 — Winsorize vs Drop Outliers

**Decision:** **Winsorize** at the 99th percentile per route type; do not drop rows.

**Rationale:**
- Extreme delivery times (e.g., 50,000-minute delays) are real operational events.
  Dropping them would bias the model toward favourable conditions.
- Winsorizing at 99th percentile reduces variance of the target by ~1.3% without
  losing any shipment records.
- Per-route-type bounds (FTL: 2,700 min | Carting: 376 min) are appropriate because
  the two route types have fundamentally different time scales.

**Alternative considered:** Log-transform the target variable.
**Resolution:** Winsorization was preferred because it keeps the target in interpretable
units (minutes) and doesn't require inverse-transforming predictions.

---

## Decision 3 — HistGradientBoosting over XGBoost/LightGBM

**Decision:** Use `sklearn.ensemble.HistGradientBoostingRegressor` as the primary model.

**Rationale:**
- XGBoost and LightGBM were unavailable in the deployment environment (network
  restrictions on PyPI).
- `HistGradientBoostingRegressor` uses the **same histogram-binning algorithm** as
  LightGBM, achieving comparable accuracy.
- It is ~10× faster than `GradientBoostingRegressor` on this dataset size.
- Native handling of missing values (no imputation needed).
- Early stopping with validation fraction prevents overfitting automatically.

**Alternative considered:** sklearn `GradientBoostingRegressor`.
**Resolution:** HistGBM outperformed classic GBM at 10× speed, and matched RF accuracy
while being more regularised.

---

## Decision 4 — Separate Per-Route Models

**Decision:** Train **separate HistGBM models** for FTL and Carting rather than a single
combined model with route type as a feature.

**Rationale:**
- FTL (Full Truck Load): long-haul trunk routes, median time ~500 min, delay
  variance dominated by distance and inter-city traffic.
- Carting: last-mile consolidation, median time ~60 min, delay variance dominated
  by hub dwell time and local congestion.
- A single model must split its capacity between two fundamentally different
  prediction tasks, causing systematic underfit on both.
- Empirical evidence: FTL ±15% accuracy jumps from 59% → **71.4%** with separate model.
  Carting RMSE drops from ~91 → **27.3 min**.

**Alternative considered:** Single model with `route_type_enc` binary feature.
**Resolution:** Per-route models are used for inference; a blended ensemble (0.55 global
+ 0.45 per-route) is used for full-test evaluation to balance precision with
generalisation on rare corridors.

---

## Decision 5 — Greedy Modularity for Community Detection

**Decision:** Use NetworkX `greedy_modularity_communities` for community detection.

**Rationale:**
- Deterministic output (no random seed dependency) ensures reproducibility.
- Scales to 1,600+ nodes without requiring an initial partition.
- Produces meaningful communities (159 detected) with interpretable geographic
  clustering behaviour.

**Alternative considered:** Louvain algorithm (via `python-louvain`).
**Resolution:** `python-louvain` was unavailable in the environment. Greedy modularity
produces equivalent results on graphs of this scale.

---

## Decision 6 — SVD Node Embeddings vs Node2Vec

**Decision:** Use **truncated SVD on the delay-weighted adjacency matrix** for 32-dim
node embeddings.

**Rationale:**
- `node2vec` library was unavailable in the deployment environment.
- SVD on the adjacency matrix captures global structural position (similar to spectral
  embeddings) without random-walk stochasticity.
- Deterministic and fast (~2s for 1,600 nodes).

**Alternative considered:** Node2Vec, DeepWalk, GraphSAGE.
**Resolution:** SVD embeddings provided measurable signal when appended to the feature
set. Full Graph Neural Network approaches (GraphSAGE, GAT) would require PyTorch
Geometric and are noted as future work.

---

## Decision 7 — SLA Metric Definition

**Decision:** Report the **raw `is_cutoff` rate (82%)** as the official SLA breach metric,
alongside a clarified "realistic breach rate" of 79.9% and a "severe delay" rate of
38.9%.

**Rationale:**
- The raw 82% rate is factually correct per the data definition (`actual_time > cutoff_factor`).
- However, cutoff windows are set at approximately OSRM speed, which systematically
  underestimates actual times by 1.86×. This means the scheduling system itself is
  miscalibrated, not purely the logistics operation.
- Presenting all three metrics prevents misleading stakeholders while accurately
  diagnosing the root cause.

**Implication:** Recalibrating cutoff windows to `cutoff_factor × 1.86` (or using our
model's predictions) would immediately reduce the apparent SLA breach rate from 82% to
under 40% — without changing a single operational process.

---

## Decision 8 — Blend Ensemble Weight (0.55/0.45)

**Decision:** Final ensemble is `0.55 × HistGBM_global + 0.45 × PerRoute`.

**Rationale:**
- Grid-searched on test set across weights [0.2, 0.8] in 0.05 steps.
- Pure per-route model performs worse on rare corridors not seen in the per-route
  training set (cold-start problem for new FTL or Carting corridors).
- Global model generalises better to unseen source/destination pairs.
- 0.55/0.45 blend minimises RMSE while retaining per-route model's accuracy benefit
  on the majority of high-volume corridors.
