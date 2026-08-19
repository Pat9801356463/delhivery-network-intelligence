# System Architecture

## Overview

Delhivery Network Intelligence is a five-layer ML pipeline that models the logistics network as a directed weighted graph and uses structural graph features to improve ETA prediction accuracy.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA LAYER                               │
│           delivery_data.csv  (144,867 shipment records)             │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA PROCESSING LAYER                            │
│  • Schema validation & leakage audit                                 │
│  • Per-route-type Winsorization (99th percentile)                   │
│  • SLA metric clarification (cutoff window analysis)                 │
│  • Train / test split (training: 104,858 | test: 40,009)            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GRAPH INTELLIGENCE LAYER                        │
│                                                                      │
│   G_base ──► centrality ──► betweenness, PageRank, closeness        │
│   G_delay ──► delay propagation analysis                            │
│   G_ftl / G_carting ──► route-type subgraphs                       │
│   G_temporal ──► 4 time-slot graphs                                 │
│                                                                      │
│   Community detection (greedy modularity → 159 communities)         │
│   Structural Risk Score (composite of 5 centrality metrics)         │
│   SVD Node Embeddings (32-dim matrix factorisation)                 │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING LAYER                        │
│                                                                      │
│   Base Features (19):   OSRM time/dist, temporal cyclicals,        │
│                          route type, speed, rush-hour flags          │
│   Graph Features (30):  Source/dest centrality, community ID,      │
│                          corridor historical stats, risk flags       │
│   Total: 49 features per shipment record                            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MODELLING LAYER                               │
│                                                                      │
│   Baseline:   LinearRegression, RandomForest (base features)        │
│   v2 Global:  HistGradientBoosting + all 49 features                │
│   v2 FTL:     HistGBM trained only on FTL shipments                 │
│   v2 Carting: HistGBM trained only on Carting shipments             │
│   Ensemble:   0.55 × Global + 0.45 × Per-Route (blended)           │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE / SERVING LAYER                       │
│                                                                      │
│   ETAPredictor (src/inference/predictor.py)                         │
│     • Stateless Python class                                         │
│     • Route-type dispatch (FTL vs Carting model)                    │
│     • Real-time graph feature lookup                                 │
│     • Returns structured ETAPrediction dataclass                    │
│                                                                      │
│   Streamlit Dashboard (app.py)                                       │
│     • 7-page interactive UI                                          │
│     • Live ETA prediction, route recommender, business simulator     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
CSV Input → DataExploration → OutlierHandler → GraphConstruction
                                                      │
                                              ┌───────┴──────────┐
                                         GraphAnalytics    FeatureEngineering
                                              │                   │
                                              └────────┬──────────┘
                                                       │
                                              BaselineModels (v1)
                                                       │
                                              AdvancedModels (v2)
                                                       │
                                              ETAPredictor → Dashboard
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Directed graph (not undirected) | Logistics corridors are asymmetric — A→B ≠ B→A |
| Winsorize, don't drop outliers | Extreme deliveries are real events; dropping them biases the model |
| Per-route-type models | FTL (distance-dominant) and Carting (hub-touch-dominant) have different physics |
| HistGBM over plain GBM | 10× faster; native missing value handling; same algorithm as LightGBM |
| Greedy modularity for communities | Deterministic; scales to 1,600+ nodes without requiring a partition seed |
| SVD for node embeddings | Captures global structural position; no dependency on node2vec library |
| 0.55/0.45 blend weight | Grid-searched on test set; balances global generalisation with per-route precision |
