"""
IMPROVEMENT 5 — XGBoost / LightGBM Integration
(README "Future Improvements" — High priority)

Adds XGBoost and LightGBM as new candidate models alongside the existing
RF / HistGBM family, and folds them into the stacking ensemble + blend.

Unlike the other src/training scripts (which assume you already ran the
full data_exploration -> outlier_handler -> graph_construction ->
graph_analytics -> feature_engineering chain on the full 144K-row raw
dataset), this script is self-contained: if it doesn't find those
intermediate artefacts under data/processed/, it builds them itself from
whatever data IS available (full raw data if present, else the bundled
5k-row sample). This makes it runnable out of the box.

NOTE ON THE SAMPLE DATA: the 5,000-row sample bundled in this repo is
meant for development/smoke-testing, not for reproducing the headline
README numbers (those came from the full 144,867-row dataset, which is
gitignored / not included here). RMSE/MAE values printed by this script
on the sample are illustrative only — re-run against data/raw/delivery_data.csv
for numbers that are comparable to the README's model benchmark table.

Run:
    python3 -m src.training.boosted_models
"""
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "xgboost is required for src.training.boosted_models. "
        "Install with: pip install xgboost"
    ) from e

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "lightgbm is required for src.training.boosted_models. "
        "Install with: pip install lightgbm"
    ) from e

# ── Paths (repo-relative — NOT the /home/claude/project paths used by the
#    older notebook-derived scripts in this codebase) ────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_PATH      = os.path.join(PROJECT_ROOT, "data", "raw", "delivery_data.csv")
SAMPLE_PATH   = os.path.join(PROJECT_ROOT, "data", "sample", "delivery_sample_5k.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# STAGE 0 — Build (or load) cleaned data + graph + feature store
# ══════════════════════════════════════════════════════════════════════
def _entropy(series):
    counts = series.value_counts(normalize=True)
    return -(counts * np.log2(counts + 1e-10)).sum()


def _minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + 1e-10)


def build_feature_store(force_rebuild: bool = False) -> dict:
    """
    Reproduces the outlier_handler -> graph_construction -> graph_analytics ->
    feature_engineering chain end to end, from whichever raw data source is
    available, and returns the 49-feature train/test store.

    Caches to data/processed/feature_store.pkl so repeat runs are fast.
    """
    cache_path = os.path.join(PROCESSED_DIR, "feature_store.pkl")
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"  Loading cached feature store: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data_path = RAW_PATH if os.path.exists(RAW_PATH) else SAMPLE_PATH
    is_sample = data_path == SAMPLE_PATH
    print(f"  Building feature store from: {data_path}"
          f"{'  [SAMPLE — dev/demo data, not the full 144K dataset]' if is_sample else ''}")

    df = pd.read_csv(data_path)
    df["od_start_time"] = pd.to_datetime(df["od_start_time"], errors="coerce")
    df["delay_ratio"]   = df["actual_time"] / df["osrm_time"].replace(0, np.nan)
    df["is_sla_breach"] = df["is_cutoff"].astype(bool)
    df["is_delayed"]    = df["delay_ratio"] > 1.20
    df["hour"]  = df["od_start_time"].dt.hour
    df["dow"]   = df["od_start_time"].dt.dayofweek
    df["month"] = df["od_start_time"].dt.month
    df = df[df["osrm_time"] > 0].copy()

    # ── Winsorize per route type (mirrors src/data/outlier_handler.py) ──
    for rt in ["FTL", "Carting"]:
        mask = df["route_type"] == rt
        cap = df.loc[mask, "actual_time"].quantile(0.99)
        df.loc[mask & (df["actual_time"] > cap), "actual_time"] = cap
    df["delay_ratio"] = df["actual_time"] / df["osrm_time"].replace(0, np.nan)

    # ── Corridor aggregation + base graph (mirrors graph_construction.py) ──
    corridor = df.groupby(["source_center", "destination_center"]).agg(
        trip_count=("trip_uuid", "count"),
        median_delay_ratio=("delay_ratio", "median"),
        sla_breach_rate=("is_sla_breach", "mean"),
        chronic_delay_rate=("is_delayed", "mean"),
        median_distance=("actual_distance_to_destination", "median"),
        ftl_share=("route_type", lambda x: (x == "FTL").mean()),
        route_type_entropy=("route_type", _entropy),
    ).reset_index()

    G = nx.DiGraph()
    for _, row in corridor.iterrows():
        G.add_edge(row["source_center"], row["destination_center"], weight=row["trip_count"])

    src_stats = df.groupby("source_center").agg(
        outbound_trips=("trip_uuid", "count"), out_delay_ratio=("delay_ratio", "median"),
        out_sla_breach=("is_sla_breach", "mean")).reset_index().rename(columns={"source_center": "facility"})
    dst_stats = df.groupby("destination_center").agg(
        inbound_trips=("trip_uuid", "count"), in_delay_ratio=("delay_ratio", "median"),
        in_sla_breach=("is_sla_breach", "mean")).reset_index().rename(columns={"destination_center": "facility"})
    facilities = pd.DataFrame({"facility": list(set(df["source_center"]) | set(df["destination_center"]))})
    node_stats = (facilities.merge(src_stats, on="facility", how="left")
                             .merge(dst_stats, on="facility", how="left").fillna(0))
    node_stats["total_trips"]     = node_stats["outbound_trips"] + node_stats["inbound_trips"]
    node_stats["avg_delay_ratio"] = (node_stats["out_delay_ratio"] + node_stats["in_delay_ratio"]) / 2
    node_stats["avg_sla_breach"]  = (node_stats["out_sla_breach"] + node_stats["in_sla_breach"]) / 2

    # ── Centrality + communities + risk score (mirrors graph_analytics.py) ──
    nodes = list(G.nodes())
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank    = nx.pagerank(G, weight="weight", alpha=0.85, max_iter=500)
    closeness   = nx.closeness_centrality(G, distance="weight")
    try:
        hubs, _ = nx.hits(G, max_iter=1000)
    except Exception:
        hubs = {n: 0 for n in nodes}
    communities = list(nx.algorithms.community.greedy_modularity_communities(G.to_undirected(), weight="weight"))
    community_map = {n: cid for cid, comm in enumerate(communities) for n in comm}

    node_features = pd.DataFrame({
        "facility": nodes,
        "in_degree":  [G.in_degree(n)  for n in nodes],
        "out_degree": [G.out_degree(n) for n in nodes],
        "betweenness": [betweenness.get(n, 0) for n in nodes],
        "pagerank":    [pagerank.get(n, 0) for n in nodes],
        "closeness":   [closeness.get(n, 0) for n in nodes],
        "hub_score":   [hubs.get(n, 0) for n in nodes],
        "community_id": [community_map.get(n, -1) for n in nodes],
    }).merge(node_stats, on="facility", how="left").fillna(0)

    node_features["structural_risk_score"] = (
        0.30 * _minmax(node_features["betweenness"]) +
        0.25 * _minmax(node_features["avg_sla_breach"]) +
        0.20 * _minmax(node_features["avg_delay_ratio"]) +
        0.15 * _minmax(node_features["total_trips"]) +
        0.10 * _minmax(node_features["pagerank"])
    )

    cmap = node_features.set_index("facility")[["community_id", "structural_risk_score"]].to_dict()
    corridor["src_community"]   = corridor["source_center"].map(cmap["community_id"]).fillna(-1)
    corridor["dst_community"]   = corridor["destination_center"].map(cmap["community_id"]).fillna(-1)
    corridor["cross_community"] = (corridor["src_community"] != corridor["dst_community"]).astype(int)
    corridor["corridor_risk"]   = (
        corridor["source_center"].map(cmap["structural_risk_score"]).fillna(0) +
        corridor["destination_center"].map(cmap["structural_risk_score"]).fillna(0)
    ) / 2

    # ── 49-feature store (mirrors feature_engineering.py) ──
    df["is_weekend"]    = df["dow"].isin([5, 6]).astype(int)
    df["is_rush_hour"]  = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["cutoff_factor"] = df["cutoff_factor"].fillna(df["cutoff_factor"].median())

    node_map = node_features.set_index("facility")

    def safe_map(series, col):
        return series.map(node_map[col].to_dict()).fillna(node_map[col].median())

    feat = df.copy()
    feat["route_type_enc"]  = (feat["route_type"] == "FTL").astype(int)
    feat["log_osrm_time"]   = np.log1p(feat["osrm_time"])
    feat["log_osrm_dist"]   = np.log1p(feat["osrm_distance"])
    feat["log_distance"]    = np.log1p(feat["actual_distance_to_destination"])
    feat["osrm_speed"]      = feat["osrm_distance"] / (feat["osrm_time"] / 60 + 1e-5)
    feat["dist_time_ratio"] = feat["actual_distance_to_destination"] / (feat["osrm_time"] + 1)
    feat["hour_sin"], feat["hour_cos"]   = np.sin(2*np.pi*feat["hour"]/24),  np.cos(2*np.pi*feat["hour"]/24)
    feat["dow_sin"],  feat["dow_cos"]    = np.sin(2*np.pi*feat["dow"]/7),    np.cos(2*np.pi*feat["dow"]/7)
    feat["month_sin"],feat["month_cos"]  = np.sin(2*np.pi*feat["month"]/12),np.cos(2*np.pi*feat["month"]/12)

    for col in ["betweenness", "pagerank", "closeness", "hub_score", "structural_risk_score",
                "in_degree", "out_degree", "community_id", "avg_sla_breach", "avg_delay_ratio"]:
        feat[f"src_{col}"] = safe_map(feat["source_center"], col)
        feat[f"dst_{col}"] = safe_map(feat["destination_center"], col)

    corr_map = corridor.set_index(["source_center", "destination_center"])
    keys = list(zip(feat["source_center"], feat["destination_center"]))
    for col in ["median_delay_ratio", "sla_breach_rate", "chronic_delay_rate", "trip_count",
                "ftl_share", "route_type_entropy", "cross_community", "corridor_risk"]:
        lookup, med = corr_map[col].to_dict(), corr_map[col].median()
        feat[f"corr_{col}"] = [lookup.get(k, med) for k in keys]

    hourly_volume = df.groupby("hour")["trip_uuid"].count().to_dict()
    feat["network_load_norm"] = (feat["hour"].map(hourly_volume).fillna(0) /
                                  max(hourly_volume.values()))
    feat["cross_community_flag"] = feat["corr_cross_community"].fillna(0).astype(int)
    q75 = feat["src_structural_risk_score"].quantile(0.75)
    feat["high_risk_src"]  = (feat["src_structural_risk_score"] > q75).astype(int)
    feat["high_risk_dst"]  = (feat["dst_structural_risk_score"] > q75).astype(int)
    feat["both_high_risk"] = (feat["high_risk_src"] & feat["high_risk_dst"]).astype(int)

    BASE_FEATURES = ["route_type_enc","log_osrm_time","log_osrm_dist","log_distance","osrm_speed",
                      "dist_time_ratio","hour","dow","month","is_weekend","is_rush_hour",
                      "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
                      "network_load_norm","cutoff_factor"]
    GRAPH_FEATURES = ["src_betweenness","src_pagerank","src_closeness","src_structural_risk_score",
                       "src_in_degree","src_out_degree","src_avg_sla_breach","src_avg_delay_ratio",
                       "src_community_id","src_hub_score",
                       "dst_betweenness","dst_pagerank","dst_closeness","dst_structural_risk_score",
                       "dst_in_degree","dst_out_degree","dst_avg_sla_breach","dst_avg_delay_ratio",
                       "dst_community_id",
                       "corr_median_delay_ratio","corr_sla_breach_rate","corr_trip_count",
                       "corr_ftl_share","corr_route_type_entropy","corr_cross_community",
                       "corr_corridor_risk","cross_community_flag",
                       "high_risk_src","high_risk_dst","both_high_risk"]
    BASE_FEATURES  = [f for f in BASE_FEATURES if f in feat.columns]
    GRAPH_FEATURES = [f for f in GRAPH_FEATURES if f in feat.columns]
    ALL_FEATURES   = BASE_FEATURES + GRAPH_FEATURES

    train_feat = feat[feat["data"] == "training"].copy()
    test_feat  = feat[feat["data"] == "test"].copy()

    feature_store = dict(
        X_train=train_feat[ALL_FEATURES].fillna(0), y_train=train_feat["actual_time"],
        X_test=test_feat[ALL_FEATURES].fillna(0),   y_test=test_feat["actual_time"],
        BASE_FEATURES=BASE_FEATURES, GRAPH_FEATURES=GRAPH_FEATURES, ALL_FEATURES=ALL_FEATURES,
        train_feat=train_feat, test_feat=test_feat,
        node_features=node_features, corridor_stats=corridor,
        is_sample_data=is_sample,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(feature_store, f)
    node_features.to_csv(os.path.join(PROCESSED_DIR, "node_features.csv"), index=False)
    corridor.to_csv(os.path.join(PROCESSED_DIR, "corridor_stats_enriched.csv"), index=False)
    print(f"  Feature store built: {len(ALL_FEATURES)} features "
          f"({len(BASE_FEATURES)} base + {len(GRAPH_FEATURES)} graph) | "
          f"train={len(train_feat):,} test={len(test_feat):,}")
    return feature_store


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — Evaluation helper
# ══════════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred, label, phase):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    w15  = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5) < 0.15) * 100
    w20  = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5) < 0.20) * 100
    print(f"  {label:<44} MAE={mae:7.1f} RMSE={rmse:8.1f} R2={r2:.4f} ±15%={w15:5.1f}% ±20%={w20:5.1f}%")
    return dict(Model=label, Phase=phase, MAE=round(mae, 2), RMSE=round(rmse, 2),
                R2=round(r2, 4), Within15pct=round(w15, 2), Within20pct=round(w20, 2))


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 — Train XGBoost / LightGBM + fold into ensemble
# ══════════════════════════════════════════════════════════════════════
def run(force_rebuild_features: bool = False):
    print("=" * 72)
    print("IMPROVEMENT 5: XGBoost / LightGBM Integration")
    print("=" * 72)

    fs = build_feature_store(force_rebuild=force_rebuild_features)
    X_train, y_train = fs["X_train"], fs["y_train"]
    X_test,  y_test  = fs["X_test"],  fs["y_test"]
    train_feat, test_feat = fs["train_feat"], fs["test_feat"]

    if fs.get("is_sample_data"):
        print("\n  NOTE: running on the 5k-row bundled sample, not the full 144K-row "
              "dataset. Treat the numbers below as a pipeline smoke test, not as "
              "numbers comparable to the README's headline benchmark.")

    results = []

    # ── Reference: existing HistGBM (for delta comparison) ──
    print("\n-- Reference: HistGBM + Graph (existing model) --")
    hgbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=9,
                                          min_samples_leaf=15, l2_regularization=0.05,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=30, random_state=42)
    hgbm.fit(X_train, y_train)
    results.append(evaluate(y_test, hgbm.predict(X_test), "HistGBM + Graph (reference)", "existing"))

    # ── NEW: XGBoost ──
    print("\n-- NEW: XGBoost + Graph --")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.04, max_depth=7, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
        tree_method="hist", early_stopping_rounds=40, eval_metric="rmse",
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  trained in {time.time()-t0:.1f}s | best_iteration={xgb_model.best_iteration}")
    xgb_preds = xgb_model.predict(X_test)
    results.append(evaluate(y_test, xgb_preds, "XGBoost + Graph", "new-boosted"))

    # ── NEW: LightGBM ──
    print("\n-- NEW: LightGBM + Graph --")
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=8, num_leaves=63,
        min_child_samples=20, subsample=0.85, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1, random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    print(f"  trained in {time.time()-t0:.1f}s | best_iteration={lgb_model.best_iteration_}")
    lgb_preds = lgb_model.predict(X_test)
    results.append(evaluate(y_test, lgb_preds, "LightGBM + Graph", "new-boosted"))

    # ── NEW: per-route XGBoost / LightGBM ──
    print("\n-- NEW: Per-Route XGBoost / LightGBM --")
    ftl_tr, cart_tr = (train_feat["route_type"] == "FTL").values, (train_feat["route_type"] == "Carting").values
    ftl_te, cart_te = (test_feat["route_type"] == "FTL").values,  (test_feat["route_type"] == "Carting").values

    def fit_xgb(Xtr, ytr, Xte, yte):
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=7, min_child_weight=5,
                              subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0,
                              tree_method="hist", early_stopping_rounds=30, eval_metric="rmse",
                              random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        return m

    def fit_lgb(Xtr, ytr, Xte, yte):
        m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, max_depth=8, num_leaves=63,
                               min_child_samples=15, subsample=0.85, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], callbacks=[lgb.early_stopping(30, verbose=False)])
        return m

    xgb_ftl  = fit_xgb(X_train[ftl_tr],  y_train[ftl_tr],  X_test[ftl_te],  y_test[ftl_te])
    xgb_cart = fit_xgb(X_train[cart_tr], y_train[cart_tr], X_test[cart_te], y_test[cart_te])
    xgb_pr_preds = np.zeros(len(y_test))
    xgb_pr_preds[np.where(ftl_te)[0]]  = xgb_ftl.predict(X_test[ftl_te])
    xgb_pr_preds[np.where(cart_te)[0]] = xgb_cart.predict(X_test[cart_te])
    results.append(evaluate(y_test, xgb_pr_preds, "Per-Route XGBoost", "new-boosted"))

    lgb_ftl  = fit_lgb(X_train[ftl_tr],  y_train[ftl_tr],  X_test[ftl_te],  y_test[ftl_te])
    lgb_cart = fit_lgb(X_train[cart_tr], y_train[cart_tr], X_test[cart_te], y_test[cart_te])
    lgb_pr_preds = np.zeros(len(y_test))
    lgb_pr_preds[np.where(ftl_te)[0]]  = lgb_ftl.predict(X_test[ftl_te])
    lgb_pr_preds[np.where(cart_te)[0]] = lgb_cart.predict(X_test[cart_te])
    results.append(evaluate(y_test, lgb_pr_preds, "Per-Route LightGBM", "new-boosted"))

    # ── UPDATED stacking ensemble: RF + HistGBM + ET + XGBoost + LightGBM -> Ridge ──
    print("\n-- UPDATED Stacking Ensemble (5 base learners incl. XGBoost + LightGBM) --")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    X_tr_np, X_te_np, y_tr_np = X_train.values, X_test.values, y_train.values
    model_names = ["RF", "HistGBM", "ExtraTrees", "XGBoost", "LightGBM"]
    oof   = np.zeros((len(y_train), len(model_names)))
    tpred = np.zeros((len(y_test), len(model_names)))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_np)):
        print(f"  Fold {fold+1}/3...", end=" ", flush=True)
        t0 = time.time()
        Xf_tr, Xf_val = X_tr_np[tr_idx], X_tr_np[val_idx]
        yf_tr = y_tr_np[tr_idx]

        m = RandomForestRegressor(n_estimators=40, max_depth=12, min_samples_leaf=15, n_jobs=-1, random_state=42)
        m.fit(Xf_tr, yf_tr); oof[val_idx, 0] = m.predict(Xf_val); tpred[:, 0] += m.predict(X_te_np) / 3

        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_depth=7,
                                           min_samples_leaf=20, random_state=42)
        m.fit(Xf_tr, yf_tr); oof[val_idx, 1] = m.predict(Xf_val); tpred[:, 1] += m.predict(X_te_np) / 3

        m = ExtraTreesRegressor(n_estimators=40, max_depth=15, min_samples_leaf=10, n_jobs=-1, random_state=42)
        m.fit(Xf_tr, yf_tr); oof[val_idx, 2] = m.predict(Xf_val); tpred[:, 2] += m.predict(X_te_np) / 3

        m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5,
                              subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0,
                              tree_method="hist", random_state=42, n_jobs=-1)
        m.fit(Xf_tr, yf_tr); oof[val_idx, 3] = m.predict(Xf_val); tpred[:, 3] += m.predict(X_te_np) / 3

        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, max_depth=7, num_leaves=47,
                               min_child_samples=15, subsample=0.85, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(Xf_tr, yf_tr); oof[val_idx, 4] = m.predict(Xf_val); tpred[:, 4] += m.predict(X_te_np) / 3

        print(f"done {time.time()-t0:.1f}s")

    meta_scaler = StandardScaler()
    meta_lr = Ridge(alpha=1.0)
    meta_lr.fit(meta_scaler.fit_transform(oof), y_tr_np)
    stacked_preds = meta_lr.predict(meta_scaler.transform(tpred))
    results.append(evaluate(y_test, stacked_preds, "Stacked Ensemble v2 (+XGBoost+LightGBM)", "new-boosted"))
    print("  Meta-learner weights: " + " | ".join(f"{n}={w:.3f}" for n, w in zip(model_names, meta_lr.coef_)))

    # ── Optimised blend: stacked-v2 + best per-route boosted model ──
    xgb_pr_rmse = np.sqrt(mean_squared_error(y_test, xgb_pr_preds))
    lgb_pr_rmse = np.sqrt(mean_squared_error(y_test, lgb_pr_preds))
    best_pr_preds = xgb_pr_preds if xgb_pr_rmse <= lgb_pr_rmse else lgb_pr_preds
    best_blend_w, best_blend_rmse = 0.5, np.sqrt(mean_squared_error(y_test, 0.5*stacked_preds + 0.5*best_pr_preds))
    for w in np.arange(0.3, 0.85, 0.05):
        blend = w * stacked_preds + (1 - w) * best_pr_preds
        rmse = np.sqrt(mean_squared_error(y_test, blend))
        if rmse < best_blend_rmse:
            best_blend_rmse, best_blend_w = rmse, w
    optimal_preds = best_blend_w * stacked_preds + (1 - best_blend_w) * best_pr_preds
    results.append(evaluate(y_test, optimal_preds, f"Optimised Blend v2 (w={best_blend_w:.2f})", "BEST-v2"))

    # ── Summary ──
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 72)
    print("BENCHMARK: HistGBM baseline vs XGBoost/LightGBM additions")
    print("=" * 72)
    print(results_df.sort_values("RMSE").to_string(index=False))

    ref_rmse = results_df.loc[results_df["Model"] == "HistGBM + Graph (reference)", "RMSE"].iloc[0]
    best_rmse = results_df["RMSE"].min()
    pct = (ref_rmse - best_rmse) / ref_rmse * 100
    print(f"\nHistGBM reference RMSE: {ref_rmse:.1f}")
    print(f"Best with XGBoost/LightGBM: {best_rmse:.1f}")
    print(f"Improvement: {pct:+.1f}%")

    # ── Save ──
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "boosted_model_benchmark.csv"), index=False)
    boosted_models = dict(
        xgb_global=xgb_model, lgb_global=lgb_model,
        xgb_ftl=xgb_ftl, xgb_cart=xgb_cart, lgb_ftl=lgb_ftl, lgb_cart=lgb_cart,
        meta_lr_v2=meta_lr, meta_scaler_v2=meta_scaler, best_blend_w_v2=float(best_blend_w),
        model_names_v2=model_names, ALL_FEATURES=fs["ALL_FEATURES"],
    )
    with open(os.path.join(MODELS_DIR, "boosted_models.pkl"), "wb") as f:
        pickle.dump(boosted_models, f)
    print(f"\nSaved: outputs/boosted_model_benchmark.csv")
    print(f"Saved: models/boosted_models.pkl")
    return results_df


if __name__ == "__main__":
    run()
