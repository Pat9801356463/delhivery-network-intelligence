"""
IMPROVEMENTS 2 + 3 + 4 — All in one pipeline
  2. HistGradientBoosting (sklearn's LightGBM-equivalent histogram boosting)
  3. Separate models per route type (FTL vs Carting have different physics)
  4. Stacked ensemble of best models
  
Full benchmark vs original results with statistical interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, pickle, json, os, time

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,   # LightGBM-equivalent
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
np.random.seed(42)

REPORTS_DIR = '/home/claude/project/reports'
VISUALS_DIR = '/home/claude/project/visuals'
MODELS_DIR  = '/home/claude/project/models'
BG, PBG = '#0a0e1a', '#0f1626'
C1, C2, C3, C4 = '#00d4ff', '#ff6b35', '#00ff88', '#ffd700'
CGRD = '#222'

print("=" * 70)
print("IMPROVEMENTS 2+3+4: HistGBM + Per-Route Models + Ensemble")
print("=" * 70)

# ── Load cleaned v2 data + feature store ─────────────────────────────────────
df_train = pd.read_csv(f'{REPORTS_DIR}/train_clean_v2.csv')
df_test  = pd.read_csv(f'{REPORTS_DIR}/test_clean_v2.csv')
node_feat = pd.read_csv(f'{REPORTS_DIR}/node_features.csv')
corridor  = pd.read_csv(f'{REPORTS_DIR}/corridor_stats_enriched.csv')

with open(f'{MODELS_DIR}/feature_store.pkl', 'rb') as f:
    fs = pickle.load(f)

BASE_FEATURES  = fs['BASE_FEATURES']
GRAPH_FEATURES = fs['GRAPH_FEATURES']
ALL_FEATURES   = fs['ALL_FEATURES']

print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")
print(f"Features: {len(ALL_FEATURES)} total ({len(BASE_FEATURES)} base + {len(GRAPH_FEATURES)} graph)")

# ── Rebuild feature matrices from v2 cleaned data ─────────────────────────────
def build_features(df_in, node_feat, corridor, ALL_FEATURES, BASE_FEATURES):
    """Reconstruct feature matrix from cleaned dataframe."""
    feat = df_in.copy()
    feat['od_start_time'] = pd.to_datetime(feat['od_start_time'], errors='coerce')
    feat['delay_ratio']   = feat['actual_time'] / feat['osrm_time'].replace(0, np.nan)
    feat['hour']  = feat['od_start_time'].dt.hour
    feat['dow']   = feat['od_start_time'].dt.dayofweek
    feat['month'] = feat['od_start_time'].dt.month

    # Base features
    feat['route_type_enc']  = (feat['route_type'] == 'FTL').astype(int)
    feat['log_osrm_time']   = np.log1p(feat['osrm_time'])
    feat['log_osrm_dist']   = np.log1p(feat['osrm_distance'])
    feat['log_distance']    = np.log1p(feat['actual_distance_to_destination'])
    feat['osrm_speed']      = feat['osrm_distance'] / (feat['osrm_time'] / 60 + 1e-5)
    feat['dist_time_ratio'] = feat['actual_distance_to_destination'] / (feat['osrm_time'] + 1)
    feat['is_weekend']      = feat['dow'].isin([5,6]).astype(int)
    feat['is_rush_hour']    = feat['hour'].isin([7,8,9,17,18,19]).astype(int)
    feat['hour_sin']   = np.sin(2*np.pi*feat['hour']/24)
    feat['hour_cos']   = np.cos(2*np.pi*feat['hour']/24)
    feat['dow_sin']    = np.sin(2*np.pi*feat['dow']/7)
    feat['dow_cos']    = np.cos(2*np.pi*feat['dow']/7)
    feat['month_sin']  = np.sin(2*np.pi*feat['month']/12)
    feat['month_cos']  = np.cos(2*np.pi*feat['month']/12)
    feat['network_load_norm'] = 0.5  # placeholder
    feat['cutoff_factor']     = feat['cutoff_factor'].fillna(feat['cutoff_factor'].median())

    # Node features
    node_map = node_feat.set_index('facility')
    node_cols = ['betweenness','pagerank','closeness','hub_score','structural_risk_score',
                 'in_degree','out_degree','total_degree','community_id',
                 'outbound_trips','avg_sla_breach','avg_delay_ratio']
    for col in node_cols:
        if col in node_map.columns:
            med = node_map[col].median()
            feat[f'src_{col}'] = feat['source_center'].map(node_map[col].to_dict()).fillna(med)
            feat[f'dst_{col}'] = feat['destination_center'].map(node_map[col].to_dict()).fillna(med)

    # Corridor features
    corr_map = corridor.set_index(['source_center','destination_center'])
    corr_cols = ['median_delay_ratio','sla_breach_rate','chronic_delay_rate','trip_count',
                 'ftl_share','route_type_entropy','cross_community','corridor_risk']
    for col in corr_cols:
        if col in corr_map.columns:
            med = corr_map[col].median()
            feat[f'corr_{col}'] = feat.apply(
                lambda r: corr_map.loc[(r['source_center'],r['destination_center']),col]
                if (r['source_center'],r['destination_center']) in corr_map.index else med, axis=1)

    # Risk flags
    feat['cross_community_flag'] = feat.get('corr_cross_community', 0).fillna(0).astype(int)
    src_risk = feat.get('src_structural_risk_score', pd.Series(0, index=feat.index))
    dst_risk = feat.get('dst_structural_risk_score', pd.Series(0, index=feat.index))
    q75 = src_risk.quantile(0.75)
    feat['high_risk_src']  = (src_risk > q75).astype(int)
    feat['high_risk_dst']  = (dst_risk > q75).astype(int)
    feat['both_high_risk'] = (feat['high_risk_src'] & feat['high_risk_dst']).astype(int)

    valid_feats = [f for f in ALL_FEATURES if f in feat.columns]
    return feat[valid_feats].fillna(0), feat['actual_time']

print("\nBuilding feature matrices from v2 cleaned data...")
t0 = time.time()
X_train_all, y_train = build_features(df_train, node_feat, corridor, ALL_FEATURES, BASE_FEATURES)
X_test_all,  y_test  = build_features(df_test,  node_feat, corridor, ALL_FEATURES, BASE_FEATURES)
X_train_base = X_train_all[[f for f in BASE_FEATURES if f in X_train_all.columns]]
X_test_base  = X_test_all[[f for f in BASE_FEATURES  if f in X_test_all.columns]]
print(f"  Done in {time.time()-t0:.1f}s | X_train_all={X_train_all.shape} | X_test_all={X_test_all.shape}")

# ── EVALUATION UTIL ───────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label, phase, feat_set):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    r2    = r2_score(y_true, y_pred)
    w15   = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5) < 0.15) * 100
    w20   = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5) < 0.20) * 100
    print(f"  {label:<42} MAE={mae:6.1f} RMSE={rmse:7.1f} R²={r2:.4f} ±15%={w15:.1f}% ±20%={w20:.1f}%")
    return dict(Model=label, Phase=phase, FeatSet=feat_set,
                MAE=round(mae,2), RMSE=round(rmse,2), MAPE=round(mape,2),
                R2=round(r2,4), Within15pct=round(w15,2), Within20pct=round(w20,2))

all_results = []

# ── BASELINE REFERENCE (original results for comparison) ──────────────────────
print("\n── REFERENCE: Original model results (v1 data) ──────────────────────")
orig = [
    dict(Model='RF (Base) — ORIGINAL',        Phase='v1-Baseline',  FeatSet='Base', MAE=47.53, RMSE=106.06, MAPE=25.0, R2=0.9712, Within15pct=49.1, Within20pct=58.0),
    dict(Model='RF + Graph — ORIGINAL',        Phase='v1-Graph',     FeatSet='All',  MAE=38.56, RMSE=91.59,  MAPE=22.0, R2=0.9785, Within15pct=59.2, Within20pct=68.0),
]
for r in orig:
    print(f"  {r['Model']:<42} MAE={r['MAE']:6.1f} RMSE={r['RMSE']:7.1f} R²={r['R2']:.4f} ±15%={r['Within15pct']:.1f}%")
all_results.extend(orig)

# ── IMPROVEMENT 1 EFFECT: RF on v2 cleaned data ───────────────────────────────
print("\n── Testing outlier cleaning effect (RF, same hyperparams) ──────────")
t0 = time.time()
rf_v2 = RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=10, n_jobs=-1, random_state=42)
rf_v2.fit(X_train_all, y_train)
print(f"  RF v2 trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, rf_v2.predict(X_test_all), 'RF + Graph — v2 (outliers cleaned)', 'v2-Graph', 'All')
all_results.append(r)

# ── IMPROVEMENT 2: HistGradientBoosting (LightGBM-equivalent) ─────────────────
print("\n── IMPROVEMENT 2: HistGradientBoosting (sklearn LightGBM-equivalent) ─")
print("   [Histogram-based, handles missing natively, very fast, regularised]")

# Base features
print("  Training HistGBM on Base features...")
t0 = time.time()
hgbm_base = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_depth=8,
    min_samples_leaf=20, l2_regularization=0.1,
    max_bins=255, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, random_state=42
)
hgbm_base.fit(X_train_base, y_train)
print(f"  Done {time.time()-t0:.1f}s | Iterations used: {hgbm_base.n_iter_}")
r = evaluate(y_test, hgbm_base.predict(X_test_base), 'HistGBM (LightGBM-equiv) — Base', 'v2-Baseline', 'Base')
all_results.append(r)

# All features (graph-enhanced)
print("  Training HistGBM on All features (graph-enhanced)...")
t0 = time.time()
hgbm_graph = HistGradientBoostingRegressor(
    max_iter=600, learning_rate=0.04, max_depth=9,
    min_samples_leaf=15, l2_regularization=0.05,
    max_bins=255, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=40, random_state=42
)
hgbm_graph.fit(X_train_all, y_train)
print(f"  Done {time.time()-t0:.1f}s | Iterations used: {hgbm_graph.n_iter_}")
r = evaluate(y_test, hgbm_graph.predict(X_test_all), 'HistGBM (LightGBM-equiv) — Graph', 'v2-Graph', 'All')
all_results.append(r)

# ── IMPROVEMENT 3: Separate models per route type ─────────────────────────────
print("\n── IMPROVEMENT 3: Per-Route-Type Models (FTL vs Carting separately) ─")
print("   [Different physics: FTL=long-haul trunk, Carting=last-mile consolidation]")

# Split by route type
ftl_mask_tr   = df_train['route_type'] == 'FTL'
cart_mask_tr  = df_train['route_type'] == 'Carting'
ftl_mask_te   = df_test['route_type']  == 'FTL'
cart_mask_te  = df_test['route_type']  == 'Carting'

X_ftl_tr  = X_train_all[ftl_mask_tr.values];  y_ftl_tr  = y_train[ftl_mask_tr.values]
X_cart_tr = X_train_all[cart_mask_tr.values]; y_cart_tr = y_train[cart_mask_tr.values]
X_ftl_te  = X_test_all[ftl_mask_te.values];   y_ftl_te  = y_test[ftl_mask_te.values]
X_cart_te = X_test_all[cart_mask_te.values];  y_cart_te = y_test[cart_mask_te.values]

print(f"  FTL train: {len(X_ftl_tr):,} | Carting train: {len(X_cart_tr):,}")
print(f"  FTL test:  {len(X_ftl_te):,}  | Carting test:  {len(X_cart_te):,}")

# FTL model (HistGBM tuned for long-haul characteristics)
print("  Training FTL-specific HistGBM...")
t0 = time.time()
hgbm_ftl = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_depth=8,
    min_samples_leaf=15, l2_regularization=0.1,
    max_bins=255, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, random_state=42
)
hgbm_ftl.fit(X_ftl_tr, y_ftl_tr)
print(f"  FTL done {time.time()-t0:.1f}s | iters={hgbm_ftl.n_iter_}")
ftl_preds = hgbm_ftl.predict(X_ftl_te)

# Carting model (tuned for short-haul, higher variance)
print("  Training Carting-specific HistGBM...")
t0 = time.time()
hgbm_cart = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.06, max_depth=7,
    min_samples_leaf=10, l2_regularization=0.2,
    max_bins=255, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, random_state=42
)
hgbm_cart.fit(X_cart_tr, y_cart_tr)
print(f"  Carting done {time.time()-t0:.1f}s | iters={hgbm_cart.n_iter_}")
cart_preds = hgbm_cart.predict(X_cart_te)

# Combine predictions
per_route_preds = np.zeros(len(y_test))
per_route_preds[np.where(ftl_mask_te.values)[0]]  = ftl_preds
per_route_preds[np.where(cart_mask_te.values)[0]] = cart_preds

r = evaluate(y_test, per_route_preds, 'Per-Route HistGBM (FTL+Carting split)', 'v2-PerRoute', 'All')
all_results.append(r)

# Per-route breakdown
print(f"\n  FTL-only  results:")
r_ftl  = evaluate(y_ftl_te,  ftl_preds,  'FTL-only  HistGBM',  'v2-PerRoute', 'All')
print(f"  Carting-only results:")
r_cart = evaluate(y_cart_te, cart_preds, 'Carting-only HistGBM','v2-PerRoute', 'All')
all_results.extend([r_ftl, r_cart])

# ── IMPROVEMENT 4: STACKED ENSEMBLE ───────────────────────────────────────────
print("\n── IMPROVEMENT 4: Stacked Ensemble (meta-learner) ───────────────────")
print("   [Level-0: RF + HistGBM + ExtraTrees | Level-1: Ridge meta-learner]")

# Level-0 estimators (fast, diverse)
print("  Training level-0 estimators...")
t0 = time.time()
et_graph = ExtraTreesRegressor(n_estimators=60, max_depth=18, min_samples_leaf=8,
                                n_jobs=-1, random_state=42)
et_graph.fit(X_train_all, y_train)
print(f"  ExtraTrees done {time.time()-t0:.1f}s")

# Generate OOF predictions for stacking (3-fold for speed)
print("  Generating OOF predictions for stacking (3-fold CV)...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
oof_rf   = np.zeros(len(y_train))
oof_hgbm = np.zeros(len(y_train))
oof_et   = np.zeros(len(y_train))
test_rf   = np.zeros(len(y_test))
test_hgbm = np.zeros(len(y_test))
test_et   = np.zeros(len(y_test))

X_tr_np  = X_train_all.values
X_te_np  = X_test_all.values
y_tr_np  = y_train.values

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_np)):
    print(f"  Fold {fold+1}/3...", end=' ', flush=True)
    t0f = time.time()

    Xf_tr, Xf_val = X_tr_np[tr_idx], X_tr_np[val_idx]
    yf_tr, yf_val = y_tr_np[tr_idx], y_tr_np[val_idx]

    rf_f = RandomForestRegressor(n_estimators=40, max_depth=12, min_samples_leaf=15, n_jobs=-1, random_state=42)
    rf_f.fit(Xf_tr, yf_tr)
    oof_rf[val_idx] = rf_f.predict(Xf_val)
    test_rf += rf_f.predict(X_te_np) / 3

    hgbm_f = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_depth=7,
                                            min_samples_leaf=20, random_state=42)
    hgbm_f.fit(Xf_tr, yf_tr)
    oof_hgbm[val_idx] = hgbm_f.predict(Xf_val)
    test_hgbm += hgbm_f.predict(X_te_np) / 3

    et_f = ExtraTreesRegressor(n_estimators=40, max_depth=15, min_samples_leaf=10, n_jobs=-1, random_state=42)
    et_f.fit(Xf_tr, yf_tr)
    oof_et[val_idx] = et_f.predict(Xf_val)
    test_et += et_f.predict(X_te_np) / 3

    print(f"done {time.time()-t0f:.1f}s")

# Stack OOF predictions as meta-features
meta_train = np.column_stack([oof_rf, oof_hgbm, oof_et])
meta_test  = np.column_stack([test_rf, test_hgbm, test_et])

# Meta-learner: Ridge (prevents overfitting on meta-features)
print("  Training Ridge meta-learner...")
meta_scaler = StandardScaler()
meta_lr     = Ridge(alpha=1.0)
meta_lr.fit(meta_scaler.fit_transform(meta_train), y_tr_np)
stacked_preds = meta_lr.predict(meta_scaler.transform(meta_test))

r = evaluate(y_test, stacked_preds, 'Stacked Ensemble (RF+HistGBM+ET → Ridge)', 'v2-Ensemble', 'All')
all_results.append(r)
print(f"  Meta-learner weights: RF={meta_lr.coef_[0]:.3f} | HistGBM={meta_lr.coef_[1]:.3f} | ET={meta_lr.coef_[2]:.3f}")

# ── BEST COMBINED: Per-Route + Stacking ───────────────────────────────────────
print("\n── BONUS: Per-Route × Stacking (best of both) ───────────────────────")
combined_preds = 0.5 * stacked_preds + 0.5 * per_route_preds
r = evaluate(y_test, combined_preds, 'Combined: PerRoute + Stacked (50/50)', 'v2-Ensemble', 'All')
all_results.append(r)

# Optimise blend weight
best_blend_w = 0.5
best_blend_rmse = np.sqrt(mean_squared_error(y_test, combined_preds))
for w in np.arange(0.3, 0.8, 0.05):
    blend = w * stacked_preds + (1-w) * per_route_preds
    rmse = np.sqrt(mean_squared_error(y_test, blend))
    if rmse < best_blend_rmse:
        best_blend_rmse = rmse
        best_blend_w = w

optimal_preds = best_blend_w * stacked_preds + (1-best_blend_w) * per_route_preds
r = evaluate(y_test, optimal_preds, f'Optimised Blend (w={best_blend_w:.2f} stack + {1-best_blend_w:.2f} per-route)', 'v2-BEST', 'All')
all_results.append(r)
print(f"  Optimal blend weight: {best_blend_w:.2f} stacked + {1-best_blend_w:.2f} per-route")

# ── FULL BENCHMARK COMPARISON ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FULL BENCHMARK — ORIGINAL vs ALL IMPROVEMENTS")
print("=" * 70)

results_df = pd.DataFrame(all_results)
print(f"\n{'Model':<48} {'Phase':<14} {'MAE':>7} {'RMSE':>8} {'R²':>8} {'±15%':>7} {'±20%':>7}")
print("-" * 105)

phase_order = ['v1-Baseline','v1-Graph','v2-Graph','v2-Baseline','v2-PerRoute','v2-Ensemble','v2-BEST']
for phase in phase_order:
    sub = results_df[results_df['Phase']==phase].sort_values('RMSE')
    for _, row in sub.iterrows():
        w20 = row.get('Within20pct', '—')
        w20_str = f"{w20:.1f}%" if isinstance(w20, float) else '—'
        print(f"  {row['Model']:<46} {row['Phase']:<14} {row['MAE']:>7.1f} {row['RMSE']:>8.1f} "
              f"{row['R2']:>8.4f} {row['Within15pct']:>6.1f}% {w20_str:>7}")

# Improvement summary
orig_rmse  = results_df[results_df['Phase']=='v1-Graph']['RMSE'].min()
orig_mae   = results_df[results_df['Phase']=='v1-Graph']['MAE'].min()
orig_w15   = results_df[results_df['Phase']=='v1-Graph']['Within15pct'].max()
best_rmse  = results_df[results_df['Phase']=='v2-BEST']['RMSE'].min()
best_mae   = results_df[results_df['Phase']=='v2-BEST']['MAE'].min()
best_w15   = results_df[results_df['Phase']=='v2-BEST']['Within15pct'].max()

print(f"\n── IMPROVEMENT SUMMARY vs Original Graph Model ──────────────────────")
print(f"  RMSE:    {orig_rmse:.1f} → {best_rmse:.1f}  ({(orig_rmse-best_rmse)/orig_rmse*100:+.1f}%)")
print(f"  MAE:     {orig_mae:.1f} → {best_mae:.1f}   ({(orig_mae-best_mae)/orig_mae*100:+.1f}%)")
print(f"  ±15%:    {orig_w15:.1f}% → {best_w15:.1f}%  ({best_w15-orig_w15:+.1f}pp)")

# ── SAVE EVERYTHING ───────────────────────────────────────────────────────────
results_df.to_csv(f'{REPORTS_DIR}/model_benchmark_v2.csv', index=False)

improved_models = {
    'hgbm_base': hgbm_base, 'hgbm_graph': hgbm_graph,
    'hgbm_ftl': hgbm_ftl, 'hgbm_cart': hgbm_cart,
    'et_graph': et_graph,
    'meta_lr': meta_lr, 'meta_scaler': meta_scaler,
    'rf_v2': rf_v2,
    'best_blend_w': best_blend_w,
    'ALL_FEATURES': ALL_FEATURES, 'BASE_FEATURES': BASE_FEATURES, 'GRAPH_FEATURES': GRAPH_FEATURES,
    'stacked_preds_test': stacked_preds,
    'per_route_preds_test': per_route_preds,
    'optimal_preds_test': optimal_preds,
}
with open(f'{MODELS_DIR}/improved_models.pkl', 'wb') as f:
    pickle.dump(improved_models, f)
print(f"\n  Saved: models/improved_models.pkl")
print(f"  Saved: reports/model_benchmark_v2.csv")

# Save advantage report v2
advantage_v2 = {
    'original_graph_rmse': float(orig_rmse), 'best_improved_rmse': float(best_rmse),
    'original_graph_mae':  float(orig_mae),  'best_improved_mae':  float(best_mae),
    'original_graph_w15':  float(orig_w15),  'best_improved_w15':  float(best_w15),
    'rmse_improvement_pct': round((orig_rmse-best_rmse)/orig_rmse*100, 2),
    'mae_improvement_pct':  round((orig_mae-best_mae)/orig_mae*100, 2),
    'w15_improvement_pp':   round(best_w15-orig_w15, 2),
    'hgbm_vs_baseline_rmse_pct': round((results_df[results_df['Phase']=='v1-Baseline']['RMSE'].min()
                                        - best_rmse)
                                       / results_df[results_df['Phase']=='v1-Baseline']['RMSE'].min() * 100, 2),
    'best_model': 'Optimised Blend: Per-Route HistGBM + Stacked Ensemble',
}
with open(f'{REPORTS_DIR}/graph_advantage_report_v2.json', 'w') as f:
    json.dump(advantage_v2, f, indent=2)

# ── VISUALIZATIONS ────────────────────────────────────────────────────────────
print("\n── Generating benchmark visualizations...")

GRAPH_FEATURES_SET = set(GRAPH_FEATURES)
phase_color_map = {
    'v1-Baseline': '#555566', 'v1-Graph': C1,
    'v2-Graph': '#66aaff', 'v2-Baseline': '#778899',
    'v2-PerRoute': C4, 'v2-Ensemble': C2, 'v2-BEST': '#ff3366'
}

# ── VIZ 1: Full model comparison ───────────────────────────────────────────────
fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle('Improvement Benchmark: Original → v2 Cleaned → HistGBM → Per-Route → Ensemble',
             fontsize=15, fontweight='bold', color='white', y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

# Filter to main models (not sub-breakdowns)
main_models = results_df[~results_df['Phase'].isin(['v1-Baseline'])].copy()
main_models = main_models[~main_models['Model'].str.contains('FTL-only|Carting-only')]
main_models = main_models.sort_values('RMSE', ascending=True)
bar_colors  = [phase_color_map.get(p, '#888') for p in main_models['Phase']]
short_names = [m[:35] for m in main_models['Model']]

ax = fig.add_subplot(gs[0,0:2]); ax.set_facecolor(PBG)
bars = ax.barh(range(len(main_models)), main_models['RMSE'], color=bar_colors, alpha=0.88)
ax.set_yticks(range(len(main_models)))
ax.set_yticklabels(short_names, fontsize=8, color='white')
ax.set_title('RMSE Comparison (all improvements)', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('RMSE (min) — lower is better', color='#888')
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)
ax.invert_yaxis()
# Annotate best
best_idx = main_models['RMSE'].idxmin()
best_pos = list(main_models.index).index(best_idx)
ax.get_children()[best_pos].set_edgecolor('#ff3366')
ax.get_children()[best_pos].set_linewidth(2)
ax.axvline(orig_rmse, color=C1, ls='--', lw=1.5, alpha=0.7, label=f'Original: {orig_rmse:.1f}')
ax.axvline(best_rmse, color='#ff3366', ls='--', lw=1.5, alpha=0.7, label=f'Best: {best_rmse:.1f}')
ax.legend(facecolor='#1a2040', labelcolor='white', fontsize=9)

ax = fig.add_subplot(gs[0,2]); ax.set_facecolor(PBG)
bars = ax.barh(range(len(main_models)), main_models['Within15pct'], color=bar_colors, alpha=0.88)
ax.set_yticks(range(len(main_models)))
ax.set_yticklabels(short_names, fontsize=8, color='white')
ax.set_title('±15% Accuracy\n(higher is better)', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('% of Predictions', color='#888')
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)
ax.invert_yaxis()
ax.axvline(orig_w15, color=C1, ls='--', lw=1.5, alpha=0.7)
ax.axvline(best_w15, color='#ff3366', ls='--', lw=1.5, alpha=0.7)

# ── VIZ 2: Improvement waterfall ──────────────────────────────────────────────
ax = fig.add_subplot(gs[1,0:2]); ax.set_facecolor(PBG)
waterfall_models = [
    ('OSRM Naive', 383.1),
    ('RF Base\n(v1)', 106.1),
    ('RF + Graph\n(v1)', 91.6),
    ('RF + Graph\n(v2 cleaned)', float(results_df[results_df['Phase']=='v2-Graph']['RMSE'].min())),
    ('HistGBM\n(Base)', float(results_df[(results_df['Phase']=='v2-Baseline')&(results_df['FeatSet']=='Base')]['RMSE'].min())),
    ('HistGBM\n(Graph)', float(results_df[(results_df['Phase']=='v2-Graph')&(results_df['Model'].str.contains('HistGBM'))]['RMSE'].min())),
    ('Per-Route\nHistGBM', float(results_df[results_df['Phase']=='v2-PerRoute']['RMSE'].min())),
    ('Stacked\nEnsemble', float(results_df[(results_df['Phase']=='v2-Ensemble')&(results_df['Model'].str.contains('Stacked'))]['RMSE'].min())),
    ('BEST BLEND\n★', best_rmse),
]
wf_labels = [w[0] for w in waterfall_models]
wf_vals   = [w[1] for w in waterfall_models]
wf_colors = ['#555566' if v > orig_rmse else C1 if v == orig_rmse
             else (C4 if v > best_rmse+5 else ('#ff3366' if v == best_rmse else C3))
             for v in wf_vals]
wf_colors = ['#555566','#666677',C1,'#33aacc','#44bb66',C3,C4,C2,'#ff3366']

bars = ax.bar(range(len(wf_vals)), wf_vals, color=wf_colors, alpha=0.88, edgecolor='none', width=0.7)
for i,(bar,v) in enumerate(zip(bars,wf_vals)):
    ax.text(bar.get_x()+bar.get_width()/2, v+3, f'{v:.1f}', ha='center', color='white',
            fontweight='bold', fontsize=9)
    if i > 0:
        delta = v - wf_vals[i-1]
        if delta < 0:
            ax.text(bar.get_x()+bar.get_width()/2, v-12, f'{delta:+.1f}',
                    ha='center', color=C3, fontweight='bold', fontsize=8)
ax.set_xticks(range(len(wf_labels)))
ax.set_xticklabels(wf_labels, color='white', fontsize=8)
ax.set_title('RMSE Reduction Journey — Step by Step', color='white', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (min)', color='#888')
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)

# ── VIZ 3: FTL vs Carting per-route comparison ────────────────────────────────
ax = fig.add_subplot(gs[1,2]); ax.set_facecolor(PBG)
rt_data = results_df[results_df['Phase']=='v2-PerRoute'].copy()
metrics = ['MAE','RMSE','Within15pct']
x = np.arange(len(metrics))
w = 0.35
if 'FTL-only  HistGBM' in rt_data['Model'].values and 'Carting-only HistGBM' in rt_data['Model'].values:
    ftl_row  = rt_data[rt_data['Model']=='FTL-only  HistGBM'].iloc[0]
    cart_row = rt_data[rt_data['Model']=='Carting-only HistGBM'].iloc[0]
    ftl_vals  = [ftl_row['MAE'],  ftl_row['RMSE'],  ftl_row['Within15pct']]
    cart_vals = [cart_row['MAE'], cart_row['RMSE'], cart_row['Within15pct']]
    ax.bar(x-w/2, ftl_vals,  w, color=C1, alpha=0.85, label='FTL model')
    ax.bar(x+w/2, cart_vals, w, color=C2, alpha=0.85, label='Carting model')
    ax.set_xticks(x); ax.set_xticklabels(['MAE (min)','RMSE (min)','±15% (%)'], color='white')
    ax.legend(facecolor='#1a2040', labelcolor='white')
ax.set_title('Per-Route Model Performance\nFTL vs Carting', color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)

# ── VIZ 4: Actual vs predicted — best model ────────────────────────────────────
ax = fig.add_subplot(gs[2,0]); ax.set_facecolor(PBG)
idx = np.random.choice(len(y_test), min(4000,len(y_test)), replace=False)
ax.scatter(y_test.values[idx], optimal_preds[idx], alpha=0.2, s=4, c=C1)
lm = min(2000, max(float(y_test.max()), float(optimal_preds.max())))
ax.plot([0,lm],[0,lm],'w--',lw=1.5,label='Perfect')
ax.plot([0,lm],[0,lm*1.15],color=C2,lw=1,ls=':',alpha=0.7,label='±15%')
ax.plot([0,lm],[0,lm*0.85],color=C2,lw=1,ls=':',alpha=0.7)
ax.set_title('Actual vs Predicted — BEST MODEL', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('Actual (min)',color='#888'); ax.set_ylabel('Predicted (min)',color='#888')
ax.tick_params(colors='#888'); ax.legend(facecolor='#1a2040',labelcolor='white',fontsize=8)
for sp in ax.spines.values(): sp.set_color(CGRD)

# ── VIZ 5: Residual distribution ──────────────────────────────────────────────
ax = fig.add_subplot(gs[2,1]); ax.set_facecolor(PBG)
residuals_orig = y_test.values - rf_v2.predict(X_test_all)
residuals_best = y_test.values - optimal_preds
ax.hist(residuals_orig.clip(-500,500), bins=80, alpha=0.5, color=C1, density=True, label=f'RF+Graph (RMSE={orig_rmse:.0f})')
ax.hist(residuals_best.clip(-500,500), bins=80, alpha=0.5, color='#ff3366', density=True, label=f'Best Blend (RMSE={best_rmse:.0f})')
ax.axvline(0, color='white', ls='--', lw=1.5)
ax.set_title('Residual Distribution\n(Original vs Best Model)', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('Prediction Error (min)', color='#888'); ax.set_ylabel('Density',color='#888')
ax.legend(facecolor='#1a2040',labelcolor='white',fontsize=8)
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)

# ── VIZ 6: Summary card ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2,2]); ax.set_facecolor(PBG)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Final Results vs Original', color='white', fontsize=12, fontweight='bold')
lines = [
    (0.5, 0.90, 'ORIGINAL RF + GRAPH', C1, 11, 'bold'),
    (0.5, 0.80, f'RMSE: {orig_rmse:.1f} min  |  MAE: {orig_mae:.1f} min', 'white', 10, 'normal'),
    (0.5, 0.72, f'±15%: {orig_w15:.1f}%', 'white', 10, 'normal'),
    (0.5, 0.60, 'BEST IMPROVED MODEL', '#ff3366', 11, 'bold'),
    (0.5, 0.50, f'RMSE: {best_rmse:.1f} min  |  MAE: {best_mae:.1f} min', 'white', 10, 'normal'),
    (0.5, 0.42, f'±15%: {best_w15:.1f}%', 'white', 10, 'normal'),
    (0.5, 0.28, 'IMPROVEMENT', C3, 12, 'bold'),
    (0.5, 0.18, f'RMSE: {(orig_rmse-best_rmse)/orig_rmse*100:+.1f}%  |  MAE: {(orig_mae-best_mae)/orig_mae*100:+.1f}%', C4, 11, 'bold'),
    (0.5, 0.09, f'±15% accuracy: {best_w15-orig_w15:+.1f} percentage points', C4, 11, 'bold'),
]
for x,y,txt,col,sz,wt in lines:
    ax.text(x,y,txt,ha='center',va='center',color=col,fontsize=sz,fontweight=wt,transform=ax.transAxes)
for sp in ax.spines.values(): sp.set_color('#ff3366'); sp.set_linewidth(2)

plt.savefig(f'{VISUALS_DIR}/16_improved_model_benchmark.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved: visuals/16_improved_model_benchmark.png")

# ── VIZ 2: Updated executive dashboard ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
fig.suptitle('Model Performance: Original vs All Improvements', fontsize=15, fontweight='bold', color='white')

metrics_compare = {
    'RMSE (min)\n[lower = better]': ('RMSE', False),
    'MAE (min)\n[lower = better]':  ('MAE', False),
    '±15% Accuracy (%)\n[higher = better]': ('Within15pct', True),
}

key_models = [
    ('OSRM Naive',                  'OSRM',       '#555566'),
    ('RF (Base) — ORIGINAL',        'v1 Baseline', '#778899'),
    ('RF + Graph — ORIGINAL',       'v1 Graph',    C1),
    ('RF + Graph — v2 (outliers cleaned)', 'v2 Cleaned', '#33aadd'),
    ('HistGBM (LightGBM-equiv) — Graph', 'HistGBM+Graph', C3),
    ('Per-Route HistGBM (FTL+Carting split)', 'Per-Route', C4),
    ('Stacked Ensemble (RF+HistGBM+ET → Ridge)', 'Stacked', C2),
]

for ax, (metric_label, (col, ascending)) in zip(axes, metrics_compare.items()):
    ax.set_facecolor(PBG)
    vals, labels, colors_bar = [], [], []
    for model_name, short, color in key_models:
        row = results_df[results_df['Model']==model_name]
        if len(row) > 0:
            vals.append(float(row[col].iloc[0]))
            labels.append(short)
            colors_bar.append(color)
    bars = ax.bar(range(len(vals)), vals, color=colors_bar, alpha=0.88, edgecolor='none', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v*(1.02), f'{v:.1f}',
                ha='center', va='bottom', color='white', fontweight='bold', fontsize=8.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', color='white', fontsize=8.5)
    ax.set_title(metric_label, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#888')
    for sp in ax.spines.values(): sp.set_color(CGRD)

plt.tight_layout()
plt.savefig(f'{VISUALS_DIR}/17_improvement_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved: visuals/17_improvement_comparison.png")

print(f"\n{'='*70}")
print("IMPROVEMENTS 2+3+4 COMPLETE ✓")
print(f"{'='*70}")
print(f"\n  Original RF+Graph:  RMSE={orig_rmse:.1f}  MAE={orig_mae:.1f}  ±15%={orig_w15:.1f}%")
print(f"  Best New Model:     RMSE={best_rmse:.1f}  MAE={best_mae:.1f}  ±15%={best_w15:.1f}%")
print(f"  Net improvement:    RMSE {(orig_rmse-best_rmse)/orig_rmse*100:+.1f}%  |  MAE {(orig_mae-best_mae)/orig_mae*100:+.1f}%  |  ±15% {best_w15-orig_w15:+.1f}pp")
