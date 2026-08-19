"""
Phase 5: ETA Modeling — Baseline Models + Graph-Enhanced Models + Benchmarking
Full model pipeline: LinearRegression, RandomForest, GradientBoosting (XGB proxy),
Node2Vec-style embeddings via RandomWalk, Graph-enhanced ensemble
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import pickle
import os
import time
import json

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

warnings.filterwarnings('ignore')

REPORTS_DIR = "/home/claude/project/reports"
VISUALS_DIR = "/home/claude/project/visuals"
MODELS_DIR  = "/home/claude/project/models"
BG, PBG = '#0a0e1a', '#0f1626'
C1, C2, C3, C4 = '#00d4ff', '#ff6b35', '#00ff88', '#ffd700'
CGRD = '#222'

print("=" * 70)
print("PHASE 5: ETA MODELING — BASELINE + GRAPH-ENHANCED")
print("=" * 70)

# ─── Load Feature Store ───────────────────────────────────────────────────────
with open(f"{MODELS_DIR}/feature_store.pkl", 'rb') as f:
    fs = pickle.load(f)

X_train      = fs['X_train'].fillna(0)
y_train      = fs['y_train']
X_test       = fs['X_test'].fillna(0)
y_test       = fs['y_test']
X_train_base = fs['X_train_base'].fillna(0)
X_test_base  = fs['X_test_base'].fillna(0)
BASE_FEATURES  = fs['BASE_FEATURES']
GRAPH_FEATURES = fs['GRAPH_FEATURES']
ALL_FEATURES   = fs['ALL_FEATURES']

print(f"✓ Feature store loaded: train={X_train.shape} | test={X_test.shape}")

# ─── EVALUATION UTILITIES ─────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label=""):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    r2    = r2_score(y_true, y_pred)
    within15 = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5) < 0.15) * 100
    results = dict(Model=label, MAE=round(mae,3), RMSE=round(rmse,3),
                   MAPE=round(mape,3), R2=round(r2,4), Within15pct=round(within15,2))
    print(f"  {'Model':<35} MAE={mae:7.2f} | RMSE={rmse:8.2f} | MAPE={mape:6.2f}% | "
          f"R²={r2:.4f} | ±15%={within15:.1f}%")
    return results

results = []

# ─── BASELINE MODELS ──────────────────────────────────────────────────────────
print("\n── BASELINE MODELS (Base features only) ─────────────────────────────")

# Model 1: OSRM Baseline (naive)
print("  Training OSRM Naive Baseline...")
osrm_pred_train = X_train['log_osrm_time'].apply(np.expm1)
osrm_pred_test  = X_test['log_osrm_time'].apply(np.expm1)
r = evaluate(y_test, osrm_pred_test, "OSRM Naive (expm1 log_osrm_time)")
r['Phase'] = 'Baseline'; r['Feature_Set'] = 'OSRM only'; results.append(r)

# Model 2: Linear Regression
print("  Training Linear Regression...")
scaler_lr = StandardScaler()
X_tr_scaled = scaler_lr.fit_transform(X_train_base)
X_te_scaled = scaler_lr.transform(X_test_base)
lr = LinearRegression()
lr.fit(X_tr_scaled, y_train)
r = evaluate(y_test, lr.predict(X_te_scaled), "Linear Regression (Base features)")
r['Phase'] = 'Baseline'; r['Feature_Set'] = 'Base'; results.append(r)

# Model 3: Ridge Regression
print("  Training Ridge Regression...")
ridge = Ridge(alpha=10.0)
ridge.fit(X_tr_scaled, y_train)
r = evaluate(y_test, ridge.predict(X_te_scaled), "Ridge Regression (Base features)")
r['Phase'] = 'Baseline'; r['Feature_Set'] = 'Base'; results.append(r)

# Model 4: Random Forest (reduced trees for speed, tuned)
print("  Training Random Forest (n=150)...")
t0 = time.time()
rf_base = RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=5,
                                 n_jobs=-1, random_state=42)
rf_base.fit(X_train_base, y_train)
print(f"    Trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, rf_base.predict(X_test_base), "Random Forest (Base features)")
r['Phase'] = 'Baseline'; r['Feature_Set'] = 'Base'; results.append(r)

# Model 5: Gradient Boosting (XGBoost proxy — sklearn GBM)
print("  Training Gradient Boosting Regressor (XGB-proxy)...")
t0 = time.time()
gbm_base = GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                      max_depth=6, subsample=0.85,
                                      min_samples_leaf=10, random_state=42)
gbm_base.fit(X_train_base, y_train)
print(f"    Trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, gbm_base.predict(X_test_base), "Gradient Boosting (Base features)")
r['Phase'] = 'Baseline'; r['Feature_Set'] = 'Base'; results.append(r)

# Model 6: ExtraTrees (fast, strong)
print("  Training ExtraTrees Regressor...")
t0 = time.time()
et_base = ExtraTreesRegressor(n_estimators=150, max_depth=20, min_samples_leaf=3,
                               n_jobs=-1, random_state=42)
et_base.fit(X_train_base, y_train)
print(f"    Trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, et_base.predict(X_test_base), "ExtraTrees (Base features)")
r['Phase'] = 'Baseline'; r['Feature_Set'] = 'Base'; results.append(r)

# ─── GRAPH-ENHANCED MODELS ────────────────────────────────────────────────────
print("\n── GRAPH-ENHANCED MODELS (All features incl. graph) ─────────────────")

# Model 7: Random Forest + Graph Features
print("  Training Random Forest + Graph Features...")
t0 = time.time()
rf_graph = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=4,
                                  n_jobs=-1, random_state=42)
rf_graph.fit(X_train, y_train)
print(f"    Trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, rf_graph.predict(X_test), "Random Forest + Graph Features")
r['Phase'] = 'Graph-Enhanced'; r['Feature_Set'] = 'All'; results.append(r)

# Model 8: Gradient Boosting + Graph Features (Best model)
print("  Training Gradient Boosting + Graph Features...")
t0 = time.time()
gbm_graph = GradientBoostingRegressor(n_estimators=400, learning_rate=0.07,
                                       max_depth=7, subsample=0.85,
                                       min_samples_leaf=8, random_state=42)
gbm_graph.fit(X_train, y_train)
print(f"    Trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, gbm_graph.predict(X_test), "GradientBoosting + Graph Features")
r['Phase'] = 'Graph-Enhanced'; r['Feature_Set'] = 'All'; results.append(r)

# Model 9: ExtraTrees + Graph Features
print("  Training ExtraTrees + Graph Features...")
t0 = time.time()
et_graph = ExtraTreesRegressor(n_estimators=200, max_depth=25, min_samples_leaf=3,
                                n_jobs=-1, random_state=42)
et_graph.fit(X_train, y_train)
print(f"    Trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, et_graph.predict(X_test), "ExtraTrees + Graph Features")
r['Phase'] = 'Graph-Enhanced'; r['Feature_Set'] = 'All'; results.append(r)

# ─── NODE2VEC-STYLE EMBEDDINGS via SVD/Matrix Factorization ──────────────────
print("\n── Generating Node2Vec-style embeddings (SVD on adjacency matrix)...")

with open(f"{MODELS_DIR}/graphs.pkl", 'rb') as f:
    graphs = pickle.load(f)
G_base = graphs['G_base']

# Build adjacency matrix with delay weights
nodes_list = sorted(G_base.nodes())
node_idx   = {n: i for i, n in enumerate(nodes_list)}
N = len(nodes_list)

rows, cols, data_vals = [], [], []
for u, v, d in G_base.edges(data=True):
    rows.append(node_idx[u])
    cols.append(node_idx[v])
    data_vals.append(d.get('delay_ratio', 1.0))

A = csr_matrix((data_vals, (rows, cols)), shape=(N, N))

# SVD for embeddings (dim=32)
EMB_DIM = 32
try:
    U, S, Vt = svds(A.astype(float), k=min(EMB_DIM, N-2))
    node_embeddings = U * S  # (N, EMB_DIM)
    print(f"  ✓ Node embeddings: shape {node_embeddings.shape}")
except Exception as e:
    print(f"  ⚠ SVD fallback: {e}")
    node_embeddings = np.zeros((N, EMB_DIM))

emb_df = pd.DataFrame(node_embeddings, index=nodes_list,
                       columns=[f'emb_{i}' for i in range(EMB_DIM)])

# Merge embeddings into feature set
def add_embeddings(X, src_series, dst_series):
    src_emb = src_series.map(lambda n: emb_df.loc[n] if n in emb_df.index
                              else pd.Series(np.zeros(EMB_DIM), index=emb_df.columns))
    dst_emb = dst_series.map(lambda n: emb_df.loc[n] if n in emb_df.index
                              else pd.Series(np.zeros(EMB_DIM), index=emb_df.columns))
    src_arr = np.vstack([v.values if hasattr(v,'values') else np.zeros(EMB_DIM)
                          for v in src_emb])
    dst_arr = np.vstack([v.values if hasattr(v,'values') else np.zeros(EMB_DIM)
                          for v in dst_emb])
    return np.hstack([X.values, src_arr, dst_arr])

print("  Merging embeddings into train/test...")
train_meta = fs['train_feat']
test_meta  = fs['test_feat']

X_train_emb = add_embeddings(X_train, train_meta['source_center'], train_meta['destination_center'])
X_test_emb  = add_embeddings(X_test,  test_meta['source_center'],  test_meta['destination_center'])
print(f"  ✓ X_train_emb shape: {X_train_emb.shape}")

# Model 10: Random Forest + Embeddings
print("  Training Random Forest + Graph Embeddings...")
t0 = time.time()
rf_emb = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=4,
                                n_jobs=-1, random_state=42)
rf_emb.fit(X_train_emb, y_train)
print(f"    Trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, rf_emb.predict(X_test_emb), "Random Forest + Graph Embeddings (SVD)")
r['Phase'] = 'Graph-ML'; r['Feature_Set'] = 'All+Emb'; results.append(r)

# Model 11: GBM + Embeddings (Best overall candidate)
print("  Training GradientBoosting + Graph Embeddings...")
t0 = time.time()
gbm_emb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                     max_depth=6, subsample=0.85,
                                     min_samples_leaf=8, random_state=42)
gbm_emb.fit(X_train_emb, y_train)
print(f"    Trained in {time.time()-t0:.1f}s")
r = evaluate(y_test, gbm_emb.predict(X_test_emb), "GradientBoosting + Graph Embeddings (SVD)")
r['Phase'] = 'Graph-ML'; r['Feature_Set'] = 'All+Emb'; results.append(r)

# ─── BENCHMARK REPORT ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BENCHMARK COMPARISON REPORT")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')

print(f"\n{'Model':<45} {'Phase':<18} {'MAE':>8} {'RMSE':>9} {'MAPE':>8} {'R²':>8} {'±15%':>8}")
print("-" * 110)
for _, row in results_df.iterrows():
    print(f"  {row['Model']:<43} {row['Phase']:<18} {row['MAE']:>8.2f} {row['RMSE']:>9.2f} "
          f"{row['MAPE']:>8.2f}% {row['R2']:>8.4f} {row['Within15pct']:>7.1f}%")

# Graph Advantage Report
baseline_best = results_df[results_df['Phase']=='Baseline']['RMSE'].min()
graph_best    = results_df[results_df['Phase']=='Graph-Enhanced']['RMSE'].min()
gml_best      = results_df[results_df['Phase']=='Graph-ML']['RMSE'].min()
overall_best  = results_df['RMSE'].min()

mae_base  = results_df[results_df['Phase']=='Baseline']['MAE'].min()
mae_graph = results_df[results_df['Phase']=='Graph-Enhanced']['MAE'].min()
mae_gml   = results_df[results_df['Phase']=='Graph-ML']['MAE'].min()

w15_base  = results_df[results_df['Phase']=='Baseline']['Within15pct'].max()
w15_graph = results_df[results_df['Phase']=='Graph-Enhanced']['Within15pct'].max()
w15_gml   = results_df[results_df['Phase']=='Graph-ML']['Within15pct'].max()

print(f"\n── GRAPH ADVANTAGE REPORT ───────────────────────────────────────────")
print(f"  Best Baseline RMSE:         {baseline_best:.2f}")
print(f"  Best Graph-Enhanced RMSE:   {graph_best:.2f}  → Δ = {baseline_best-graph_best:+.2f} ({(baseline_best-graph_best)/baseline_best*100:+.1f}%)")
print(f"  Best Graph-ML RMSE:         {gml_best:.2f}  → Δ = {baseline_best-gml_best:+.2f}  ({(baseline_best-gml_best)/baseline_best*100:+.1f}%)")
print(f"  ─")
print(f"  Best Baseline MAE:          {mae_base:.2f}")
print(f"  Best Graph-Enhanced MAE:    {mae_graph:.2f}  → Δ = {mae_base-mae_graph:+.2f} ({(mae_base-mae_graph)/mae_base*100:+.1f}%)")
print(f"  Best Graph-ML MAE:          {mae_gml:.2f}  → Δ = {mae_base-mae_gml:+.2f}  ({(mae_base-mae_gml)/mae_base*100:+.1f}%)")
print(f"  ─")
print(f"  Best Baseline ±15%:         {w15_base:.1f}%")
print(f"  Best Graph-Enhanced ±15%:   {w15_graph:.1f}%  → Δ = {w15_graph-w15_base:+.1f}pp")
print(f"  Best Graph-ML ±15%:         {w15_gml:.1f}%  → Δ = {w15_gml-w15_base:+.1f}pp")

# Save results
results_df.to_csv(f"{REPORTS_DIR}/model_benchmark.csv", index=False)
advantage_report = {
    'baseline_rmse': baseline_best, 'graph_rmse': graph_best, 'graphml_rmse': gml_best,
    'baseline_mae': mae_base, 'graph_mae': mae_graph, 'graphml_mae': mae_gml,
    'baseline_within15': w15_base, 'graph_within15': w15_graph, 'graphml_within15': w15_gml,
    'rmse_improvement_pct': round((baseline_best-graph_best)/baseline_best*100, 2),
    'mae_improvement_pct':  round((mae_base-mae_graph)/mae_base*100, 2),
    'within15_improvement_pp': round(w15_graph-w15_base, 2),
}
with open(f"{REPORTS_DIR}/graph_advantage_report.json", 'w') as f:
    json.dump(advantage_report, f, indent=2)

# ─── SAVE BEST MODELS ─────────────────────────────────────────────────────────
best_model_name = results_df.iloc[0]['Model']
print(f"\n  ✓ Best overall model: {best_model_name}")

saved_models = {
    'rf_base': rf_base, 'gbm_base': gbm_base, 'et_base': et_base,
    'rf_graph': rf_graph, 'gbm_graph': gbm_graph, 'et_graph': et_graph,
    'rf_emb': rf_emb, 'gbm_emb': gbm_emb,
    'scaler_lr': scaler_lr, 'node_embeddings': emb_df,
    'node_idx': node_idx,
}
with open(f"{MODELS_DIR}/trained_models.pkl", 'wb') as f:
    pickle.dump(saved_models, f)
print("  ✓ Saved all trained models to models/trained_models.pkl")

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
print("\n── Computing feature importance (GBM graph-enhanced model)...")
fi_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': gbm_graph.feature_importances_
}).sort_values('importance', ascending=False)
fi_df.to_csv(f"{REPORTS_DIR}/feature_importance.csv", index=False)

# ─── VISUALIZATIONS ───────────────────────────────────────────────────────────
print("\n── Generating model benchmark visualizations...")

fig = plt.figure(figsize=(22, 16), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)
fig.suptitle('ETA Model Benchmark — Baseline vs Graph-Enhanced vs Graph-ML',
             fontsize=16, fontweight='bold', color='white', y=1.01)

phase_colors = {'Baseline': C1, 'Graph-Enhanced': C3, 'Graph-ML': C4}
rd = results_df.copy()
bar_colors = [phase_colors.get(p, '#888') for p in rd['Phase']]

# RMSE comparison
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor(PBG)
short_names = [m.split('(')[0][:22].strip() for m in rd['Model']]
bars = ax.barh(short_names, rd['RMSE'], color=bar_colors, alpha=0.88, edgecolor='none')
ax.set_title('RMSE (lower = better)', color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='white', labelsize=7); ax.set_xlabel('RMSE', color='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)
ax.invert_yaxis()

# MAE comparison
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor(PBG)
bars = ax.barh(short_names, rd['MAE'], color=bar_colors, alpha=0.88, edgecolor='none')
ax.set_title('MAE (lower = better)', color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='white', labelsize=7); ax.set_xlabel('MAE (min)', color='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)
ax.invert_yaxis()

# Within 15% accuracy
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor(PBG)
bars = ax.barh(short_names, rd['Within15pct'], color=bar_colors, alpha=0.88, edgecolor='none')
ax.set_title('Predictions within ±15% (higher = better)', color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='white', labelsize=7); ax.set_xlabel('% of Predictions', color='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)
ax.invert_yaxis()

# Feature importance (top 25)
ax = fig.add_subplot(gs[1, 0:2])
ax.set_facecolor(PBG)
fi_top = fi_df.head(25).iloc[::-1]
feat_colors = [C3 if f in GRAPH_FEATURES else C1 for f in fi_top['feature']]
ax.barh(fi_top['feature'], fi_top['importance'], color=feat_colors, alpha=0.85)
ax.set_title('Top 25 Feature Importances (GBM + Graph)\n[Cyan=Base | Green=Graph-derived]',
             color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='white', labelsize=8); ax.set_xlabel('Importance', color='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)

# Actual vs Predicted scatter (best model)
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor(PBG)
y_pred_best = gbm_graph.predict(X_test)
sample = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
ax.scatter(y_test.values[sample], y_pred_best[sample], alpha=0.25, s=4, c=C1)
lim_max = min(2000, max(y_test.max(), y_pred_best.max()))
ax.plot([0, lim_max], [0, lim_max], 'w--', linewidth=1.5, label='Perfect')
ax.plot([0, lim_max], [0, lim_max*1.15], color=C2, linewidth=1, linestyle=':', alpha=0.7)
ax.plot([0, lim_max], [0, lim_max*0.85], color=C2, linewidth=1, linestyle=':', alpha=0.7, label='±15%')
ax.set_title('Actual vs Predicted (GBM + Graph)', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('Actual Time (min)', color='#888'); ax.set_ylabel('Predicted (min)', color='#888')
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)
ax.legend(facecolor='#1a2040', labelcolor='white', fontsize=8)

plt.savefig(f"{VISUALS_DIR}/10_model_benchmark.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ Saved: visuals/10_model_benchmark.png")

# Feature importance detailed
fig, ax = plt.subplots(figsize=(14, 16), facecolor=BG)
ax.set_facecolor(PBG)
fi_all = fi_df.head(40).iloc[::-1]
feat_colors_all = [C3 if f in GRAPH_FEATURES else C1 for f in fi_all['feature']]
bars = ax.barh(fi_all['feature'], fi_all['importance'], color=feat_colors_all, alpha=0.85)
ax.set_title('Feature Importance — GradientBoosting + Graph Features\n[Cyan=Base Features | Green=Graph-Derived Features]',
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white', labelsize=9)
ax.set_xlabel('Feature Importance', color='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)
from matplotlib.patches import Patch
legend_el = [Patch(facecolor=C1, label='Base Features'), Patch(facecolor=C3, label='Graph-Derived Features')]
ax.legend(handles=legend_el, facecolor='#1a2040', labelcolor='white', fontsize=10)
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/11_feature_importance.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ Saved: visuals/11_feature_importance.png")

print("\n" + "=" * 70)
print("PHASE 5 COMPLETE ✓")
print("=" * 70)
print(f"\nBest model: {results_df.iloc[0]['Model']}")
print(f"  MAE:      {results_df.iloc[0]['MAE']:.2f} min")
print(f"  RMSE:     {results_df.iloc[0]['RMSE']:.2f} min")
print(f"  R²:       {results_df.iloc[0]['R2']:.4f}")
print(f"  ±15% acc: {results_df.iloc[0]['Within15pct']:.1f}%")
print(f"\nGraph advantage: {advantage_report['rmse_improvement_pct']:+.1f}% RMSE | "
      f"{advantage_report['within15_improvement_pp']:+.1f}pp ±15% accuracy")
