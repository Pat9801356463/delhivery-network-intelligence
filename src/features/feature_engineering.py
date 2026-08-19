"""
Phase 4: Delay Corridor Audit + Feature Engineering
Top delayed corridors, bottleneck analysis, full feature store creation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

REPORTS_DIR = "/home/claude/project/reports"
VISUALS_DIR = "/home/claude/project/visuals"
MODELS_DIR  = "/home/claude/project/models"

print("=" * 70)
print("PHASE 4: DELAY CORRIDOR AUDIT + FEATURE ENGINEERING")
print("=" * 70)

BG, PBG = '#0a0e1a', '#0f1626'
C1, C2, C3, C4 = '#00d4ff', '#ff6b35', '#00ff88', '#ffd700'

# ─── Load Data ────────────────────────────────────────────────────────────────
df         = pd.read_csv(f"{REPORTS_DIR}/full_clean.csv")
corridor   = pd.read_csv(f"{REPORTS_DIR}/corridor_stats_enriched.csv")
node_feat  = pd.read_csv(f"{REPORTS_DIR}/node_features.csv")

df['od_start_time'] = pd.to_datetime(df['od_start_time'], errors='coerce')
df['delay_ratio']   = df['actual_time'] / df['osrm_time'].replace(0, np.nan)
df['is_sla_breach'] = df['is_cutoff'].astype(bool)
df['hour']          = df['od_start_time'].dt.hour
df['dow']           = df['od_start_time'].dt.dayofweek
df['month']         = df['od_start_time'].dt.month
df['is_weekend']    = df['dow'].isin([5, 6]).astype(int)
df['is_rush_hour']  = df['hour'].isin([7,8,9,17,18,19]).astype(int)

print(f"✓ Loaded {len(df):,} rows | {len(corridor):,} corridors | {len(node_feat):,} facilities")

# ─── PART A: DELAY CORRIDOR AUDIT ────────────────────────────────────────────

# Top 50 delayed corridors (min 5 trips, delay > 20%)
delayed_corridors = corridor[
    (corridor['median_delay_ratio'] > 1.20) & (corridor['trip_count'] >= 5)
].copy()

delayed_corridors['delay_pct_over_osrm'] = (delayed_corridors['median_delay_ratio'] - 1) * 100
delayed_corridors['revenue_risk_proxy']  = (
    delayed_corridors['trip_count'] * delayed_corridors['sla_breach_rate']
)
delayed_corridors['corridor_label'] = (
    delayed_corridors['source_center'].str[-8:] + " → " +
    delayed_corridors['destination_center'].str[-8:]
)

top50_delayed = delayed_corridors.nlargest(50, 'delay_pct_over_osrm')
top50_delayed.to_csv(f"{REPORTS_DIR}/top50_delayed_corridors.csv", index=False)
print(f"\n── TOP 50 DELAYED CORRIDORS ─────────────────────────────────────────")
print(f"  Total corridors with >20% delay: {len(delayed_corridors):,} / {len(corridor):,} "
      f"({len(delayed_corridors)/len(corridor)*100:.1f}%)")
print(f"  Top 50 saved to reports/top50_delayed_corridors.csv")

print(f"\n  {'Rank':<5} {'Corridor':<35} {'Delay%':<12} {'Trips':<8} {'SLA Breach%':<13} {'Route Type'}")
print("  " + "-"*80)
for rank, (_, row) in enumerate(top50_delayed.head(20).iterrows(), 1):
    label   = str(row['corridor_label'])[:33]
    delay   = f"+{row['delay_pct_over_osrm']:.0f}%"
    trips   = int(row['trip_count'])
    sla     = f"{row['sla_breach_rate']*100:.0f}%"
    ftl_sh  = "FTL" if row.get('ftl_share',0) > 0.5 else "Carting"
    print(f"  {rank:<5} {label:<35} {delay:<12} {trips:<8} {sla:<13} {ftl_sh}")

# ─── CORRIDOR CLUSTERING ──────────────────────────────────────────────────────
print("\n── Clustering delayed corridors by distance & delay profile...")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cluster_features = delayed_corridors[['median_delay_ratio','trip_count',
                                       'sla_breach_rate','median_distance',
                                       'ftl_share','route_type_entropy']].fillna(0)
scaler = StandardScaler()
X_clust = scaler.fit_transform(cluster_features)

# K=5 clusters for interpretability
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
delayed_corridors['cluster'] = kmeans.fit_predict(X_clust)

# Characterize clusters
cluster_summary = delayed_corridors.groupby('cluster').agg(
    corridor_count     = ('source_center','count'),
    avg_delay_ratio    = ('median_delay_ratio','mean'),
    avg_sla_breach     = ('sla_breach_rate','mean'),
    avg_distance_km    = ('median_distance','mean'),
    avg_trips          = ('trip_count','mean'),
    ftl_dominance      = ('ftl_share','mean'),
).round(3)

cluster_names = {
    cluster_summary['avg_delay_ratio'].idxmax(): 'Extreme Delay',
    cluster_summary['avg_trips'].idxmax(): 'High-Volume Risk',
    cluster_summary['avg_sla_breach'].idxmax(): 'SLA Critical',
    cluster_summary['avg_distance_km'].idxmax(): 'Long-Haul Delay',
}
remaining = [c for c in range(5) if c not in cluster_names]
for r in remaining:
    cluster_names[r] = f'Moderate Risk (C{r})'

cluster_summary['cluster_name'] = [cluster_names.get(i, f'Cluster {i}') for i in cluster_summary.index]
print("\n  Corridor Cluster Profiles:")
print(cluster_summary.to_string())

# ─── PART B: FEATURE ENGINEERING ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("FEATURE ENGINEERING — Building Production Feature Store")
print("=" * 70)

# Node centrality lookup maps
node_map = node_feat.set_index('facility')

def safe_map(series, lookup, col):
    return series.map(lookup[col].to_dict()).fillna(lookup[col].median())

# ── Trip-level features ───────────────────────────────────────────────────────
print("\n── Building trip-level features...")

feat = df.copy()

# Basic features (no leakage)
feat['route_type_enc']    = (feat['route_type'] == 'FTL').astype(int)
feat['log_osrm_time']     = np.log1p(feat['osrm_time'])
feat['log_osrm_dist']     = np.log1p(feat['osrm_distance'])
feat['log_distance']      = np.log1p(feat['actual_distance_to_destination'])
feat['osrm_speed']        = feat['osrm_distance'] / (feat['osrm_time'] / 60 + 1e-5)
feat['dist_time_ratio']   = feat['actual_distance_to_destination'] / (feat['osrm_time'] + 1)

# Temporal features
feat['hour_sin']   = np.sin(2 * np.pi * feat['hour'] / 24)
feat['hour_cos']   = np.cos(2 * np.pi * feat['hour'] / 24)
feat['dow_sin']    = np.sin(2 * np.pi * feat['dow'] / 7)
feat['dow_cos']    = np.cos(2 * np.pi * feat['dow'] / 7)
feat['month_sin']  = np.sin(2 * np.pi * feat['month'] / 12)
feat['month_cos']  = np.cos(2 * np.pi * feat['month'] / 12)

# ── Source node graph features ─────────────────────────────────────────────────
print("── Merging source node graph features...")
src_cols = ['betweenness','pagerank','closeness','hub_score','structural_risk_score',
            'in_degree','out_degree','total_degree','community_id',
            'outbound_trips','avg_sla_breach','avg_delay_ratio']
for col in src_cols:
    if col in node_map.columns:
        feat[f'src_{col}'] = safe_map(feat['source_center'], node_map, col)

# ── Destination node graph features ───────────────────────────────────────────
print("── Merging destination node graph features...")
for col in src_cols:
    if col in node_map.columns:
        feat[f'dst_{col}'] = safe_map(feat['destination_center'], node_map, col)

# ── Corridor-level features ────────────────────────────────────────────────────
print("── Merging corridor-level features...")
corr_map = corridor.set_index(['source_center','destination_center'])

corridor_lookup_cols = ['median_delay_ratio','sla_breach_rate','chronic_delay_rate',
                         'trip_count','ftl_share','route_type_entropy','cross_community',
                         'corridor_risk','src_betweenness','dst_betweenness']

for col in corridor_lookup_cols:
    if col in corr_map.columns:
        feat[f'corr_{col}'] = feat.apply(
            lambda r: corr_map.loc[(r['source_center'], r['destination_center']), col]
            if (r['source_center'], r['destination_center']) in corr_map.index else np.nan,
            axis=1
        )

# Fill missing corridors with global median
for col in [c for c in feat.columns if c.startswith('corr_')]:
    feat[col] = feat[col].fillna(feat[col].median())

# ── Network congestion proxy ───────────────────────────────────────────────────
print("── Computing network congestion proxy...")
hourly_volume = df.groupby('hour')['trip_uuid'].count().to_dict()
feat['network_load_hour'] = feat['hour'].map(hourly_volume).fillna(0)
feat['network_load_norm'] = feat['network_load_hour'] / feat['network_load_hour'].max()

# ── Cross-community flag ───────────────────────────────────────────────────────
feat['cross_community_flag'] = feat.get('corr_cross_community', 0).astype(int)

# ── Derived risk flags ─────────────────────────────────────────────────────────
feat['high_risk_src']     = (feat.get('src_structural_risk_score', 0) >
                              feat.get('src_structural_risk_score', 0).quantile(0.75)).astype(int)
feat['high_risk_dst']     = (feat.get('dst_structural_risk_score', 0) >
                              feat.get('dst_structural_risk_score', 0).quantile(0.75)).astype(int)
feat['both_high_risk']    = (feat['high_risk_src'] & feat['high_risk_dst']).astype(int)

print(f"  ✓ Feature set complete")

# ─── DEFINE FINAL FEATURE COLUMNS ─────────────────────────────────────────────
BASE_FEATURES = [
    'route_type_enc', 'log_osrm_time', 'log_osrm_dist', 'log_distance',
    'osrm_speed', 'dist_time_ratio',
    'hour', 'dow', 'month', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'network_load_norm', 'cutoff_factor',
]

GRAPH_FEATURES = [
    'src_betweenness', 'src_pagerank', 'src_closeness', 'src_structural_risk_score',
    'src_in_degree', 'src_out_degree', 'src_avg_sla_breach', 'src_avg_delay_ratio',
    'src_community_id', 'src_hub_score',
    'dst_betweenness', 'dst_pagerank', 'dst_closeness', 'dst_structural_risk_score',
    'dst_in_degree', 'dst_out_degree', 'dst_avg_sla_breach', 'dst_avg_delay_ratio',
    'dst_community_id',
    'corr_median_delay_ratio', 'corr_sla_breach_rate', 'corr_trip_count',
    'corr_ftl_share', 'corr_route_type_entropy', 'corr_cross_community',
    'corr_corridor_risk', 'cross_community_flag',
    'high_risk_src', 'high_risk_dst', 'both_high_risk',
]

# Only keep features that actually exist in the dataframe
BASE_FEATURES  = [f for f in BASE_FEATURES  if f in feat.columns]
GRAPH_FEATURES = [f for f in GRAPH_FEATURES if f in feat.columns]
ALL_FEATURES   = BASE_FEATURES + GRAPH_FEATURES

TARGET = 'actual_time'

print(f"\n  ✓ Base features:  {len(BASE_FEATURES)}")
print(f"  ✓ Graph features: {len(GRAPH_FEATURES)}")
print(f"  ✓ Total features: {len(ALL_FEATURES)}")

# ─── SPLIT TRAIN / TEST ───────────────────────────────────────────────────────
train_feat = feat[feat['data'] == 'training'].copy()
test_feat  = feat[feat['data'] == 'test'].copy()

X_train      = train_feat[ALL_FEATURES].fillna(0)
y_train      = train_feat[TARGET]
X_train_base = train_feat[BASE_FEATURES].fillna(0)
X_test       = test_feat[ALL_FEATURES].fillna(0)
y_test       = test_feat[TARGET]
X_test_base  = test_feat[BASE_FEATURES].fillna(0)

print(f"\n  ✓ X_train shape: {X_train.shape}")
print(f"  ✓ X_test  shape: {X_test.shape}")

# ─── SAVE FEATURE STORE ───────────────────────────────────────────────────────
feature_store = {
    'X_train': X_train, 'y_train': y_train,
    'X_test':  X_test,  'y_test':  y_test,
    'X_train_base': X_train_base,
    'X_test_base':  X_test_base,
    'BASE_FEATURES':  BASE_FEATURES,
    'GRAPH_FEATURES': GRAPH_FEATURES,
    'ALL_FEATURES':   ALL_FEATURES,
    'train_feat': train_feat,
    'test_feat':  test_feat,
}
with open(f"{MODELS_DIR}/feature_store.pkl", 'wb') as f:
    pickle.dump(feature_store, f)

print(f"  ✓ Saved feature_store.pkl")

# ─── VISUALIZATIONS ───────────────────────────────────────────────────────────
print("\n── Generating delay corridor visualizations...")

# VIZ: Top 20 delayed corridors
fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor=BG)

ax = axes[0]
ax.set_facecolor(PBG)
top20_plot = top50_delayed.head(20).copy().iloc[::-1]
delay_vals = top20_plot['delay_pct_over_osrm'].values
colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.95, len(delay_vals)))
bars = ax.barh(range(len(top20_plot)), delay_vals, color=colors_bar, edgecolor='none', alpha=0.9)
ax.set_yticks(range(len(top20_plot)))
ax.set_yticklabels([str(l)[:32] for l in top20_plot['corridor_label'].values],
                    fontsize=7.5, color='white')
ax.set_xlabel('Delay % Over OSRM Estimate', color='#888', fontsize=11)
ax.set_title('Top 20 Most Delayed Corridors\n(Median Delay vs OSRM)',
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color(CGRD) if (CGRD:='#222') else None
ax.axvline(20, color=C3, linestyle='--', linewidth=1.5, alpha=0.7, label='20% threshold')
ax.legend(facecolor='#1a2040', labelcolor='white')

ax = axes[1]
ax.set_facecolor(PBG)
scatter = ax.scatter(
    delayed_corridors['median_distance'],
    delayed_corridors['delay_pct_over_osrm'].clip(0, 1500),
    c=delayed_corridors['sla_breach_rate'],
    cmap='RdYlGn_r', alpha=0.6, s=delayed_corridors['trip_count'].clip(0,500)/3 + 5,
    vmin=0, vmax=1
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('SLA Breach Rate', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
ax.set_xlabel('Median Distance (km)', color='#888', fontsize=11)
ax.set_ylabel('Delay % Over OSRM', color='#888', fontsize=11)
ax.set_title('Corridor Delay vs Distance\n(Bubble size = Trip volume | Color = SLA breach rate)',
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color('#222')

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/08_delay_corridor_audit.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ Saved: visuals/08_delay_corridor_audit.png")

# VIZ: Feature correlation heatmap
fig, ax = plt.subplots(figsize=(16, 14), facecolor=BG)
ax.set_facecolor(PBG)
corr_cols = [f for f in ALL_FEATURES[:25] if f in X_train.columns]
corr_matrix = X_train[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, ax=ax, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=False, linewidths=0.3, cbar_kws={'label':'Correlation'})
ax.set_title('Feature Correlation Matrix (Top 25 Features)', color='white',
             fontsize=13, fontweight='bold')
ax.tick_params(colors='white', labelsize=7)
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/09_feature_correlation.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ Saved: visuals/09_feature_correlation.png")

print("\n" + "=" * 70)
print("PHASE 4 COMPLETE ✓")
print("=" * 70)
print(f"\nKey Findings:")
print(f"  • {len(delayed_corridors):,} of {len(corridor):,} corridors show >20% chronic delay")
print(f"  • 5 corridor clusters identified (extreme delay, high-volume, SLA critical, long-haul, moderate)")
print(f"  • Feature store: {len(ALL_FEATURES)} features ({len(BASE_FEATURES)} base + {len(GRAPH_FEATURES)} graph-derived)")
