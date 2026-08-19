"""
Phase 3: Advanced Graph Analytics
Centrality measures, community detection, bottleneck identification, structural risk scoring
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import pickle
import json
import os

warnings.filterwarnings('ignore')

REPORTS_DIR = "/home/claude/project/reports"
VISUALS_DIR = "/home/claude/project/visuals"
MODELS_DIR  = "/home/claude/project/models"

print("=" * 70)
print("PHASE 3: ADVANCED GRAPH ANALYTICS")
print("=" * 70)

# ─── Load graphs and data ─────────────────────────────────────────────────────
with open(f"{MODELS_DIR}/graphs.pkl", 'rb') as f:
    graphs = pickle.load(f)

G_base   = graphs['G_base']
G_delay  = graphs['G_delay']

df          = pd.read_csv(f"{REPORTS_DIR}/full_clean.csv")
corridor_df = pd.read_csv(f"{REPORTS_DIR}/corridor_stats.csv")
node_raw    = pd.read_csv(f"{REPORTS_DIR}/node_stats_raw.csv")

df['delay_ratio']   = df['actual_time'] / df['osrm_time'].replace(0, np.nan)
df['is_sla_breach'] = df['is_cutoff'].astype(bool)

print(f"\n✓ Loaded G_base: {G_base.number_of_nodes():,} nodes | {G_base.number_of_edges():,} edges")

# ─── NODE CENTRALITY METRICS ──────────────────────────────────────────────────
print("\n── Computing centrality metrics (this may take ~30 sec)...")

nodes = list(G_base.nodes())

# Degree metrics
in_deg  = dict(G_base.in_degree())
out_deg = dict(G_base.out_degree())
deg     = dict(G_base.degree())

# Weighted degree
in_deg_w  = dict(G_base.in_degree(weight='weight'))
out_deg_w = dict(G_base.out_degree(weight='weight'))

# Betweenness centrality (expensive — use sample for large graphs)
print("  Computing betweenness centrality...")
betweenness = nx.betweenness_centrality(G_base, weight='weight', normalized=True)

# PageRank
print("  Computing PageRank...")
pagerank = nx.pagerank(G_base, weight='weight', alpha=0.85, max_iter=500)

# Closeness centrality
print("  Computing closeness centrality...")
closeness = nx.closeness_centrality(G_base, distance='weight')

# HITS (hub & authority scores)
print("  Computing HITS hub/authority scores...")
try:
    hubs, authorities = nx.hits(G_base, max_iter=1000)
except:
    hubs       = {n: 0 for n in nodes}
    authorities = {n: 0 for n in nodes}

print("  ✓ All centrality metrics computed")

# ─── COMMUNITY DETECTION — Louvain via greedy modularity ─────────────────────
print("\n── Community detection (greedy modularity on undirected projection)...")
G_und = G_base.to_undirected()

# Greedy modularity community detection (available in networkx)
communities_gen = nx.algorithms.community.greedy_modularity_communities(G_und, weight='weight')
communities_list = list(communities_gen)
community_map = {}
for comm_id, comm in enumerate(communities_list):
    for node in comm:
        community_map[node] = comm_id

print(f"  ✓ {len(communities_list)} communities detected")
sizes = [len(c) for c in communities_list]
print(f"  ✓ Community sizes: min={min(sizes)} | median={int(np.median(sizes))} | max={max(sizes)}")

# ─── BUILD NODE FEATURE DATAFRAME ─────────────────────────────────────────────
print("\n── Building node feature dataframe...")

node_features = pd.DataFrame({
    'facility': nodes,
    'in_degree':       [in_deg.get(n, 0)   for n in nodes],
    'out_degree':      [out_deg.get(n, 0)  for n in nodes],
    'total_degree':    [deg.get(n, 0)      for n in nodes],
    'in_degree_w':     [in_deg_w.get(n,0)  for n in nodes],
    'out_degree_w':    [out_deg_w.get(n,0) for n in nodes],
    'betweenness':     [betweenness.get(n, 0) for n in nodes],
    'pagerank':        [pagerank.get(n, 0)    for n in nodes],
    'closeness':       [closeness.get(n, 0)   for n in nodes],
    'hub_score':       [hubs.get(n, 0)        for n in nodes],
    'authority_score': [authorities.get(n, 0) for n in nodes],
    'community_id':    [community_map.get(n, -1) for n in nodes],
})

# Merge operational stats
node_features = node_features.merge(node_raw, on='facility', how='left')
node_features = node_features.fillna(0)

# ─── STRUCTURAL RISK SCORE ────────────────────────────────────────────────────
print("\n── Computing Structural Risk Score...")

def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + 1e-10)

# Components of risk:
# 1. Centrality (betweenness) — high centrality = single point of failure
# 2. SLA breach contribution
# 3. Delay contribution
# 4. Throughput (more trips = more revenue at risk)

node_features['norm_betweenness']   = minmax(node_features['betweenness'])
node_features['norm_sla_breach']    = minmax(node_features['avg_sla_breach'])
node_features['norm_delay']         = minmax(node_features['avg_delay_ratio'])
node_features['norm_throughput']    = minmax(node_features['total_trips'])
node_features['norm_pagerank']      = minmax(node_features['pagerank'])

# Weighted composite risk score
node_features['structural_risk_score'] = (
    0.30 * node_features['norm_betweenness']   +
    0.25 * node_features['norm_sla_breach']    +
    0.20 * node_features['norm_delay']         +
    0.15 * node_features['norm_throughput']    +
    0.10 * node_features['norm_pagerank']
)

node_features['risk_rank'] = node_features['structural_risk_score'].rank(ascending=False).astype(int)
node_features = node_features.sort_values('structural_risk_score', ascending=False).reset_index(drop=True)

# ─── TOP BOTTLENECK HUBS ──────────────────────────────────────────────────────
print("\n── TOP 20 BOTTLENECK HUBS BY STRUCTURAL RISK ────────────────────────")

top20 = node_features.head(20)[['facility','structural_risk_score','betweenness',
                                  'pagerank','total_trips','avg_sla_breach',
                                  'avg_delay_ratio','community_id','in_degree','out_degree']]
top20['betweenness_pct']  = (top20['betweenness'] * 100).round(3)
top20['sla_breach_pct']   = (top20['avg_sla_breach'] * 100).round(1)
top20['delay_ratio']      = top20['avg_delay_ratio'].round(2)
top20['risk_score']       = top20['structural_risk_score'].round(4)

print(f"\n{'Rank':<5} {'Facility':<16} {'Risk Score':<12} {'Betweenness%':<14} {'Trips':<10} {'SLA Breach%':<13} {'Delay Ratio':<13} {'Community'}")
print("-" * 95)
for rank, row in enumerate(top20.itertuples(), 1):
    print(f"  {rank:<4} {row.facility:<16} {row.risk_score:<12} {row.betweenness_pct:<14} "
          f"{int(row.total_trips):<10} {row.sla_breach_pct:<13} {row.delay_ratio:<13} {row.community_id}")

# ─── CORRIDOR EDGE FEATURES ───────────────────────────────────────────────────
print("\n── Computing corridor edge features...")

# Map node centrality to corridors
cent_map = node_features.set_index('facility')[
    ['betweenness','pagerank','closeness','structural_risk_score','community_id']].to_dict()

corridor_df['src_betweenness']  = corridor_df['source_center'].map(cent_map['betweenness']).fillna(0)
corridor_df['dst_betweenness']  = corridor_df['destination_center'].map(cent_map['betweenness']).fillna(0)
corridor_df['src_pagerank']     = corridor_df['source_center'].map(cent_map['pagerank']).fillna(0)
corridor_df['dst_pagerank']     = corridor_df['destination_center'].map(cent_map['pagerank']).fillna(0)
corridor_df['src_risk']         = corridor_df['source_center'].map(cent_map['structural_risk_score']).fillna(0)
corridor_df['dst_risk']         = corridor_df['destination_center'].map(cent_map['structural_risk_score']).fillna(0)
corridor_df['src_community']    = corridor_df['source_center'].map(cent_map['community_id']).fillna(-1)
corridor_df['dst_community']    = corridor_df['destination_center'].map(cent_map['community_id']).fillna(-1)
corridor_df['cross_community']  = (corridor_df['src_community'] != corridor_df['dst_community']).astype(int)
corridor_df['corridor_risk']    = (corridor_df['src_risk'] + corridor_df['dst_risk']) / 2

# ─── SAVE OUTPUTS ─────────────────────────────────────────────────────────────
node_features.to_csv(f"{REPORTS_DIR}/node_features.csv", index=False)
corridor_df.to_csv(f"{REPORTS_DIR}/corridor_stats_enriched.csv", index=False)

with open(f"{MODELS_DIR}/node_features.pkl", 'wb') as f:
    pickle.dump(node_features, f)

print(f"\n  ✓ Saved node_features.csv ({len(node_features):,} facilities)")
print(f"  ✓ Saved corridor_stats_enriched.csv ({len(corridor_df):,} corridors)")

# ─── VISUALIZATIONS ───────────────────────────────────────────────────────────
print("\n── Generating analytics visualizations...")

BG   = '#0a0e1a'
PBG  = '#0f1626'
C1   = '#00d4ff'
C2   = '#ff6b35'
C3   = '#00ff88'
C4   = '#ffd700'
CGRD = '#222'

# ── VIZ 1: Centrality Rankings ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor=BG)
fig.suptitle('Hub Centrality Rankings — Delhivery Logistics Network',
             fontsize=16, fontweight='bold', color='white', y=1.01)

top_n = 20

metrics = [
    ('betweenness', 'Betweenness Centrality', C1),
    ('pagerank',    'PageRank Score',         C2),
    ('structural_risk_score', 'Structural Risk Score', C4),
    ('total_trips', 'Total Throughput (Trips)', C3),
]

for ax, (metric, title, color) in zip(axes.flat, metrics):
    ax.set_facecolor(PBG)
    top = node_features.nlargest(top_n, metric)[['facility', metric]].iloc[::-1]
    labels = [f[-12:] for f in top['facility']]  # Truncate
    bars = ax.barh(labels, top[metric], color=color, alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, top[metric]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}' if metric not in ['total_trips'] else f'{int(val):,}',
                va='center', color='white', fontsize=7.5, fontweight='bold')
    ax.set_title(f'Top {top_n} Hubs — {title}', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=7.5)
    ax.set_facecolor(PBG)
    for sp in ax.spines.values(): sp.set_color(CGRD)
    ax.xaxis.label.set_color('#888')

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/04_centrality_rankings.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ Saved: visuals/04_centrality_rankings.png")

# ── VIZ 2: Bottleneck Hub Network ─────────────────────────────────────────────
top5_facilities = node_features.head(5)['facility'].tolist()
top20_facilities = node_features.head(30)['facility'].tolist()

# Build ego subgraph around top bottleneck hubs
ego_nodes = set(top20_facilities)
for hub in top5_facilities:
    ego_nodes |= set(G_base.predecessors(hub))
    ego_nodes |= set(G_base.successors(hub))
ego_nodes = list(ego_nodes)[:120]  # cap for clarity

G_ego = G_base.subgraph(ego_nodes)

fig, ax = plt.subplots(figsize=(18, 14), facecolor=BG)
ax.set_facecolor(BG)
ax.set_title('Top Bottleneck Hubs — Network Neighborhood\n(Red = Top 5 Risk Hubs | Orange = Top 30)',
             fontsize=14, fontweight='bold', color='white', pad=16)

pos = nx.spring_layout(G_ego, k=1.2, seed=42)

# Color nodes by category
node_colors = []
node_sizes  = []
for n in G_ego.nodes():
    if n in top5_facilities:
        node_colors.append('#ff3366')
        node_sizes.append(800)
    elif n in top20_facilities:
        node_colors.append(C2)
        node_sizes.append(300)
    else:
        node_colors.append(C1)
        node_sizes.append(80)

edge_weights = [G_ego[u][v].get('weight', 1) for u, v in G_ego.edges()]
max_ew = max(edge_weights) if edge_weights else 1
edge_widths = [0.3 + 2.5*(w/max_ew) for w in edge_weights]

nx.draw_networkx_edges(G_ego, pos, ax=ax, alpha=0.25, edge_color='#334466',
                       width=edge_widths, arrows=True, arrowsize=8)
nx.draw_networkx_nodes(G_ego, pos, ax=ax, node_color=node_colors,
                       node_size=node_sizes, alpha=0.92)

# Labels for top facilities
labels = {n: n[-10:] for n in top20_facilities if n in G_ego.nodes()}
nx.draw_networkx_labels(G_ego, pos, labels, ax=ax, font_size=7,
                        font_color='white', font_weight='bold')

# Legend
from matplotlib.patches import Patch
legend_el = [
    Patch(facecolor='#ff3366', label='Top 5 Critical Hubs'),
    Patch(facecolor=C2,        label='Top 6-30 Risk Hubs'),
    Patch(facecolor=C1,        label='Other Facilities'),
]
ax.legend(handles=legend_el, facecolor='#1a2040', labelcolor='white',
          fontsize=10, framealpha=0.8, loc='upper left')
ax.axis('off')

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/05_bottleneck_hub_network.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ Saved: visuals/05_bottleneck_hub_network.png")

# ── VIZ 3: Community Clusters ─────────────────────────────────────────────────
# Use a manageable subgraph for community visualization
top_comm_nodes = node_features[node_features['total_trips'] > 5]['facility'].tolist()
G_comm = G_base.subgraph([n for n in top_comm_nodes if n in G_base.nodes()])
pos_comm = nx.spring_layout(G_comm, k=0.9, seed=7)

comm_ids = [community_map.get(n, 0) for n in G_comm.nodes()]
max_comm = max(comm_ids) if comm_ids else 1
norm_comm = mcolors.Normalize(vmin=0, vmax=max_comm)
cmap_comm = plt.cm.tab20

fig, ax = plt.subplots(figsize=(18, 13), facecolor=BG)
ax.set_facecolor(BG)
ax.set_title(f'Community Clusters — {len(communities_list)} Communities Detected\n(Greedy Modularity Optimization)',
             fontsize=14, fontweight='bold', color='white', pad=14)

node_colors_comm = [cmap_comm(norm_comm(c)) for c in comm_ids]
nx.draw_networkx_edges(G_comm, pos_comm, ax=ax, alpha=0.15,
                       edge_color='#334466', width=0.4)
nx.draw_networkx_nodes(G_comm, pos_comm, ax=ax,
                       node_color=node_colors_comm, node_size=60, alpha=0.9)
ax.axis('off')

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/06_community_clusters.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ Saved: visuals/06_community_clusters.png")

# ── VIZ 4: Risk Score Heatmap ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor=BG)

ax = axes[0]
ax.set_facecolor(PBG)
top25 = node_features.head(25)
risk_matrix = top25[['norm_betweenness','norm_sla_breach','norm_delay',
                       'norm_throughput','norm_pagerank']].values

im = ax.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels([f[-13:] for f in top25['facility']], fontsize=7, color='white')
ax.set_xticks(range(5))
ax.set_xticklabels(['Betweenness','SLA Breach','Delay','Throughput','PageRank'],
                    rotation=30, ha='right', fontsize=9, color='white')
ax.set_title('Risk Component Heatmap — Top 25 Facilities', color='white',
             fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized Score').ax.yaxis.set_tick_params(color='white')

ax = axes[1]
ax.set_facecolor(PBG)
top30 = node_features.head(30).iloc[::-1]
colors_risk = ['#ff3366' if i < 5 else C2 if i < 15 else C1
               for i in range(len(top30)-1, -1, -1)]
bars = ax.barh(range(len(top30)), top30['structural_risk_score'], color=colors_risk[::-1], alpha=0.88)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels([f[-13:] for f in top30['facility']], fontsize=7.5, color='white')
ax.set_title('Structural Risk Score — Top 30 Hubs', color='white', fontsize=13, fontweight='bold')
ax.set_xlabel('Risk Score', color='#888')
ax.tick_params(colors='#888')
for sp in ax.spines.values(): sp.set_color(CGRD)

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/07_risk_score_heatmap.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ Saved: visuals/07_risk_score_heatmap.png")

# ─── PRINT SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 3 COMPLETE ✓")
print("=" * 70)

print("\n── TOP 5 CRITICAL BOTTLENECK HUBS:")
for i, row in node_features.head(5).iterrows():
    print(f"  #{i+1}  {row['facility']}")
    print(f"       Risk Score: {row['structural_risk_score']:.4f} | "
          f"Betweenness: {row['betweenness']:.4f} | "
          f"Trips: {int(row['total_trips']):,} | "
          f"SLA Breach: {row['avg_sla_breach']*100:.1f}%")

print(f"\n── {len(communities_list)} communities detected across {len(nodes):,} facilities")
print(f"── Cross-community corridors: {corridor_df['cross_community'].sum():,} "
      f"({corridor_df['cross_community'].mean()*100:.1f}% of all corridors)")
