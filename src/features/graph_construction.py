"""
Phase 2: Graph Construction Pipeline
Builds directed weighted logistics graphs for network analysis
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
import pickle
import json
import os

warnings.filterwarnings('ignore')

REPORTS_DIR = "/home/claude/project/reports"
VISUALS_DIR = "/home/claude/project/visuals"
MODELS_DIR  = "/home/claude/project/models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 2: GRAPH CONSTRUCTION")
print("=" * 70)

# ─── Load Cleaned Data ────────────────────────────────────────────────────────
df = pd.read_csv(f"{REPORTS_DIR}/full_clean.csv")
df['od_start_time'] = pd.to_datetime(df['od_start_time'], errors='coerce')
df['hour'] = df['od_start_time'].dt.hour
df['month'] = df['od_start_time'].dt.month
df['delay_ratio'] = df['actual_time'] / df['osrm_time'].replace(0, np.nan)
df['is_delayed'] = df['delay_ratio'] > 1.20
df['is_sla_breach'] = df['is_cutoff'].astype(bool)
df['corridor'] = df['source_center'] + " → " + df['destination_center']

# ─── CORRIDOR-LEVEL AGGREGATION ───────────────────────────────────────────────
print("\n── Computing corridor-level aggregations...")

def entropy(series):
    """Shannon entropy of categorical distribution"""
    counts = series.value_counts(normalize=True)
    return -(counts * np.log2(counts + 1e-10)).sum()

corridor_stats = df.groupby(['source_center','destination_center']).agg(
    trip_count         = ('trip_uuid', 'count'),
    median_actual_time = ('actual_time', 'median'),
    mean_actual_time   = ('actual_time', 'mean'),
    median_osrm_time   = ('osrm_time', 'median'),
    median_delay_ratio = ('delay_ratio', 'median'),
    mean_delay_ratio   = ('delay_ratio', 'mean'),
    std_delay_ratio    = ('delay_ratio', 'std'),
    delay_pct          = ('delay_ratio', lambda x: ((x - 1)*100).median()),
    sla_breach_rate    = ('is_sla_breach', 'mean'),
    chronic_delay_rate = ('is_delayed', 'mean'),
    median_distance    = ('actual_distance_to_destination', 'median'),
    ftl_share          = ('route_type', lambda x: (x=='FTL').mean()),
    route_type_entropy = ('route_type', entropy),
).reset_index()

corridor_stats['is_bottleneck_corridor'] = corridor_stats['median_delay_ratio'] > 1.5
print(f"  ✓ {len(corridor_stats):,} unique corridors aggregated")
print(f"  ✓ Bottleneck corridors (delay > 1.5×): {corridor_stats['is_bottleneck_corridor'].sum():,}")

# ─── GRAPH 1: Base Graph ──────────────────────────────────────────────────────
print("\n── Building G_base (raw corridor graph)...")
G_base = nx.DiGraph()

for _, row in corridor_stats.iterrows():
    src = row['source_center']
    dst = row['destination_center']
    G_base.add_edge(src, dst,
        trip_count=row['trip_count'],
        median_actual_time=row['median_actual_time'],
        median_osrm_time=row['median_osrm_time'],
        weight=row['trip_count'],   # edge weight = frequency
        delay_ratio=row['median_delay_ratio'],
        sla_breach_rate=row['sla_breach_rate'],
        distance=row['median_distance'],
    )

print(f"  ✓ G_base: {G_base.number_of_nodes():,} nodes | {G_base.number_of_edges():,} edges")

# ─── GRAPH 2: Delay-Weighted Graph ───────────────────────────────────────────
print("\n── Building G_delay (delay-ratio weighted)...")
G_delay = nx.DiGraph()
for _, row in corridor_stats.iterrows():
    G_delay.add_edge(row['source_center'], row['destination_center'],
        weight=row['median_delay_ratio'],
        delay_ratio=row['median_delay_ratio'],
        sla_breach_rate=row['sla_breach_rate'],
        trip_count=row['trip_count'],
        chronic_delay_rate=row['chronic_delay_rate'],
    )
print(f"  ✓ G_delay: {G_delay.number_of_nodes():,} nodes | {G_delay.number_of_edges():,} edges")

# ─── GRAPH 3: Route-Type Graph ────────────────────────────────────────────────
print("\n── Building G_route_type (separate FTL & Carting subgraphs)...")

def build_route_graph(df_sub, route_type_label):
    G = nx.DiGraph()
    sub_stats = df_sub.groupby(['source_center','destination_center']).agg(
        trip_count=('trip_uuid','count'),
        median_delay=('delay_ratio','median'),
        sla_rate=('is_sla_breach','mean'),
    ).reset_index()
    for _, row in sub_stats.iterrows():
        G.add_edge(row['source_center'], row['destination_center'],
            weight=row['median_delay'],
            trip_count=row['trip_count'],
            sla_rate=row['sla_rate'],
            route_type=route_type_label,
        )
    return G

G_ftl     = build_route_graph(df[df['route_type']=='FTL'], 'FTL')
G_carting = build_route_graph(df[df['route_type']=='Carting'], 'Carting')
print(f"  ✓ G_ftl: {G_ftl.number_of_nodes():,} nodes | {G_ftl.number_of_edges():,} edges")
print(f"  ✓ G_carting: {G_carting.number_of_nodes():,} nodes | {G_carting.number_of_edges():,} edges")

# ─── GRAPH 4: Temporal Graphs ─────────────────────────────────────────────────
print("\n── Building G_temporal (time-of-day slices)...")
G_temporal = {}
time_slots = {
    'night_00_06': (0, 6),
    'morning_06_12': (6, 12),
    'afternoon_12_18': (12, 18),
    'evening_18_24': (18, 24),
}
for slot_name, (h_start, h_end) in time_slots.items():
    mask = (df['hour'] >= h_start) & (df['hour'] < h_end)
    df_slot = df[mask]
    G_slot = nx.DiGraph()
    if len(df_slot) > 0:
        slot_stats = df_slot.groupby(['source_center','destination_center']).agg(
            trip_count=('trip_uuid','count'),
            median_delay=('delay_ratio','median'),
        ).reset_index()
        for _, row in slot_stats.iterrows():
            G_slot.add_edge(row['source_center'], row['destination_center'],
                weight=row['median_delay'],
                trip_count=row['trip_count'],
            )
    G_temporal[slot_name] = G_slot
    print(f"  ✓ G_temporal[{slot_name}]: {G_slot.number_of_nodes()} nodes | {G_slot.number_of_edges()} edges")

# ─── NODE-LEVEL FACILITY STATISTICS ──────────────────────────────────────────
print("\n── Computing node-level facility statistics...")

# As source
src_stats = df.groupby('source_center').agg(
    outbound_trips   = ('trip_uuid','count'),
    out_delay_ratio  = ('delay_ratio','median'),
    out_sla_breach   = ('is_sla_breach','mean'),
).reset_index().rename(columns={'source_center':'facility'})

# As destination
dst_stats = df.groupby('destination_center').agg(
    inbound_trips    = ('trip_uuid','count'),
    in_delay_ratio   = ('delay_ratio','median'),
    in_sla_breach    = ('is_sla_breach','mean'),
).reset_index().rename(columns={'destination_center':'facility'})

# All facilities
all_facilities = pd.DataFrame({'facility': list(set(df['source_center'].unique()) |
                                                  set(df['destination_center'].unique()))})
node_stats = all_facilities.merge(src_stats, on='facility', how='left')
node_stats = node_stats.merge(dst_stats, on='facility', how='left')
node_stats = node_stats.fillna(0)
node_stats['total_trips'] = node_stats['outbound_trips'] + node_stats['inbound_trips']
node_stats['avg_delay_ratio'] = (node_stats['out_delay_ratio'] + node_stats['in_delay_ratio']) / 2
node_stats['avg_sla_breach']  = (node_stats['out_sla_breach'] + node_stats['in_sla_breach']) / 2

# ─── SAVE GRAPHS & STATS ──────────────────────────────────────────────────────
graphs = {
    'G_base': G_base,
    'G_delay': G_delay,
    'G_ftl': G_ftl,
    'G_carting': G_carting,
    'G_temporal': G_temporal,
}
with open(f"{MODELS_DIR}/graphs.pkl", 'wb') as f:
    pickle.dump(graphs, f)

corridor_stats.to_csv(f"{REPORTS_DIR}/corridor_stats.csv", index=False)
node_stats.to_csv(f"{REPORTS_DIR}/node_stats_raw.csv", index=False)
print(f"  ✓ Saved graphs to models/graphs.pkl")
print(f"  ✓ Saved corridor_stats.csv ({len(corridor_stats):,} corridors)")
print(f"  ✓ Saved node_stats_raw.csv ({len(node_stats):,} facilities)")

# ─── VISUALIZATION: Network Overview ─────────────────────────────────────────
print("\n── Generating network visualizations...")

# Use a high-trip-count subgraph for visualization clarity
top_corridors = corridor_stats.nlargest(300, 'trip_count')
G_vis = nx.DiGraph()
for _, row in top_corridors.iterrows():
    G_vis.add_edge(row['source_center'], row['destination_center'],
        weight=row['trip_count'],
        delay=row['median_delay_ratio'])

fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor='#0a0e1a')
fig.suptitle('Delhivery Logistics Network Graph\n(Top 300 Corridors by Volume)',
             fontsize=16, fontweight='bold', color='white', y=1.01)

for ax_idx, ax in enumerate(axes):
    ax.set_facecolor('#0a0e1a')
    
    # Layout
    if ax_idx == 0:
        pos = nx.spring_layout(G_vis, k=0.8, seed=42)
        title = 'Network Topology (Spring Layout)'
        edge_color_attr = 'weight'
    else:
        pos = nx.kamada_kawai_layout(G_vis)
        title = 'Network Topology (Kamada-Kawai Layout)'
        edge_color_attr = 'delay'

    # Node sizes by degree
    degrees = dict(G_vis.degree())
    node_sizes = [max(30, degrees.get(n, 1) * 15) for n in G_vis.nodes()]
    
    # Edge colors by delay
    edge_delays = [G_vis[u][v].get('delay', 1.0) for u, v in G_vis.edges()]
    e_min, e_max = min(edge_delays), max(edge_delays)
    
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=e_min, vmax=min(e_max, 4.0))
    edge_colors = [cmap(norm(d)) for d in edge_delays]
    
    nx.draw_networkx_edges(G_vis, pos, ax=ax,
        edge_color=edge_colors, alpha=0.4, width=0.6,
        arrows=True, arrowsize=6, arrowstyle='-|>')
    
    nx.draw_networkx_nodes(G_vis, pos, ax=ax,
        node_size=node_sizes,
        node_color='#00d4ff', alpha=0.85)
    
    # Label top nodes only
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:15]
    labels = {n: n[-7:] for n in top_nodes}  # Truncate long IDs
    nx.draw_networkx_labels(G_vis, pos, labels, ax=ax,
        font_size=6, font_color='white', font_weight='bold')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Delay Ratio', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/02_network_graph.png", dpi=150, bbox_inches='tight',
            facecolor='#0a0e1a')
plt.close()
print("  ✓ Saved: visuals/02_network_graph.png")

# ─── Delay distribution by corridor ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='#0a0e1a')
fig.suptitle('Corridor Delay Analysis', fontsize=15, color='white', fontweight='bold')

c1, c2, c3 = '#00d4ff', '#ff6b35', '#ffd700'
panel_bg = '#0f1626'

# Plot 1: Delay ratio distribution by route type
ax = axes[0]
ax.set_facecolor(panel_bg)
for rt, color in [('FTL', c1), ('Carting', c2)]:
    sub = df[df['route_type']==rt]['delay_ratio'].clip(0,8)
    ax.hist(sub, bins=60, alpha=0.6, color=color, label=rt, density=True)
ax.set_title('Delay Ratio by Route Type', color='white', fontweight='bold')
ax.set_xlabel('Delay Ratio', color='#888'); ax.set_ylabel('Density', color='#888')
ax.legend(facecolor='#1a2040', labelcolor='white')
ax.tick_params(colors='#888')
for s in ax.spines.values(): s.set_color('#222')

# Plot 2: Corridor frequency histogram
ax = axes[1]
ax.set_facecolor(panel_bg)
ax.hist(corridor_stats['trip_count'].clip(0,500), bins=60, color=c3, alpha=0.85)
ax.set_title('Corridor Trip Frequency', color='white', fontweight='bold')
ax.set_xlabel('Trips per Corridor', color='#888'); ax.set_ylabel('Corridors', color='#888')
ax.tick_params(colors='#888')
for s in ax.spines.values(): s.set_color('#222')

# Plot 3: SLA breach rate distribution
ax = axes[2]
ax.set_facecolor(panel_bg)
ax.hist(corridor_stats['sla_breach_rate'] * 100, bins=50, color=c1, alpha=0.85)
ax.axvline(50, color=c2, linestyle='--', linewidth=2, label='50% breach threshold')
ax.set_title('SLA Breach Rate per Corridor (%)', color='white', fontweight='bold')
ax.set_xlabel('SLA Breach Rate (%)', color='#888'); ax.set_ylabel('Corridors', color='#888')
ax.legend(facecolor='#1a2040', labelcolor='white')
ax.tick_params(colors='#888')
for s in ax.spines.values(): s.set_color('#222')

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/03_corridor_delay_analysis.png", dpi=150, bbox_inches='tight',
            facecolor='#0a0e1a')
plt.close()
print("  ✓ Saved: visuals/03_corridor_delay_analysis.png")

print("\n" + "=" * 70)
print("PHASE 2 COMPLETE ✓")
print("=" * 70)
print(f"\nGraph Summary:")
print(f"  G_base:    {G_base.number_of_nodes():,} nodes | {G_base.number_of_edges():,} edges")
print(f"  G_delay:   {G_delay.number_of_nodes():,} nodes | {G_delay.number_of_edges():,} edges")
print(f"  G_ftl:     {G_ftl.number_of_nodes():,} nodes | {G_ftl.number_of_edges():,} edges")
print(f"  G_carting: {G_carting.number_of_nodes():,} nodes | {G_carting.number_of_edges():,} edges")
print(f"  G_temporal: {len(G_temporal)} time slots")
