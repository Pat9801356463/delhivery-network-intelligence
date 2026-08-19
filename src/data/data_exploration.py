"""
Phase 1: Data Exploration, Quality Assessment, and Schema Documentation
Delhivery Logistics Network Intelligence Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import json
import os

warnings.filterwarnings('ignore')

DATA_PATH = "/home/claude/project/data/delivery_data.csv"
REPORTS_DIR = "/home/claude/project/reports"
VISUALS_DIR = "/home/claude/project/visuals"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 1: DATA EXPLORATION & QUALITY ASSESSMENT")
print("=" * 70)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─── Data Dictionary ──────────────────────────────────────────────────────────
column_meta = {
    "data":                           ("str",     "Split indicator: 'training' or 'test'"),
    "trip_creation_time":             ("datetime","Timestamp when trip was created"),
    "route_schedule_uuid":            ("str",     "Unique ID for route schedule"),
    "route_type":                     ("str",     "Route type: FTL (Full Truck Load) or Carting"),
    "trip_uuid":                      ("str",     "Unique trip identifier"),
    "source_center":                  ("str",     "Source facility code (node)"),
    "source_name":                    ("str",     "Human-readable source facility name"),
    "destination_center":             ("str",     "Destination facility code (node)"),
    "destination_name":               ("str",     "Human-readable destination facility name"),
    "od_start_time":                  ("datetime","Origin-Destination start time"),
    "od_end_time":                    ("datetime","Origin-Destination end time"),
    "start_scan_to_end_scan":         ("float",   "Time (min) from first to last scan"),
    "is_cutoff":                      ("bool",    "Whether shipment missed cutoff deadline"),
    "cutoff_factor":                  ("int",     "Cutoff time factor in minutes"),
    "cutoff_timestamp":               ("str",     "Actual cutoff timestamp"),
    "actual_distance_to_destination": ("float",   "Actual distance traveled (km)"),
    "actual_time":                    ("float",   "Actual delivery time (minutes)"),
    "osrm_time":                      ("float",   "OSRM-estimated time (minutes)"),
    "osrm_distance":                  ("float",   "OSRM-estimated distance (km)"),
    "factor":                         ("float",   "actual_time / osrm_time (delay ratio for full trip)"),
    "segment_actual_time":            ("float",   "Actual segment time (minutes)"),
    "segment_osrm_time":              ("float",   "OSRM-estimated segment time (minutes)"),
    "segment_osrm_distance":          ("float",   "OSRM-estimated segment distance (km)"),
    "segment_factor":                 ("float",   "segment_actual_time / segment_osrm_time"),
}

print("\n── DATA DICTIONARY ──────────────────────────────────────────────────")
print(f"{'Column':<40} {'Type':<10} {'Description'}")
print("-" * 90)
for col, (dtype, desc) in column_meta.items():
    print(f"  {col:<38} {dtype:<10} {desc}")

# ─── Schema Identification ────────────────────────────────────────────────────
print("\n── SCHEMA IDENTIFICATION ────────────────────────────────────────────")
print(f"  Primary Key (trip):        trip_uuid ({df['trip_uuid'].nunique():,} unique)")
print(f"  Route Identifier:          route_schedule_uuid ({df['route_schedule_uuid'].nunique():,} unique)")
print(f"  Source Node (facility):    source_center ({df['source_center'].nunique():,} unique)")
print(f"  Destination Node:          destination_center ({df['destination_center'].nunique():,} unique)")
print(f"  OSRM Time Column:          osrm_time / segment_osrm_time")
print(f"  Actual Time Column:        actual_time / segment_actual_time")
print(f"  Delay Ratio Column:        factor (actual/osrm), segment_factor")
print(f"  Route Type Column:         route_type {df['route_type'].value_counts().to_dict()}")
print(f"  SLA Proxy:                 is_cutoff (SLA breach flag)")

# ─── Parse Timestamps ─────────────────────────────────────────────────────────
df['trip_creation_time'] = pd.to_datetime(df['trip_creation_time'], errors='coerce')
df['od_start_time'] = pd.to_datetime(df['od_start_time'], errors='coerce')
df['od_end_time'] = pd.to_datetime(df['od_end_time'], errors='coerce')

# ─── Derived columns ──────────────────────────────────────────────────────────
df['delay_ratio'] = df['actual_time'] / df['osrm_time'].replace(0, np.nan)
df['delay_pct'] = (df['delay_ratio'] - 1) * 100
df['is_delayed'] = df['delay_ratio'] > 1.20   # 20%+ over OSRM = delayed
df['is_sla_breach'] = df['is_cutoff'].astype(bool)
df['corridor'] = df['source_center'] + " → " + df['destination_center']
df['hour'] = df['od_start_time'].dt.hour
df['dow'] = df['od_start_time'].dt.dayofweek
df['month'] = df['od_start_time'].dt.month
df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
df['distance_km'] = df['actual_distance_to_destination']

# ─── Missing Value Report ──────────────────────────────────────────────────────
print("\n── MISSING VALUE REPORT ─────────────────────────────────────────────")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
mv_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct.round(3)})
mv_df = mv_df[mv_df['Missing Count'] > 0]
print(mv_df.to_string())
print(f"\n  Total columns with missing data: {len(mv_df)}")
print(f"  Missing source_name: {df['source_name'].isnull().sum()} rows (imputed from source_center)")
print(f"  Missing destination_name: {df['destination_name'].isnull().sum()} rows (imputed)")
# Impute
df['source_name'] = df['source_name'].fillna(df['source_center'])
df['destination_name'] = df['destination_name'].fillna(df['destination_center'])

# ─── Data Quality Report ──────────────────────────────────────────────────────
print("\n── DATA QUALITY REPORT ──────────────────────────────────────────────")
q_issues = []

# Negative times
neg_actual = (df['actual_time'] <= 0).sum()
neg_osrm = (df['osrm_time'] <= 0).sum()
q_issues.append(("Negative/zero actual_time", neg_actual))
q_issues.append(("Negative/zero osrm_time", neg_osrm))

# Extreme delay ratios
extreme_delay = (df['delay_ratio'] > 10).sum()
q_issues.append(("Delay ratio > 10x OSRM", extreme_delay))

# Negative segment_factor
neg_seg_factor = (df['segment_factor'] < 0).sum()
q_issues.append(("Negative segment_factor (data anomaly)", neg_seg_factor))

# Same source/dest
same_od = (df['source_center'] == df['destination_center']).sum()
q_issues.append(("Same source & destination", same_od))

for issue, count in q_issues:
    flag = "⚠️ " if count > 0 else "✓ "
    print(f"  {flag} {issue}: {count:,} rows")

print(f"\n  Overall data quality: {100*(1 - neg_actual/len(df)):.1f}% valid actual_time records")
print(f"  SLA breach rate: {df['is_sla_breach'].mean()*100:.1f}%")
print(f"  Chronic delay rate (>20% over OSRM): {df['is_delayed'].mean()*100:.1f}%")

# ─── Outlier Report ───────────────────────────────────────────────────────────
print("\n── OUTLIER REPORT ───────────────────────────────────────────────────")
numeric_cols = ['actual_time', 'osrm_time', 'delay_ratio', 'distance_km', 'segment_actual_time']
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    out_low = (df[col] < q1 - 3*iqr).sum()
    out_high = (df[col] > q3 + 3*iqr).sum()
    print(f"  {col:<40} Low outliers: {out_low:>5,}  High outliers: {out_high:>5,}  (3×IQR rule)")

# ─── Leakage Assessment ───────────────────────────────────────────────────────
print("\n── LEAKAGE ASSESSMENT ───────────────────────────────────────────────")
leakage_risk = [
    ("factor", "HIGH", "= actual_time/osrm_time — leaks actual_time directly. EXCLUDE from features."),
    ("segment_factor", "HIGH", "= segment_actual_time/osrm_time — leaks target. EXCLUDE."),
    ("start_scan_to_end_scan", "MEDIUM", "Partially observed in real-time; use with caution."),
    ("od_end_time", "HIGH", "Reveals actual delivery end; only available post-delivery. EXCLUDE."),
    ("cutoff_timestamp", "MEDIUM", "Derived from cutoff event; exclude from ETA prediction."),
    ("actual_distance_to_destination", "LOW", "Physical distance — safe to use."),
    ("osrm_time", "LOW", "OSRM estimate — key predictor, not a leak."),
]
for col, risk, note in leakage_risk:
    print(f"  [{risk}] {col:<40} {note}")

# ─── Summary Stats ────────────────────────────────────────────────────────────
print("\n── SUMMARY STATISTICS ───────────────────────────────────────────────")
print(df[['actual_time','osrm_time','delay_ratio','distance_km']].describe().round(2).to_string())

# ─── Save cleaned data ────────────────────────────────────────────────────────
train_df = df[df['data'] == 'training'].copy()
test_df  = df[df['data'] == 'test'].copy()

train_df.to_csv(f"{REPORTS_DIR}/train_clean.csv", index=False)
test_df.to_csv(f"{REPORTS_DIR}/test_clean.csv", index=False)
df.to_csv(f"{REPORTS_DIR}/full_clean.csv", index=False)
print(f"\n✓ Saved: train ({len(train_df):,}), test ({len(test_df):,}), full ({len(df):,}) to reports/")

# ─── Visualizations ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor='#0a0e1a')
fig.suptitle('Delhivery Delivery Network — Data Quality Dashboard', 
             fontsize=18, fontweight='bold', color='white', y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

c1, c2, c3, c4 = '#00d4ff', '#ff6b35', '#00ff88', '#ffd700'
bg = '#0a0e1a'
panel_bg = '#0f1626'

ax1 = fig.add_subplot(gs[0, 0:2])
ax1.set_facecolor(panel_bg)
vals = df['route_type'].value_counts()
colors_pie = [c1, c2]
wedges, texts, autotexts = ax1.pie(vals.values, labels=vals.index, autopct='%1.1f%%',
    colors=colors_pie, startangle=90, textprops={'color':'white','fontsize':11})
for at in autotexts: at.set_fontsize(12); at.set_fontweight('bold')
ax1.set_title('Route Type Distribution', color='white', fontsize=13, fontweight='bold', pad=8)

ax2 = fig.add_subplot(gs[0, 2:4])
ax2.set_facecolor(panel_bg)
sla = df.groupby('route_type')['is_sla_breach'].mean() * 100
bars = ax2.bar(sla.index, sla.values, color=[c1, c2], edgecolor='none', width=0.5)
for bar, v in zip(bars, sla.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}%',
             ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
ax2.set_facecolor(panel_bg)
ax2.set_title('SLA Breach Rate by Route Type', color='white', fontsize=13, fontweight='bold')
ax2.set_ylabel('Breach Rate (%)', color='#aaa')
ax2.tick_params(colors='white')
for spine in ax2.spines.values(): spine.set_color('#222')
ax2.set_ylim(0, sla.max()*1.25)
ax2.yaxis.label.set_color('#888')
ax2.tick_params(axis='y', colors='#888')
ax2.tick_params(axis='x', colors='white')

ax3 = fig.add_subplot(gs[1, 0:2])
ax3.set_facecolor(panel_bg)
delay_clip = df['delay_ratio'].clip(0, 8)
ax3.hist(delay_clip, bins=80, color=c1, alpha=0.85, edgecolor='none')
ax3.axvline(1.0, color='white', linestyle='--', linewidth=1.5, label='OSRM baseline (1.0×)')
ax3.axvline(df['delay_ratio'].median(), color=c3, linestyle='--', linewidth=1.5,
            label=f"Median: {df['delay_ratio'].median():.2f}×")
ax3.set_title('Delay Ratio Distribution (actual / OSRM)', color='white', fontsize=13, fontweight='bold')
ax3.set_xlabel('Delay Ratio', color='#888')
ax3.set_ylabel('Count', color='#888')
ax3.legend(fontsize=9, facecolor='#1a2040', labelcolor='white', framealpha=0.7)
ax3.tick_params(colors='#888')
for spine in ax3.spines.values(): spine.set_color('#222')

ax4 = fig.add_subplot(gs[1, 2:4])
ax4.set_facecolor(panel_bg)
hourly_delay = df.groupby('hour')['delay_ratio'].mean()
ax4.fill_between(hourly_delay.index, hourly_delay.values, alpha=0.35, color=c2)
ax4.plot(hourly_delay.index, hourly_delay.values, color=c2, linewidth=2.5, marker='o', markersize=5)
ax4.axhline(1.0, color='white', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_title('Average Delay Ratio by Hour of Day', color='white', fontsize=13, fontweight='bold')
ax4.set_xlabel('Hour of Day', color='#888')
ax4.set_ylabel('Delay Ratio', color='#888')
ax4.tick_params(colors='#888')
for spine in ax4.spines.values(): spine.set_color('#222')
ax4.set_xticks(range(0, 24, 3))

ax5 = fig.add_subplot(gs[2, 0:2])
ax5.set_facecolor(panel_bg)
monthly = df.groupby('month')['delay_ratio'].mean()
bars5 = ax5.bar(monthly.index, monthly.values, color=c4, edgecolor='none', alpha=0.85)
ax5.axhline(monthly.mean(), color=c3, linestyle='--', linewidth=1.5, label=f"Mean: {monthly.mean():.2f}×")
ax5.set_title('Average Delay Ratio by Month', color='white', fontsize=13, fontweight='bold')
ax5.set_xlabel('Month', color='#888')
ax5.set_ylabel('Delay Ratio', color='#888')
ax5.tick_params(colors='#888')
for spine in ax5.spines.values(): spine.set_color('#222')
ax5.legend(fontsize=9, facecolor='#1a2040', labelcolor='white')
ax5.set_xticks(monthly.index)
mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax5.set_xticklabels([mnames[m-1] for m in monthly.index], color='white', fontsize=9)

ax6 = fig.add_subplot(gs[2, 2:4])
ax6.set_facecolor(panel_bg)
ax6.scatter(df['osrm_time'].clip(0,1000), df['actual_time'].clip(0,1000),
            alpha=0.03, s=2, c=c1)
max_val = min(1000, max(df['osrm_time'].max(), df['actual_time'].max()))
ax6.plot([0, max_val], [0, max_val], 'w--', linewidth=1.5, label='Perfect prediction')
ax6.plot([0, max_val], [0, max_val*1.2], color=c2, linewidth=1, linestyle=':', label='+20% (SLA risk)')
ax6.set_title('Actual Time vs OSRM Estimate', color='white', fontsize=13, fontweight='bold')
ax6.set_xlabel('OSRM Time (min)', color='#888')
ax6.set_ylabel('Actual Time (min)', color='#888')
ax6.tick_params(colors='#888')
for spine in ax6.spines.values(): spine.set_color('#222')
ax6.legend(fontsize=8, facecolor='#1a2040', labelcolor='white')

plt.savefig(f"{VISUALS_DIR}/01_data_quality_dashboard.png", dpi=150, bbox_inches='tight',
            facecolor=bg, edgecolor='none')
plt.close()
print(f"✓ Saved: visuals/01_data_quality_dashboard.png")

# ─── Save JSON summary for downstream use ─────────────────────────────────────
summary = {
    "total_rows": len(df),
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "unique_sources": int(df['source_center'].nunique()),
    "unique_destinations": int(df['destination_center'].nunique()),
    "unique_trips": int(df['trip_uuid'].nunique()),
    "unique_corridors": int(df['corridor'].nunique()),
    "route_type_dist": df['route_type'].value_counts().to_dict(),
    "sla_breach_rate": float(df['is_sla_breach'].mean()),
    "chronic_delay_rate": float(df['is_delayed'].mean()),
    "median_delay_ratio": float(df['delay_ratio'].median()),
    "mean_delay_ratio": float(df['delay_ratio'].mean()),
    "columns": list(df.columns),
}
with open(f"{REPORTS_DIR}/data_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 1 COMPLETE ✓")
print("=" * 70)
print(f"\nKey Insights:")
print(f"  • {len(df):,} total shipment records across {df['trip_uuid'].nunique():,} trips")
print(f"  • {df['source_center'].nunique():,} source + {df['destination_center'].nunique():,} destination facilities")
print(f"  • {df['corridor'].nunique():,} unique corridors in the network")
print(f"  • FTL: {df['route_type'].value_counts()['FTL']:,} | Carting: {df['route_type'].value_counts()['Carting']:,}")
print(f"  • Median delay ratio: {df['delay_ratio'].median():.2f}× OSRM (OSRM consistently underestimates)")
print(f"  • SLA breach rate: {df['is_sla_breach'].mean()*100:.1f}%")
print(f"  • Chronic delay rate (>20%): {df['is_delayed'].mean()*100:.1f}%")
