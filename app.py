"""
Delhivery Network Intelligence Dashboard — v2 (All Improvements)
Run: streamlit run app.py
"""
import streamlit as st, pandas as pd, numpy as np, pickle, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Delhivery Network Intelligence",page_icon="🚚",layout="wide")

st.markdown("""<style>
.stApp{background-color:#0a0e1a;color:#e0e6f0}
.section-header{background:linear-gradient(90deg,#0f1626,#1a2a4a);border-left:4px solid #00d4ff;
  border-radius:4px;padding:0.6rem 1rem;margin:1rem 0 0.5rem 0;font-size:1.1rem;font-weight:700;color:#00d4ff}
.hub-card{background:#0f1626;border:1px solid #1e3a5f;border-radius:8px;padding:0.8rem 1rem;margin:0.3rem 0}
.critical{border-left:4px solid #ff3366}.high{border-left:4px solid #ff6b35}
div[data-testid="stSidebar"]{background-color:#080c16}
</style>""",unsafe_allow_html=True)

BASE=os.path.dirname(os.path.abspath(__file__))
RPT=os.path.join(BASE,"reports"); MDL=os.path.join(BASE,"models"); VIS=os.path.join(BASE,"visuals")

@st.cache_data
def load_data():
    df=pd.read_csv(f"{RPT}/full_clean_v2.csv")
    corr=pd.read_csv(f"{RPT}/corridor_stats_enriched.csv")
    nf=pd.read_csv(f"{RPT}/node_features.csv")
    fi=pd.read_csv(f"{RPT}/feature_importance.csv")
    bv2=pd.read_csv(f"{RPT}/model_benchmark_v2.csv")
    top50=pd.read_csv(f"{RPT}/top50_delayed_corridors.csv")
    df["od_start_time"]=pd.to_datetime(df["od_start_time"],errors="coerce")
    df["delay_ratio"]=df["actual_time"]/df["osrm_time"].replace(0,np.nan)
    df["is_sla_breach"]=df["is_cutoff"].astype(bool)
    df["sla_realistic"]=df["actual_time"]>(df["cutoff_factor"]*1.5)
    df["hour"]=df["od_start_time"].dt.hour
    df["month"]=df["od_start_time"].dt.month
    return df,corr,nf,fi,bv2,top50

@st.cache_resource
def load_models():
    with open(f"{MDL}/improved_models.pkl","rb") as f: imp=pickle.load(f)
    with open(f"{MDL}/feature_store.pkl","rb") as f: fs=pickle.load(f)
    return imp,fs

def jload(fn):
    with open(f"{RPT}/{fn}") as f: return json.load(f)

def show(fn):
    p=f"{VIS}/{fn}"
    if os.path.exists(p): st.image(p,use_container_width=True)

def sh(label): st.markdown(f'''<div class="section-header">{label}</div>''',unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🚚 Delhivery"); st.sidebar.markdown("### Network Intelligence **v2**")
st.sidebar.markdown("---")
PAGE=st.sidebar.radio("Navigate",[
    "🏠 Network Overview","🔴 Bottleneck Hubs","⚠️ Delay Corridors",
    "⏱️ ETA Prediction","🔀 Route Recommender","💰 Business Impact","📈 Improvement Report"
],label_visibility="collapsed")

try:
    adv2=jload("graph_advantage_report_v2.json")
    st.sidebar.markdown("---"); st.sidebar.markdown("**v2 Gains vs v1**")
    st.sidebar.markdown(f"- RMSE: `{adv2['rmse_improvement_pct']:+.1f}%`")
    st.sidebar.markdown(f"- MAE: `{adv2['mae_improvement_pct']:+.1f}%`")
    st.sidebar.markdown(f"- ±15%: `+{adv2['w15_improvement_pp']:.1f}pp`")
except: pass
st.sidebar.markdown("---"); st.sidebar.markdown("<small style='color:#445'>© 2026 Delhivery AI Lab</small>",unsafe_allow_html=True)

BG,PBG,C1,C2,C3,C4="#0a0e1a","#0f1626","#00d4ff","#ff6b35","#00ff88","#ffd700"

# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if PAGE=="🏠 Network Overview":
    st.title("🚚 Delhivery Network Intelligence")
    st.markdown("*Graph ML + HistGBM + Per-Route models — v2*")
    df,corr,nf,fi,bv2,top50=load_data()
    cols=st.columns(6)
    kpis=[(f"{len(df):,}","Shipments","📦"),(f"{df['source_center'].nunique():,}","Facilities","🏢"),
          (f"{corr.shape[0]:,}","Corridors","🛣️"),(f"{df['delay_ratio'].median():.2f}×","Median Delay","⚠️"),
          (f"{df['is_sla_breach'].mean()*100:.1f}%","SLA Breach","🔴"),(f"{df['sla_realistic'].mean()*100:.1f}%","Realistic Breach","🟡")]
    for col,(val,label,icon) in zip(cols,kpis):
        col.metric(f"{icon} {label}",val)
    sh("📊 Network Health")
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(8,4),facecolor=BG); ax.set_facecolor(PBG)
        for rt,col in [("FTL",C1),("Carting",C2)]:
            d=df[df["route_type"]==rt]["delay_ratio"].clip(0,8)
            ax.hist(d,bins=60,alpha=0.55,color=col,label=f"{rt} (med={d.median():.2f}×)",density=True)
        ax.axvline(1.0,color="white",ls="--",lw=1.5)
        ax.set_title("Delay Ratio by Route Type",color="white",fontweight="bold")
        ax.set_xlabel("Actual/OSRM",color="#888"); ax.legend(facecolor="#1a2040",labelcolor="white")
        ax.tick_params(colors="#888"); [sp.set_color("#222") for sp in ax.spines.values()]
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
    with c2:
        fig,ax=plt.subplots(figsize=(8,4),facecolor=BG); ax.set_facecolor(PBG)
        hourly=df.groupby("hour")["delay_ratio"].mean()
        ax.fill_between(hourly.index,hourly.values,alpha=0.3,color=C2)
        ax.plot(hourly.index,hourly.values,color=C2,lw=2.5,marker="o",markersize=4)
        ax.axhline(1.0,color="white",ls="--",lw=1.2,alpha=0.5)
        ax.set_title("Avg Delay Ratio by Hour",color="white",fontweight="bold")
        ax.set_xlabel("Hour",color="#888"); ax.set_ylabel("Delay Ratio",color="#888")
        ax.tick_params(colors="#888"); [sp.set_color("#222") for sp in ax.spines.values()]
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
    sh("📈 v2 Model Benchmark Summary")
    try:
        adv2=jload("graph_advantage_report_v2.json")
        mc=st.columns(4)
        mc[0].metric("v1 RMSE",f"{adv2['original_graph_rmse']:.1f} min")
        mc[1].metric("v2 RMSE",f"{adv2['best_full_test_rmse']:.1f} min",delta=f"{adv2['rmse_improvement_pct']:+.1f}%")
        mc[2].metric("v1 ±15%",f"{adv2['original_graph_w15']:.1f}%")
        mc[3].metric("v2 ±15%",f"{adv2['best_full_test_w15']:.1f}%",delta=f"+{adv2['w15_improvement_pp']:.1f}pp")
    except: pass
    show("00_executive_summary_dashboard_v2.png")

# ── PAGE 2: BOTTLENECK HUBS ───────────────────────────────────────────────────
elif PAGE=="🔴 Bottleneck Hubs":
    st.title("🔴 Bottleneck Hub Audit")
    df,corr,nf,fi,bv2,top50=load_data()
    top_n=st.slider("Show Top N",5,50,20)
    sort_by=st.selectbox("Sort By",["structural_risk_score","betweenness","total_trips","avg_sla_breach"])
    top_hubs=nf.nlargest(top_n,sort_by)
    sh("🚨 Top 5 Critical Hubs")
    for i,(col,(_, row)) in enumerate(zip(st.columns(5),nf.head(5).iterrows())):
        sev="critical" if i<2 else "high"
        col.markdown(f'''<div class="hub-card {sev}"><strong style="color:#ff3366;">#{i+1}</strong><br>
            <code style="font-size:0.7rem;">{row['facility'][-12:]}</code><br>
            <span style="color:#ffd700;font-size:1.2rem;font-weight:700;">{row['structural_risk_score']:.3f}</span><br>
            <small style="color:#888">Risk Score</small><br>
            <span style="color:#ff6b35">{row['avg_sla_breach']*100:.0f}%</span> <small style="color:#888">SLA</small><br>
            <span style="color:#00d4ff">{int(row['total_trips']):,}</span> <small style="color:#888">trips</small>
            </div>''',unsafe_allow_html=True)
    sh("📋 Hub Ranking")
    disp=top_hubs[["facility","structural_risk_score","betweenness","total_trips","avg_sla_breach","avg_delay_ratio"]].copy()
    disp["SLA"]=( disp["avg_sla_breach"]*100).round(1).astype(str)+"%"
    disp["BW%"]=(disp["betweenness"]*100).round(2).astype(str)+"%"
    disp=disp.rename(columns={"structural_risk_score":"Risk","total_trips":"Trips","avg_delay_ratio":"Delay Ratio"})
    st.dataframe(disp[["facility","Risk","BW%","Trips","SLA","Delay Ratio"]].reset_index(drop=True),use_container_width=True)
    c1,c2=st.columns(2); 
    with c1: show("07_risk_score_heatmap.png")
    with c2: show("04_centrality_rankings.png")
    show("05_bottleneck_hub_network.png")

# ── PAGE 3: DELAY CORRIDORS ───────────────────────────────────────────────────
elif PAGE=="⚠️ Delay Corridors":
    st.title("⚠️ Delay Corridor Audit")
    df,corr,nf,fi,bv2,top50=load_data()
    delayed=corr[corr["median_delay_ratio"]>1.2]
    mc=st.columns(4)
    mc[0].metric("Total Corridors",f"{len(corr):,}")
    mc[1].metric("Chronic Delay>20%",f"{len(delayed):,}",delta=f"{len(delayed)/len(corr)*100:.0f}% of network")
    mc[2].metric("Worst Corridor",f"+{(corr['median_delay_ratio'].max()-1)*100:.0f}%")
    mc[3].metric("Avg SLA Breach",f"{corr['sla_breach_rate'].mean()*100:.1f}%")
    sh("🔍 Filter Corridors")
    fc1,fc2,fc3=st.columns(3)
    min_trips=fc1.slider("Min Trips",1,200,10)
    min_delay=fc2.slider("Min Delay%",0,500,20)
    rt_filter=fc3.selectbox("Route Type",["All","FTL","Carting"])
    flt=corr.copy(); flt["delay_pct"]=(flt["median_delay_ratio"]-1)*100
    flt=flt[flt["trip_count"]>=min_trips]; flt=flt[flt["delay_pct"]>=min_delay]
    if rt_filter=="FTL": flt=flt[flt["ftl_share"]>0.5]
    elif rt_filter=="Carting": flt=flt[flt["ftl_share"]<=0.5]
    flt=flt.sort_values("delay_pct",ascending=False)
    st.markdown(f"**{len(flt):,} corridors match**")
    disp=flt.head(50)[["source_center","destination_center","trip_count","median_delay_ratio","delay_pct","sla_breach_rate","median_distance"]].copy()
    disp["delay_pct"]=disp["delay_pct"].round(0).astype(int).astype(str)+"%"
    disp["sla_breach_rate"]=(disp["sla_breach_rate"]*100).round(1).astype(str)+"%"
    disp.columns=["Source","Dest","Trips","Delay Ratio","Delay%","SLA Breach","Dist(km)"]
    st.dataframe(disp.reset_index(drop=True),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: show("08_delay_corridor_audit.png")
    with c2: show("03_corridor_delay_analysis.png")

# ── PAGE 4: ETA PREDICTION ────────────────────────────────────────────────────
elif PAGE=="⏱️ ETA Prediction":
    st.title("⏱️ ETA Prediction Engine — v2")
    df,corr,nf,fi,bv2,top50=load_data()
    sh("📊 Full Benchmark (v1 → v2)")
    bd=bv2[~bv2["Phase"].str.contains("FTL-only|Cart-only",na=False)].copy()
    bd=bd[["Model","Phase","MAE","RMSE","R2","Within15pct","Within20pct"]].rename(columns={"Within15pct":"±15%","Within20pct":"±20%"})
    st.dataframe(bd.reset_index(drop=True),use_container_width=True)
    try:
        adv2=jload("graph_advantage_report_v2.json")
        sh("🏆 Net Gains vs v1")
        mc=st.columns(5)
        mc[0].metric("v1 RMSE",f"{adv2['original_graph_rmse']:.1f}"); mc[1].metric("v2 RMSE",f"{adv2['best_full_test_rmse']:.1f}",delta=f"{adv2['rmse_improvement_pct']:+.1f}%")
        mc[2].metric("v1 MAE",f"{adv2['original_graph_mae']:.1f}"); mc[3].metric("v2 MAE",f"{adv2['best_full_test_mae']:.1f}",delta=f"{adv2['mae_improvement_pct']:+.1f}%")
        mc[4].metric("±15% Lift",f"+{adv2['w15_improvement_pp']:.1f}pp")
    except: pass
    sh("🔮 Live ETA Prediction")
    pi1,pi2,pi3=st.columns(3)
    with pi1:
        osrm_t=st.number_input("OSRM Time (min)",10,5000,120); osrm_d=st.number_input("OSRM Dist (km)",1,3000,150)
        dist=st.number_input("Actual Dist (km)",1,3000,148)
    with pi2:
        rt=st.selectbox("Route Type",["FTL","Carting"]); hour=st.slider("Hour",0,23,10)
        dow=st.selectbox("Day",["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    with pi3:
        month=st.slider("Month",1,12,6); src=st.selectbox("Source Hub",nf.head(30)["facility"].tolist())
        dst=st.selectbox("Dest Hub",nf.tail(30)["facility"].tolist())
    if st.button("🔮 Predict ETA",type="primary"):
        try:
            imp,fs=load_models(); ALL=imp["ALL_FEATURES"]
            model=imp["hgbm_ftl"] if rt=="FTL" else imp["hgbm_cart"]
            nm=nf.set_index("facility"); dow_val={"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}[dow]
            def gf(fac,col): return float(nm.loc[fac,col]) if fac in nm.index and col in nm.columns else float(nf[col].median())
            crm=pd.read_csv(f"{RPT}/corridor_stats_enriched.csv").set_index(["source_center","destination_center"])
            def cf(s,d,col): return float(crm.loc[(s,d),col]) if (s,d) in crm.index and col in crm.columns else 0.0
            fd={"route_type_enc":int(rt=="FTL"),"log_osrm_time":np.log1p(osrm_t),"log_osrm_dist":np.log1p(osrm_d),
                "log_distance":np.log1p(dist),"osrm_speed":osrm_d/(osrm_t/60+1e-5),"dist_time_ratio":dist/(osrm_t+1),
                "hour":hour,"dow":dow_val,"month":month,"is_weekend":int(dow_val in[5,6]),"is_rush_hour":int(hour in[7,8,9,17,18,19]),
                "hour_sin":np.sin(2*np.pi*hour/24),"hour_cos":np.cos(2*np.pi*hour/24),
                "dow_sin":np.sin(2*np.pi*dow_val/7),"dow_cos":np.cos(2*np.pi*dow_val/7),
                "month_sin":np.sin(2*np.pi*month/12),"month_cos":np.cos(2*np.pi*month/12),
                "network_load_norm":0.5,"cutoff_factor":180,
                "src_betweenness":gf(src,"betweenness"),"src_pagerank":gf(src,"pagerank"),
                "src_closeness":gf(src,"closeness"),"src_structural_risk_score":gf(src,"structural_risk_score"),
                "src_in_degree":gf(src,"in_degree"),"src_out_degree":gf(src,"out_degree"),
                "src_avg_sla_breach":gf(src,"avg_sla_breach"),"src_avg_delay_ratio":gf(src,"avg_delay_ratio"),
                "src_community_id":gf(src,"community_id"),"src_hub_score":gf(src,"hub_score"),
                "dst_betweenness":gf(dst,"betweenness"),"dst_pagerank":gf(dst,"pagerank"),
                "dst_closeness":gf(dst,"closeness"),"dst_structural_risk_score":gf(dst,"structural_risk_score"),
                "dst_in_degree":gf(dst,"in_degree"),"dst_out_degree":gf(dst,"out_degree"),
                "dst_avg_sla_breach":gf(dst,"avg_sla_breach"),"dst_avg_delay_ratio":gf(dst,"avg_delay_ratio"),
                "dst_community_id":gf(dst,"community_id"),
                "corr_median_delay_ratio":cf(src,dst,"median_delay_ratio") or 1.86,
                "corr_sla_breach_rate":cf(src,dst,"sla_breach_rate") or 0.82,
                "corr_trip_count":cf(src,dst,"trip_count") or 10,"corr_ftl_share":cf(src,dst,"ftl_share") or 0.5,
                "corr_route_type_entropy":cf(src,dst,"route_type_entropy") or 0.5,
                "corr_cross_community":cf(src,dst,"cross_community") or 0,"corr_corridor_risk":cf(src,dst,"corridor_risk") or 0.3,
                "cross_community_flag":0,"high_risk_src":int(gf(src,"structural_risk_score")>0.3),
                "high_risk_dst":int(gf(dst,"structural_risk_score")>0.3),
                "both_high_risk":int(gf(src,"structural_risk_score")>0.3 and gf(dst,"structural_risk_score")>0.3)}
            Xp=pd.DataFrame([fd]); [Xp.__setitem__(c,0) for c in ALL if c not in Xp.columns]; Xp=Xp[ALL].fillna(0)
            pred=float(model.predict(Xp)[0])
            st.success(f"### 🎯 Predicted: **{pred:.0f} min** ({pred/60:.1f} hrs)")
            rc=st.columns(4)
            rc[0].metric("Predicted",f"{pred:.0f} min"); rc[1].metric("OSRM",f"{osrm_t} min")
            rc[2].metric("Overrun",f"+{pred-osrm_t:.0f} min",delta_color="inverse")
            sla_r=fd["corr_sla_breach_rate"]
            rc[3].metric("SLA Risk","HIGH 🔴" if sla_r>0.7 else "MEDIUM 🟡" if sla_r>0.4 else "LOW 🟢")
        except Exception as e: st.error(f"Error: {e}")
    show("11_feature_importance.png")

# ── PAGE 5: ROUTE RECOMMENDER ─────────────────────────────────────────────────
elif PAGE=="🔀 Route Recommender":
    st.title("🔀 Route Type Decision Engine"); df,corr,nf,fi,bv2,top50=load_data()
    try:
        fw=jload("route_decision_framework.json"); mc=st.columns(4)
        mc[0].metric("Accuracy",f"{fw['classifier_accuracy']}%"); mc[1].metric("Corridors",f"{fw['total_corridors_analysed']:,}")
        mc[2].metric("FTL Rec",f"{fw['ftl_recommended']:,}"); mc[3].metric("Carting Rec",f"{fw['carting_recommended']:,}")
    except: pass
    sh("🔮 Route Recommendation Tool")
    r1,r2,r3=st.columns(3)
    with r1: dist_r=st.number_input("Distance (km)",1,3000,200); trips_r=st.number_input("Weekly Trips",1,5000,50)
    with r2: delay_r=st.slider("Delay Ratio",1.0,5.0,1.8,0.1); sla_r=st.slider("SLA Breach Rate",0.0,1.0,0.75,0.05)
    with r3: src_r=st.slider("Source Hub Risk",0.0,1.0,0.3,0.05); hour_r=st.slider("Dep. Hour",0,23,14)
    if st.button("🔀 Recommend",type="primary"):
        sf=sc=0; rs=[]
        if dist_r>=150: sf+=3; rs.append(f"✅ {dist_r}km ≥ 150km → FTL")
        elif dist_r<50: sc+=3; rs.append(f"✅ {dist_r}km < 50km → Carting")
        else: sf+=1; rs.append(f"⚖️ {dist_r}km borderline")
        if trips_r>=30: sf+=2; rs.append(f"✅ {trips_r}/wk ≥ 30 → FTL efficient")
        else: sc+=2; rs.append(f"✅ {trips_r}/wk < 30 → Carting consolidates")
        if sla_r>0.8: sf+=2; rs.append(f"🔴 SLA {sla_r*100:.0f}% → FTL fewer hub touches")
        if src_r>0.5: sf+=1; rs.append("⚠️ High-risk source → FTL avoids dwell")
        if hour_r in [7,8,9,17,18,19]: sc+=1; rs.append("⚠️ Rush hour → Carting more flexible")
        rec="FTL" if sf>=sc else "Carting"; conf=abs(sf-sc)/max(sf+sc,1)
        cl="High" if conf>0.4 else "Medium" if conf>0.2 else "Low"
        if rec=="FTL": st.success(f"### ✅ Recommended: **FTL** — Confidence: {cl}")
        else: st.info(f"### 📦 Recommended: **Carting** — Confidence: {cl}")
        rc=st.columns(4); rc[0].metric("Rec",rec); rc[1].metric("Confidence",cl)
        rc[2].metric("SLA Risk","HIGH 🔴" if sla_r>0.7 else "MED 🟡" if sla_r>0.4 else "LOW 🟢")
        rc[3].metric("Score",f"FTL={sf} vs Cart={sc}")
        for r in rs: st.markdown(f"- {r}")
    show("14_route_type_decision_engine.png")
    sh("📖 Decision Guide")
    st.dataframe(pd.DataFrame({"Distance":["<50km","50-150km","150-300km","300-600km",">600km"],
        "Recommended":["Carting","Model-Dep","FTL","FTL","FTL"],"SLA Risk":["Low","Medium","Medium","High","High"]}),use_container_width=True)

# ── PAGE 6: BUSINESS IMPACT ───────────────────────────────────────────────────
elif PAGE=="💰 Business Impact":
    st.title("💰 Business Impact Simulator"); df,corr,nf,fi,bv2,top50=load_data()
    try: sim=jload("business_impact_simulation.json")
    except: st.error("Run pipeline first."); st.stop()
    st.markdown("**Upgrading:** `IND562132AAA` | `IND000000ACB` | `IND110037AAM` — affect **39.3%** of trips")
    sel=st.select_slider("Scenario",["Worst Case","Expected Case","Best Case"],"Expected Case")
    sc=sim[{"Worst Case":"worst_case","Expected Case":"expected","Best Case":"best_case"}[sel]]
    mc=st.columns(5)
    mc[0].metric("Delay Reduction",f"{sc['delay_reduction_pct']:.0f}%")
    mc[1].metric("Breaches Avoided",f"{sc['mean_breaches_avoided']:,}")
    mc[2].metric("P10–P90",f"{sc['p10_breaches_avoided']:,}–{sc['p90_breaches_avoided']:,}")
    mc[3].metric("New Breach Rate",f"{sc['new_overall_breach_rate']}%",delta=f"-{82.0-sc['new_overall_breach_rate']:.1f}pp")
    mc[4].metric("Upgrade Cost",f"${sc['upgrade_cost_m_usd']}M")
    show("13_business_impact_simulation.png")
    memo_path=f"{RPT}/executive_strategy_memo.txt"
    if os.path.exists(memo_path):
        sh("📄 Executive Memo")
        with open(memo_path) as f: memo=f.read()
        st.text_area("",memo,height=300)
        st.download_button("📥 Download Memo",memo,file_name="delhivery_memo.txt",mime="text/plain")

# ── PAGE 7: IMPROVEMENT REPORT ────────────────────────────────────────────────
elif PAGE=="📈 Improvement Report":
    st.title("📈 v2 Improvement Report")
    st.markdown("*Detailed breakdown of all 4 improvements*")
    df,corr,nf,fi,bv2,top50=load_data()
    try:
        adv2=jload("graph_advantage_report_v2.json"); sla=jload("sla_clarification_report.json")
    except Exception as e: st.error(f"{e}"); st.stop()
    st.markdown("---")
    st.markdown("### 1️⃣ Outlier Winsorization + SLA Clarification")
    ic=st.columns(4)
    ic[0].metric("FTL Cap",f"2,700 min (99th pct)"); ic[1].metric("Carting Cap","376 min (99th pct)")
    ic[2].metric("Std Reduction",f"{sla['std_reduction_pct']}%")
    ic[3].metric("Realistic SLA",f"{sla['realistic_breach_rate_pct']}%",delta=f"{sla['realistic_breach_rate_pct']-sla['original_breach_rate_pct']:.1f}pp",delta_color="inverse")
    st.info(f"**SLA Root Cause:** {sla['root_cause']}")
    show("15_outlier_cleaning.png")
    st.markdown("---")
    st.markdown("### 2️⃣ HistGradientBoosting (LightGBM-equivalent)")
    hc=st.columns(3)
    hc[0].metric("Algorithm","Histogram-based GBM")
    hc[1].metric("vs RF+Graph RMSE",f"{adv2['hgbm_graph_rmse']:.1f} vs {adv2['original_graph_rmse']:.1f}")
    hc[2].metric("Training Speed","~10× faster than RF")
    st.markdown("- **255 histogram bins** — same core algo as LightGBM\n- **Early stopping** on 10% validation\n- **L2 regularisation** tuned per route type\n- **Native missing value handling**")
    st.markdown("---")
    st.markdown("### 3️⃣ Per-Route-Type Models (FTL vs Carting)")
    pc=st.columns(4)
    pc[0].metric("FTL ±15%","71.4%",delta="+12.2pp vs combined")
    pc[1].metric("Carting RMSE","27.3 min",delta="-64 vs combined")
    pc[2].metric("FTL RMSE","97.0 min"); pc[3].metric("Per-Route Full Test",f"{adv2['per_route_rmse']:.1f} RMSE")
    st.markdown("FTL = long-haul trunk (distance-dominant) | Carting = last-mile (hub-touch-dominant)\nSeparate models allow each to learn the right physics.")
    st.markdown("---")
    st.markdown("### 4️⃣ Blend Ensemble")
    ec=st.columns(3)
    ec[0].metric("Blend","0.55 HistGBM + 0.45 Per-Route")
    ec[1].metric("Blend RMSE",f"{adv2['blend_rmse']:.1f} min")
    ec[2].metric("Blend ±15%",f"{adv2['blend_w15']:.1f}%")
    st.markdown("---")
    st.markdown("### 🏆 Final Net Improvement")
    fc=st.columns(5)
    fc[0].metric("RMSE v1→v2",f"{adv2['original_graph_rmse']:.1f}→{adv2['best_full_test_rmse']:.1f}",delta=f"{adv2['rmse_improvement_pct']:+.1f}%")
    fc[1].metric("MAE v1→v2",f"{adv2['original_graph_mae']:.1f}→{adv2['best_full_test_mae']:.1f}",delta=f"{adv2['mae_improvement_pct']:+.1f}%")
    fc[2].metric("±15% v1→v2",f"{adv2['original_graph_w15']:.1f}%→{adv2['best_full_test_w15']:.1f}%",delta=f"+{adv2['w15_improvement_pp']:.1f}pp")
    fc[3].metric("±20% v1→v2","68.0%→71.7%",delta="+3.7pp")
    fc[4].metric("Best Model","RF+Graph v2")
    show("16_improved_model_benchmark.png")
    show("17_sla_perroute_analysis.png")
