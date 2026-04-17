"""
GCC Export Opportunity Dashboard
OCO Global x AUB MSBA Capstone
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="GCC Export Opportunities · OCO Global",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0f2847 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #fff !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f8f9fb;
        border: 1px solid #e8ecf1;
        border-radius: 10px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6b7a8d !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #0f2847 !important;
    }

    /* Page titles */
    h1 {
        color: #0f2847 !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        color: #1a3a5c !important;
        font-weight: 700 !important;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Download button */
    .stDownloadButton button {
        background: #0f2847 !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stDownloadButton button:hover {
        background: #1a4a7a !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DATA = Path("data")
GCC_COLORS = {
    "Bahrain": "#1B9AAA", "Kuwait": "#06D6A0", "Oman": "#EF476F",
    "Qatar": "#8338EC", "Saudi Arabia": "#0F4C75", "United Arab Emirates": "#FF6B35",
}
TRANSPORT_ICONS = {"Sea": "🚢", "Air": "✈️", "Land": "🚛"}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
@st.cache_data
def load(name):
    path = DATA / name
    return pd.read_csv(path) if path.exists() else None


def require(*names):
    frames = {}
    missing = []
    for n in names:
        df = load(n)
        if df is None:
            missing.append(n)
        frames[n] = df
    missing = [n for n in names if frames.get(n) is None]
    if missing:
        st.error(f"Missing: **{', '.join(missing)}**. Place CSVs from notebook Section 45 into `data/`.")
        st.stop()
    return frames if len(names) > 1 else frames[names[0]]


def fmt_usd(val, d=1):
    if pd.isna(val) or val == 0:
        return "—"
    for t, s in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(val) >= t:
            return f"${val/t:.{d}f}{s}"
    return f"${val:,.0f}"


@st.cache_data
def derive_yearly():
    pen = load("gcc_export_penetration.csv")
    if pen is None:
        return None
    return pen.groupby("year").agg(
        total_demand=("world_demand", "sum"),
        total_gcc_exports=("gcc_exports", "sum"),
    ).reset_index()


@st.cache_data
def backtest_metrics():
    bt = load("backtest_results.csv")
    if bt is None:
        return {}
    bt = bt.dropna(subset=["actual", "predicted"]).query("actual > 0")
    if bt.empty:
        return {}
    ss_res = np.sum((bt["actual"] - bt["predicted"]) ** 2)
    ss_tot = np.sum((bt["actual"] - bt["actual"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    bt_sig = bt[bt["actual"] > 1e6]
    mape = (
        np.mean(np.abs((bt_sig["actual"] - bt_sig["predicted"]) / bt_sig["actual"])) * 100
        if len(bt_sig) > 0 else 0
    )
    return {"r2": r2, "mape": mape, "n": len(bt),
            "mae": np.mean(np.abs(bt["actual"] - bt["predicted"]))}


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1.2rem 0 1.5rem'>"
        "<div style='font-size:2rem;font-weight:800;color:#fff;line-height:1.2'>"
        "🌍 GCC Export<br>Opportunities</div>"
        "<div style='font-size:0.7rem;color:#7fa3c9;margin-top:0.5rem;letter-spacing:0.05em'>"
        "OCO GLOBAL · AUB MSBA CAPSTONE</div></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        ["Opportunity Finder", "Executive Summary", "Market Demand",
         "GCC Penetration", "Demand Forecasts"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: UN Comtrade · ITC Trade Map")
    st.caption("HS27 / HS71 / HS93 / HS99 excluded")


# ═══════════════════════════════════════════════════════════════════════════
# OPPORTUNITY FINDER
# ═══════════════════════════════════════════════════════════════════════════
if page == "Opportunity Finder":
    st.title("Opportunity Finder")
    st.markdown(
        "Pick a **GCC exporter** and a **commodity** — the dashboard surfaces "
        "the highest-potential destination markets ranked by a composite opportunity score."
    )
    st.caption(
        "Score = Demand (25%) · Penetration Gap (20%) · Country Viability (20%) "
        "· Landing Cost (15%) · ML Growth Signal (10%) · Price Quality (10%)"
    )

    opp = require("opportunity_rankings_full.csv")

    # --- SELECTORS ---
    col_gcc, col_search = st.columns([1, 2])
    with col_gcc:
        gcc_sel = st.selectbox("GCC Exporter", sorted(opp["gcc_country"].unique()))

    df_gcc = opp[opp["gcc_country"] == gcc_sel].copy()

    # Commodity list for this GCC country
    cmd_scores = (
        df_gcc.groupby(["cmdCode", "commodity"])["opportunity_score"]
        .mean().reset_index().sort_values("opportunity_score", ascending=False)
    )
    cmd_labels = cmd_scores.apply(
        lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1
    ).tolist()
    cmd_code_map = dict(zip(cmd_labels, cmd_scores["cmdCode"]))

    with col_search:
        search = st.text_input("🔍 Filter commodities", "", placeholder="e.g. plastic, aluminium, dairy...")

    if search.strip():
        cmd_labels = [l for l in cmd_labels if search.strip().lower() in l.lower()]

    if not cmd_labels:
        st.warning("No commodities match your search.")
        st.stop()

    cmd_sel = st.selectbox("Commodity", cmd_labels)
    sel_code = cmd_code_map[cmd_sel]

    df = df_gcc[df_gcc["cmdCode"] == sel_code].sort_values("opportunity_score", ascending=False).copy()

    if df.empty:
        st.info("No scored opportunities for this combination.")
        st.stop()

    commodity_name = df["commodity"].iloc[0]

    # --- KPI ROW ---
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Markets Ranked", f"{df['dest_country'].nunique()}")
    k2.metric("Top Score", f"{df['opportunity_score'].max():.3f}")
    avg_pen = df["penetration_pct"].mean() if "penetration_pct" in df.columns else 0
    k3.metric("Avg GCC Penetration", f"{avg_pen:.1f}%")
    if "demand_4y_total" in df.columns:
        k4.metric("4-Year Demand (Total)", fmt_usd(df["demand_4y_total"].sum()))

    # --- TOP 15 BAR CHART ---
    st.divider()
    st.subheader(f"Top 15 Destinations — {commodity_name[:55]}")
    st.caption(f"Exporter: **{gcc_sel}** · Re-exports & saturated markets excluded")

    df_top = df.head(15).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_top["dest_country"],
        x=df_top["opportunity_score"],
        orientation="h",
        marker=dict(
            color=df_top["opportunity_score"],
            colorscale=[[0, "#c8d6e5"], [0.5, "#2e86de"], [1, "#0a3d62"]],
            line=dict(width=0),
            cornerradius=4,
        ),
        text=df_top["opportunity_score"].round(3),
        textposition="outside",
        textfont=dict(size=12, color="#1a3a5c"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.3f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        margin=dict(t=10, b=30, l=10, r=60),
        height=max(380, len(df_top) * 36),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        xaxis=dict(title="Opportunity Score", range=[0, df_top["opportunity_score"].max() * 1.18]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    # --- DETAIL TABLE ---
    st.divider()
    st.subheader("Detailed Breakdown")

    col_map = {
        "dest_country": "Target Market",
        "opportunity_score": "Score",
        "grade": "Grade",
        "demand_4y_total": "4Y Demand",
        "penetration_pct": "GCC Pen %",
        "pen_opportunity": "Entry Room",
        "uv_mean": "UV ($/kg)",
        "uv_cagr": "Price CAGR %",
        "ml_growth_prob": "ML Growth P",
        "dist_km": "Distance (km)",
        "lpi_score": "LPI",
        "mfn_tariff_rate": "Tariff %",
        "recommended_transport": "Transport",
        "opportunity_rationale": "Rationale",
    }
    avail = {k: v for k, v in col_map.items() if k in df_top.columns}
    table = df_top[list(avail.keys())].rename(columns=avail).copy()

    if "4Y Demand" in table.columns:
        table["4Y Demand"] = table["4Y Demand"].apply(lambda x: fmt_usd(x) if pd.notna(x) else "—")
    if "Score" in table.columns:
        table["Score"] = table["Score"].round(3)
    if "GCC Pen %" in table.columns:
        table["GCC Pen %"] = table["GCC Pen %"].round(1)
    if "Entry Room" in table.columns:
        table["Entry Room"] = table["Entry Room"].round(2)
    if "UV ($/kg)" in table.columns:
        table["UV ($/kg)"] = table["UV ($/kg)"].round(2)
    if "Price CAGR %" in table.columns:
        table["Price CAGR %"] = table["Price CAGR %"].round(1)
    if "ML Growth P" in table.columns:
        table["ML Growth P"] = table["ML Growth P"].round(2)
    if "Distance (km)" in table.columns:
        table["Distance (km)"] = table["Distance (km)"].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
    if "LPI" in table.columns:
        table["LPI"] = table["LPI"].round(2)
    if "Tariff %" in table.columns:
        table["Tariff %"] = table["Tariff %"].round(1)
    if "Transport" in table.columns:
        table["Transport"] = table["Transport"].apply(
            lambda x: f"{TRANSPORT_ICONS.get(str(x), '')} {x}" if pd.notna(x) else "—")
    if "Rationale" in table.columns:
        table["Rationale"] = table["Rationale"].str[:90]

    st.dataframe(table, use_container_width=True, hide_index=True,
                 height=min(620, len(df_top) * 42 + 50))

    # --- RATIONALE EXPANDER ---
    if "opportunity_rationale" in df_top.columns:
        with st.expander("📋 Full scoring rationale for each market"):
            for _, row in df_top.iterrows():
                transport = row.get("recommended_transport", "")
                icon = TRANSPORT_ICONS.get(str(transport), "")
                st.markdown(
                    f"**{row['dest_country']}** · "
                    f"Score **{row['opportunity_score']:.3f}** · "
                    f"Grade {row.get('grade', '—')} · {icon} {transport}"
                )
                st.caption(row.get("opportunity_rationale", "—"))
                st.markdown("---")

    # --- DOWNLOAD ---
    st.divider()
    dl_cols = [c for c in list(col_map.keys()) if c in df.columns]
    st.download_button(
        "⬇️ Download full results as CSV",
        df[dl_cols].to_csv(index=False).encode("utf-8"),
        f"opportunities_{gcc_sel.replace(' ', '_')}_{sel_code}.csv",
        "text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Executive Summary":
    st.title("Executive Summary")
    st.markdown("Where should GCC countries focus non-fuel export efforts over the next 3–5 years?")

    opp = require("opportunity_rankings_full.csv")
    yearly = derive_yearly()
    bm = backtest_metrics()

    latest_yr = int(yearly["year"].max()) if yearly is not None else "—"
    total_demand = yearly.loc[yearly["year"] == latest_yr, "total_demand"].iloc[0] if yearly is not None else 0
    total_gcc = yearly.loc[yearly["year"] == latest_yr, "total_gcc_exports"].iloc[0] if yearly is not None else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Addressable Import Demand", fmt_usd(total_demand), f"{latest_yr}")
    c2.metric("GCC Non-Fuel Exports", fmt_usd(total_gcc), f"{latest_yr}")
    c3.metric("Scored Opportunities", f"{len(opp):,}", f"{opp['dest_country'].nunique()} markets")
    c4.metric("Best Score", f"{opp['opportunity_score'].max():.3f}", "out of 1.000")

    st.divider()
    st.subheader("Top Opportunity per GCC Country")
    top1 = (
        opp.sort_values("opportunity_score", ascending=False)
        .groupby("gcc_country").first().reset_index()
    )
    display = top1[["gcc_country", "dest_country", "commodity", "opportunity_score", "grade"]].copy()
    display.columns = ["GCC Exporter", "Best Target", "Top Commodity", "Score", "Grade"]
    display["Top Commodity"] = display["Top Commodity"].str[:55]
    display["Score"] = display["Score"].round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    col_l, col_r = st.columns(2)
    if yearly is not None:
        with col_l:
            st.subheader("Import Demand — 40 Destinations")
            yearly["d_B"] = yearly["total_demand"] / 1e9
            fig = px.area(yearly, x="year", y="d_B",
                          labels={"d_B": "USD (Billions)", "year": ""},
                          color_discrete_sequence=["#0F4C75"])
            fig.update_layout(margin=dict(t=10, b=30), height=300, showlegend=False,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.subheader("GCC Non-Fuel Exports")
            yearly["g_B"] = yearly["total_gcc_exports"] / 1e9
            fig2 = px.area(yearly, x="year", y="g_B",
                           labels={"g_B": "USD (Billions)", "year": ""},
                           color_discrete_sequence=["#FF6B35"])
            fig2.update_layout(margin=dict(t=10, b=30), height=300, showlegend=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    if bm:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast MAPE", f"{bm['mape']:.1f}%", "on $1M+ series")
        c2.metric("Forecast R²", f"{bm['r2']:.3f}", f"N = {bm['n']:,}")
        c3.metric("Forecast MAE", fmt_usd(bm["mae"]))


# ═══════════════════════════════════════════════════════════════════════════
# MARKET DEMAND
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Market Demand":
    st.title("Destination Market Demand")
    st.markdown("What do the 40 destination countries import — and which commodities matter most?")

    pen = require("gcc_export_penetration.csv")

    yearly = pen.groupby("year")["world_demand"].sum().reset_index()
    yearly["d_B"] = yearly["world_demand"] / 1e9

    st.subheader("Total Addressable Import Demand Over Time")
    fig = px.bar(yearly, x="year", y="d_B",
                 labels={"d_B": "USD (Billions)", "year": ""},
                 color_discrete_sequence=["#0F4C75"])
    fig.update_layout(margin=dict(t=10), height=320,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Top 20 Commodities by Import Demand")
    top_cmd = (
        pen.groupby(["cmdCode", "commodity"])["world_demand"]
        .sum().reset_index().sort_values("world_demand", ascending=False).head(20)
    )
    top_cmd["label"] = top_cmd["cmdCode"].astype(str) + " — " + top_cmd["commodity"].str[:45]
    top_cmd["d_B"] = top_cmd["world_demand"] / 1e9
    fig2 = px.bar(top_cmd, y="label", x="d_B", orientation="h",
                  labels={"d_B": "USD (Billions)", "label": ""},
                  color_discrete_sequence=["#1B9AAA"])
    fig2.update_layout(margin=dict(t=10, l=10), height=540,
                       yaxis=dict(autorange="reversed"),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Commodity Demand Trend")
    cmd_opts = pen.groupby(["cmdCode", "commodity"])["world_demand"].sum().reset_index().sort_values("world_demand", ascending=False)
    labels = cmd_opts.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
    code_map = dict(zip(labels, cmd_opts["cmdCode"]))
    selected = st.selectbox("Select a commodity", labels[:80])
    sel_code = code_map[selected]
    trend = pen[pen["cmdCode"] == sel_code].groupby("year")["world_demand"].sum().reset_index()
    trend["d_B"] = trend["world_demand"] / 1e9
    cname = pen.loc[pen["cmdCode"] == sel_code, "commodity"].iloc[0]
    fig3 = px.line(trend, x="year", y="d_B", markers=True,
                   labels={"d_B": "USD (B)", "year": ""},
                   color_discrete_sequence=["#0F4C75"], title=cname)
    fig3.update_layout(margin=dict(t=40), height=320,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# GCC PENETRATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "GCC Penetration":
    st.title("GCC Export Penetration")
    st.markdown("**Penetration %** = GCC exports / import demand. Low penetration + high demand = opportunity.")

    pen = require("gcc_export_penetration.csv")
    latest_yr = int(pen["year"].max())
    snap = pen[pen["year"] == latest_yr].copy()

    st.subheader("GCC Non-Fuel Exports Over Time")
    gcc_yr = pen.groupby("year")["gcc_exports"].sum().reset_index()
    gcc_yr["g_B"] = gcc_yr["gcc_exports"] / 1e9
    fig = px.area(gcc_yr, x="year", y="g_B",
                  labels={"g_B": "GCC Exports (USD B)", "year": ""},
                  color_discrete_sequence=["#FF6B35"])
    fig.update_layout(margin=dict(t=10), height=280, showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(f"Highest Penetration ({latest_yr})")
        high = snap.sort_values("penetration_pct", ascending=False).head(15).copy()
        high["label"] = high["cmdCode"].astype(str) + " — " + high["commodity"].str[:40]
        fig2 = px.bar(high, y="label", x="penetration_pct", orientation="h",
                      labels={"penetration_pct": "GCC Pen %", "label": ""},
                      color_discrete_sequence=["#8338EC"])
        fig2.update_layout(margin=dict(t=10, l=10), height=420,
                           yaxis=dict(autorange="reversed"),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    with col_r:
        st.subheader(f"Biggest Gaps ({latest_yr})")
        st.caption("High demand + low GCC penetration (<5%).")
        gaps = snap[(snap["penetration_pct"] < 5) &
                    (snap["world_demand"] > snap["world_demand"].quantile(0.5))]
        gaps = gaps.sort_values("world_demand", ascending=False).head(15).copy()
        if not gaps.empty:
            gaps["label"] = gaps["cmdCode"].astype(str) + " — " + gaps["commodity"].str[:40]
            gaps["d_B"] = gaps["world_demand"] / 1e9
            fig3 = px.bar(gaps, y="label", x="d_B", orientation="h",
                          labels={"d_B": "Demand (USD B)", "label": ""},
                          color_discrete_sequence=["#D62828"])
            fig3.update_layout(margin=dict(t=10, l=10), height=420,
                               yaxis=dict(autorange="reversed"),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Penetration Trend")
    cmd_opts = pen.groupby(["cmdCode", "commodity"])["world_demand"].sum().reset_index().sort_values("world_demand", ascending=False)
    labels = cmd_opts.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
    code_map = dict(zip(labels, cmd_opts["cmdCode"]))
    selected = st.selectbox("Select a commodity", labels[:80])
    sel_code = code_map[selected]
    pt = pen[pen["cmdCode"] == sel_code].sort_values("year")
    fig4 = px.line(pt, x="year", y="penetration_pct", markers=True,
                   labels={"penetration_pct": "GCC Penetration %", "year": ""},
                   color_discrete_sequence=["#8338EC"])
    fig4.update_layout(margin=dict(t=10), height=300,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# DEMAND FORECASTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Demand Forecasts":
    st.title("Demand Forecasts (2025–2028)")
    st.markdown("Holt-Winters forecasts per commodity. Confidence band: ±5% yr1 → ±20% yr4.")

    files = require("demand_forecast_global.csv", "gcc_export_penetration.csv")
    fc = files["demand_forecast_global.csv"]
    pen = files["gcc_export_penetration.csv"]
    hist = pen.groupby(["cmdCode", "commodity", "year"])["world_demand"].sum().reset_index()

    fc_totals = (
        fc.groupby(["cmdCode", "commodity"])["demand_ensemble"]
        .sum().reset_index().sort_values("demand_ensemble", ascending=False)
    )
    labels = fc_totals.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
    code_map = dict(zip(labels, fc_totals["cmdCode"]))
    selected = st.selectbox("Select a commodity", labels[:80])
    sel_code = code_map[selected]

    h = hist[hist["cmdCode"] == sel_code].sort_values("year")
    f = fc[fc["cmdCode"] == sel_code].sort_values("year")
    cname = f["commodity"].iloc[0] if len(f) > 0 else sel_code

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h["year"], y=h["world_demand"], mode="lines+markers", name="Historical",
        line=dict(color="#0F4C75", width=2.5), marker=dict(size=6)))

    if not f.empty and not h.empty:
        bridge_yr = h["year"].max()
        bv = h.loc[h["year"] == bridge_yr, "world_demand"]
        if not bv.empty:
            f_ext = pd.concat([
                pd.DataFrame({"year": [bridge_yr], "demand_ensemble": [bv.iloc[0]],
                               "ci_lower": [bv.iloc[0]], "ci_upper": [bv.iloc[0]]}),
                f[["year", "demand_ensemble", "ci_lower", "ci_upper"]],
            ], ignore_index=True)
        else:
            f_ext = f[["year", "demand_ensemble", "ci_lower", "ci_upper"]]
        fig.add_trace(go.Scatter(
            x=f_ext["year"], y=f_ext["demand_ensemble"], mode="lines+markers", name="Forecast",
            line=dict(color="#D62828", width=2.5, dash="dash"), marker=dict(size=6)))
        fig.add_trace(go.Scatter(
            x=pd.concat([f_ext["year"], f_ext["year"][::-1]]),
            y=pd.concat([f_ext["ci_upper"], f_ext["ci_lower"][::-1]]),
            fill="toself", fillcolor="rgba(214,40,40,0.10)",
            line=dict(width=0), name="Confidence Band"))

    fig.update_layout(title=f"Global Demand Forecast — {cname}",
                      yaxis_title="Import Demand (USD)", xaxis_title="",
                      margin=dict(t=50), height=430,
                      legend=dict(orientation="h", y=-0.12),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    if not f.empty:
        st.subheader("Forecast Values")
        tbl = f[["year", "demand_ensemble", "ci_lower", "ci_upper"]].copy()
        tbl.columns = ["Year", "Forecast", "Lower Bound", "Upper Bound"]
        for c in ["Forecast", "Lower Bound", "Upper Bound"]:
            tbl[c] = tbl[c].apply(fmt_usd)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Largest Forecasted Markets (4-Year Total)")
    top_fc = fc_totals.head(15).copy()
    top_fc["label"] = top_fc["cmdCode"].astype(str) + " — " + top_fc["commodity"].str[:45]
    top_fc["d_B"] = top_fc["demand_ensemble"] / 1e9
    fig2 = px.bar(top_fc, y="label", x="d_B", orientation="h",
                  labels={"d_B": "4Y Forecast (USD B)", "label": ""},
                  color_discrete_sequence=["#D62828"])
    fig2.update_layout(margin=dict(t=10, l=10), height=430,
                       yaxis=dict(autorange="reversed"),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
