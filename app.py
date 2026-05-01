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

    /* Typography */
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

    /* Buttons */
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

    /* Selectbox labels */
    .stSelectbox label, .stTextInput label {
        font-weight: 600;
        color: #1a3a5c;
    }

    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DATA = Path("data")
TRANSPORT_ICONS = {"Sea": "🚢", "Air": "✈️", "Land": "🚛"}

# Shared chart palette
BLUE_SCALE = [[0, "#c8d6e5"], [0.5, "#2e86de"], [1, "#0a3d62"]]
TEAL_SCALE = [[0, "#c8e6e5"], [0.5, "#1B9AAA"], [1, "#0d5c63"]]
ORANGE_SCALE = [[0, "#fde8d0"], [0.5, "#FF6B35"], [1, "#c4420a"]]
PURPLE_SCALE = [[0, "#e0d4f5"], [0.5, "#8338EC"], [1, "#4a1d8e"]]
RED_SCALE = [[0, "#f5d4d4"], [0.5, "#D62828"], [1, "#8b1a1a"]]

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, sans-serif"),
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
@st.cache_data
def load(name):
    path = DATA / name
    return pd.read_csv(path) if path.exists() else None


# Columns the dashboard actually uses from opportunity_rankings_full.csv.
# Loading only these cuts parse time and memory significantly on large files.
_OPP_COLS = [
    "gcc_country", "cmdCode", "commodity", "dest_country", "opportunity_score",
    "grade", "demand_4y_total", "penetration_pct", "pen_opportunity",
    "ml_growth_prob", "uv_mean", "uv_cagr",
    "weighted_dist_km", "dist_km",          # keep both; one may be absent
    "lpi_score", "mfn_tariff_rate",
    "recommended_transport", "opportunity_rationale",
]

_OPP_DTYPES = {
    "gcc_country": "category",
    "dest_country": "category",
    "commodity": "category",
    "grade": "category",
    "recommended_transport": "category",
    "cmdCode": "int32",
    "opportunity_score": "float32",
    "demand_4y_total": "float32",
    "penetration_pct": "float32",
    "pen_opportunity": "float32",
    "ml_growth_prob": "float32",
    "uv_mean": "float32",
    "uv_cagr": "float32",
    "weighted_dist_km": "float32",
    "dist_km": "float32",
    "lpi_score": "float32",
    "mfn_tariff_rate": "float32",
}


@st.cache_data(show_spinner="Loading opportunity rankings…")
def load_opp():
    """
    Load opportunity_rankings_full.csv with memory-efficient dtypes,
    restrict to needed columns, and deduplicate to one row per
    (gcc_country, cmdCode, dest_country) — keeping the highest score.

    The file may be large (90+ MB) if the notebook produced yearly rows
    or HS4-level rows; deduplication ensures the dashboard always shows
    the single best score per opportunity triplet.
    """
    path = DATA / "opportunity_rankings_full.csv"
    if not path.exists():
        return None

    # Read only columns that exist in the file to avoid KeyErrors on
    # optional columns (weighted_dist_km, dist_km, etc.)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use_cols = [c for c in _OPP_COLS if c in header]
    dtype_map = {k: v for k, v in _OPP_DTYPES.items() if k in use_cols}

    df = pd.read_csv(path, usecols=use_cols, dtype=dtype_map,
                     low_memory=False)

    # Deduplicate: one row per (gcc_country, cmdCode, dest_country).
    # If the notebook produced one row per year or multiple model runs,
    # keep the row with the highest opportunity_score.
    key = ["gcc_country", "cmdCode", "dest_country"]
    before = len(df)
    df = (df.sort_values("opportunity_score", ascending=False)
            .drop_duplicates(subset=key, keep="first")
            .reset_index(drop=True))
    after = len(df)

    if before != after:
        st.sidebar.caption(
            f"ℹ️ {before - after:,} duplicate opportunity rows removed "
            f"(kept highest score per market)."
        )

    return df


def require(*names):
    frames = {}
    missing = []
    for n in names:
        frames[n] = load(n)
        if frames[n] is None:
            missing.append(n)
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


def hbar(labels, values, colorscale=BLUE_SCALE, text_fmt=None, height=None, x_title=""):
    """Polished horizontal bar chart matching the Opportunity Finder style."""
    if text_fmt is None:
        text_fmt = [f"{v:.3f}" for v in values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=values, colorscale=colorscale, line=dict(width=0), cornerradius=4),
        text=text_fmt, textposition="outside",
        textfont=dict(size=11, color="#1a3a5c"),
        hovertemplate="<b>%{y}</b><br>%{x:,.2f}<extra></extra>",
    ))
    h = height or max(350, len(labels) * 34)
    fig.update_layout(
        **CHART_LAYOUT,
        margin=dict(t=10, b=30, l=10, r=70), height=h,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        xaxis=dict(title=x_title, range=[0, max(values) * 1.2] if len(values) > 0 else None),
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    return fig


def area_chart(x, y, color="#0F4C75", y_label="", height=300):
    """Polished area chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", fill="tozeroy",
        line=dict(color=color, width=2.5),
        fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if "rgb" in color
                  else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
        hovertemplate="%{x}<br>" + y_label + ": %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        margin=dict(t=10, b=30, l=10, r=10), height=height,
        yaxis=dict(title=y_label, gridcolor="rgba(0,0,0,0.05)"),
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0.05)"),
        showlegend=False,
    )
    return fig


def line_chart(x, y, color="#0F4C75", y_label="", title="", height=320):
    """Polished line chart with markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color, line=dict(color="white", width=1.5)),
        hovertemplate="%{x}<br>" + y_label + ": %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        margin=dict(t=40 if title else 10, b=30, l=10, r=10), height=height,
        title=dict(text=title, font=dict(size=14, color="#1a3a5c")) if title else None,
        yaxis=dict(title=y_label, gridcolor="rgba(0,0,0,0.05)"),
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0.05)"),
        showlegend=False,
    )
    return fig


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
    st.caption("Data: UN Comtrade")
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
        "Score = Demand Forecast (30%) · Penetration Gap (20%) · Country Viability (20%) "
        "· ML Growth Signal (15%) · Price Quality (10%) · Landing Cost (5%)"
    )

    opp = load_opp()
    if opp is None:
        st.error("Missing: **opportunity_rankings_full.csv**. Place it into `data/`.")
        st.stop()

    col_gcc, col_search = st.columns([1, 2])
    with col_gcc:
        gcc_sel = st.selectbox("GCC Exporter", sorted(opp["gcc_country"].unique()))

    df_gcc = opp[opp["gcc_country"] == gcc_sel].copy()
    cmd_scores = (
        df_gcc.groupby(["cmdCode", "commodity"])["opportunity_score"]
        .max().reset_index().sort_values("opportunity_score", ascending=False)
    )
    cmd_labels = cmd_scores.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
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

    # KPIs
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Markets Ranked", f"{df['dest_country'].nunique()}")
    k2.metric("Top Score", f"{df['opportunity_score'].max():.3f}")
    avg_pen = df["penetration_pct"].mean() if "penetration_pct" in df.columns else 0
    k3.metric("Avg GCC Penetration", f"{avg_pen:.1f}%")
    if "demand_4y_total" in df.columns:
        k4.metric("4-Year Demand (Total)", fmt_usd(df["demand_4y_total"].sum()))

    # Top 15 bar chart
    st.divider()
    st.subheader(f"Top 15 Destinations — {commodity_name[:55]}")
    st.caption(f"Exporter: **{gcc_sel}** · Re-exports & saturated markets excluded")
    df_top = df.head(15)
    fig = hbar(
        df_top["dest_country"], df_top["opportunity_score"],
        colorscale=BLUE_SCALE,
        text_fmt=[f"{v:.3f}" for v in df_top["opportunity_score"]],
        x_title="Opportunity Score",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail table
    st.divider()
    st.subheader("Detailed Breakdown")
    col_map = {
        "dest_country": "Target Market", "opportunity_score": "Score", "grade": "Grade",
        "demand_4y_total": "4Y Demand", "penetration_pct": "GCC Pen %",
        "pen_opportunity": "Entry Room", "ml_growth_prob": "ML Growth P",
        "uv_mean": "UV ($/kg)", "uv_cagr": "Price CAGR %",
        "lpi_score": "LPI", "mfn_tariff_rate": "Tariff %",
        "weighted_dist_km": "Distance (km)", "dist_km": "Distance (km)",
        "recommended_transport": "Transport", "opportunity_rationale": "Rationale",
    }
    # prefer weighted_dist_km; drop dist_km if both present to avoid duplicate columns
    if "weighted_dist_km" in df_top.columns and "dist_km" in df_top.columns:
        col_map.pop("dist_km", None)
    elif "weighted_dist_km" not in df_top.columns:
        col_map.pop("weighted_dist_km", None)
    avail = {k: v for k, v in col_map.items() if k in df_top.columns}
    table = df_top[list(avail.keys())].rename(columns=avail).copy()
    if "4Y Demand" in table.columns:
        table["4Y Demand"] = table["4Y Demand"].apply(lambda x: fmt_usd(x) if pd.notna(x) else "—")
    for c, r in [("Score", 3), ("GCC Pen %", 1), ("Entry Room", 2), ("UV ($/kg)", 2),
                  ("Price CAGR %", 1), ("ML Growth P", 2), ("LPI", 2), ("Tariff %", 1)]:
        if c in table.columns:
            table[c] = table[c].round(r)
    if "Distance (km)" in table.columns:
        table["Distance (km)"] = table["Distance (km)"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
    if "Transport" in table.columns:
        table["Transport"] = table["Transport"].apply(
            lambda x: f"{TRANSPORT_ICONS.get(str(x), '')} {x}" if pd.notna(x) else "—")
    if "Rationale" in table.columns:
        table["Rationale"] = table["Rationale"].str[:90]
    st.dataframe(table, use_container_width=True, hide_index=True, height=min(620, len(df_top) * 42 + 50))

    # Rationale expander
    if "opportunity_rationale" in df_top.columns:
        with st.expander("📋 Full scoring rationale for each market"):
            for _, row in df_top.iterrows():
                transport = row.get("recommended_transport", "")
                icon = TRANSPORT_ICONS.get(str(transport), "")
                st.markdown(
                    f"**{row['dest_country']}** · Score **{row['opportunity_score']:.3f}** · "
                    f"Grade {row.get('grade', '—')} · {icon} {transport}")
                st.caption(row.get("opportunity_rationale", "—"))
                st.markdown("---")

    # Download
    st.divider()
    dl_cols = [c for c in avail if c in df.columns]
    st.download_button(
        "⬇️ Download full results as CSV",
        df[dl_cols].to_csv(index=False).encode("utf-8"),
        f"opportunities_{gcc_sel.replace(' ', '_')}_{sel_code}.csv", "text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Executive Summary":
    st.title("Executive Summary")
    st.markdown("Where should GCC countries focus non-fuel export efforts over the next 3–5 years?")

    opp = load_opp()
    if opp is None:
        st.error("Missing: **opportunity_rankings_full.csv**. Place it into `data/`.")
        st.stop()
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

    # Trend charts
    st.divider()
    col_l, col_r = st.columns(2)
    if yearly is not None:
        with col_l:
            st.subheader("Import Demand — 40 Destinations")
            yearly["d_B"] = yearly["total_demand"] / 1e9
            st.plotly_chart(
                area_chart(yearly["year"], yearly["d_B"], "#0F4C75", "USD (Billions)"),
                use_container_width=True,
            )
        with col_r:
            st.subheader("GCC Non-Fuel Exports")
            yearly["g_B"] = yearly["total_gcc_exports"] / 1e9
            st.plotly_chart(
                area_chart(yearly["year"], yearly["g_B"], "#FF6B35", "USD (Billions)"),
                use_container_width=True,
            )

    # Model reliability
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

    # Yearly trend
    yearly = pen.groupby("year")["world_demand"].sum().reset_index()
    yearly["d_B"] = yearly["world_demand"] / 1e9

    st.subheader("Total Addressable Import Demand Over Time")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["d_B"],
        marker=dict(color=yearly["d_B"], colorscale=BLUE_SCALE, cornerradius=4, line=dict(width=0)),
        text=[f"${v:.0f}B" for v in yearly["d_B"]], textposition="outside",
        textfont=dict(size=11, color="#1a3a5c"),
        hovertemplate="%{x}<br>$%{y:.1f}B<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, margin=dict(t=10, b=30), height=340,
                      yaxis=dict(title="USD (Billions)", gridcolor="rgba(0,0,0,0.05)"),
                      xaxis=dict(title=""))
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Top 20 commodities
    st.divider()
    st.subheader("Top 20 Commodities by Import Demand")
    top_cmd = (
        pen.groupby(["cmdCode", "commodity"])["world_demand"]
        .sum().reset_index().sort_values("world_demand", ascending=False).head(20)
    )
    top_cmd["label"] = top_cmd["cmdCode"].astype(str) + " — " + top_cmd["commodity"].str[:45]
    top_cmd["d_B"] = top_cmd["world_demand"] / 1e9
    fig2 = hbar(
        top_cmd["label"], top_cmd["d_B"], colorscale=TEAL_SCALE,
        text_fmt=[f"${v:.1f}B" for v in top_cmd["d_B"]],
        height=560, x_title="USD (Billions)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Single commodity trend
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
    st.plotly_chart(
        line_chart(trend["year"], trend["d_B"], "#0F4C75", "USD (Billions)", cname),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# GCC PENETRATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "GCC Penetration":
    st.title("GCC Export Penetration")
    st.markdown("**Penetration %** = Combined GCC exports / destination import demand. Low penetration + high demand = opportunity. Figures aggregate all 6 GCC member states.")

    pen = require("gcc_export_penetration.csv")
    latest_yr = int(pen["year"].max())
    snap = pen[pen["year"] == latest_yr].copy()

    # Aggregate trend
    gcc_yr = pen.groupby("year")["gcc_exports"].sum().reset_index()
    gcc_yr["g_B"] = gcc_yr["gcc_exports"] / 1e9

    st.subheader("GCC Non-Fuel Exports Over Time")
    st.plotly_chart(
        area_chart(gcc_yr["year"], gcc_yr["g_B"], "#FF6B35", "USD (Billions)", height=280),
        use_container_width=True,
    )

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(f"Highest Penetration ({latest_yr})")
        st.caption("Commodities where GCC already holds significant market share.")
        high = snap.sort_values("penetration_pct", ascending=False).head(15).copy()
        high["label"] = high["cmdCode"].astype(str) + " — " + high["commodity"].str[:40]
        fig2 = hbar(
            high["label"], high["penetration_pct"], colorscale=PURPLE_SCALE,
            text_fmt=[f"{v:.1f}%" for v in high["penetration_pct"]],
            x_title="GCC Penetration %",
        )
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
            fig3 = hbar(
                gaps["label"], gaps["d_B"], colorscale=RED_SCALE,
                text_fmt=[f"${v:.1f}B" for v in gaps["d_B"]],
                x_title="Import Demand (USD B)",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # Penetration trend
    st.divider()
    st.subheader("Penetration Trend Over Time")
    cmd_opts = pen.groupby(["cmdCode", "commodity"])["world_demand"].sum().reset_index().sort_values("world_demand", ascending=False)
    labels = cmd_opts.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
    code_map = dict(zip(labels, cmd_opts["cmdCode"]))
    selected = st.selectbox("Select a commodity", labels[:80])
    sel_code = code_map[selected]
    pt = pen[pen["cmdCode"] == sel_code].sort_values("year")
    st.plotly_chart(
        line_chart(pt["year"], pt["penetration_pct"], "#8338EC", "GCC Penetration %"),
        use_container_width=True,
    )


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

    # Forecast chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h["year"], y=h["world_demand"], mode="lines+markers", name="Historical",
        line=dict(color="#0F4C75", width=2.5),
        marker=dict(size=7, color="#0F4C75", line=dict(color="white", width=1.5)),
    ))
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
            line=dict(color="#D62828", width=2.5, dash="dash"),
            marker=dict(size=7, color="#D62828", line=dict(color="white", width=1.5)),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([f_ext["year"], f_ext["year"][::-1]]),
            y=pd.concat([f_ext["ci_upper"], f_ext["ci_lower"][::-1]]),
            fill="toself", fillcolor="rgba(214,40,40,0.08)",
            line=dict(width=0), name="Confidence Band",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"Global Demand Forecast — {cname}", font=dict(size=14, color="#1a3a5c")),
        yaxis=dict(title="Import Demand (USD)", gridcolor="rgba(0,0,0,0.05)"),
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0.05)"),
        margin=dict(t=50, b=30), height=430,
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    if not f.empty:
        st.subheader("Forecast Values")
        tbl = f[["year", "demand_ensemble", "ci_lower", "ci_upper"]].copy()
        tbl.columns = ["Year", "Forecast", "Lower Bound", "Upper Bound"]
        for c in ["Forecast", "Lower Bound", "Upper Bound"]:
            tbl[c] = tbl[c].apply(fmt_usd)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Top forecasted markets
    st.divider()
    st.subheader("Largest Forecasted Markets (4-Year Total)")
    top_fc = fc_totals.head(15).copy()
    top_fc["label"] = top_fc["cmdCode"].astype(str) + " — " + top_fc["commodity"].str[:45]
    top_fc["d_B"] = top_fc["demand_ensemble"] / 1e9
    fig2 = hbar(
        top_fc["label"], top_fc["d_B"], colorscale=RED_SCALE,
        text_fmt=[f"${v:.1f}B" for v in top_fc["d_B"]],
        height=430, x_title="4-Year Forecast (USD B)",
    )
    st.plotly_chart(fig2, use_container_width=True)