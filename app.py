"""
GCC Export Opportunity Dashboard
OCO Global x AUB MSBA Capstone
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Trade Opportunity Engine · OCO Global",
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

    /* Hero title — must outrank the h1 rule above */
    .hero-title {
        color: #ffffff !important;
        font-size: 2.6rem !important;
        font-weight: 900 !important;
        line-height: 1.15 !important;
        margin: 0 0 1rem !important;
        letter-spacing: -0.02em !important;
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
    if pen is None or "year" not in pen.columns:
        # pen_scored (new notebook output) is a 2022-2023 average with no year
        # column — year-level historical aggregation is not possible from this file.
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
        "<div style='text-align:center;padding:1.4rem 0 1.6rem'>"
        "<div style='font-size:0.65rem;font-weight:700;color:#4a90c4;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem'>"
        "OCO Global · AUB MSBA Capstone</div>"
        "<div style='font-size:1.35rem;font-weight:800;color:#fff;line-height:1.3;letter-spacing:-0.01em'>"
        "Trade Opportunity<br>Engine</div>"
        "<div style='width:40px;height:2px;background:linear-gradient(90deg,#2e86de,#1B9AAA);margin:0.8rem auto 0;border-radius:2px'></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    _pages = ["Home", "Opportunity Finder", "Executive Summary", "Market Demand",
              "GCC Penetration", "Demand Forecasts"]

    # Allow Home page module cards to trigger navigation via session state
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Home"

    page = st.radio(
        "Navigate",
        _pages,
        index=_pages.index(st.session_state["nav_page"]),
        key="sidebar_nav",
        label_visibility="collapsed",
    )
    # Keep session state in sync with manual sidebar clicks
    st.session_state["nav_page"] = page

    st.divider()
    st.caption("Data: UN Comtrade")
    st.caption("HS27 / HS71 / HS93 / HS99 excluded")


# ═══════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "Home":

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,#0a1628 0%,#0f2847 55%,#0d3d6b 100%);
            border-radius:16px;padding:3rem 3.2rem 2.6rem;margin-bottom:0.5rem;
            box-shadow:0 4px 24px rgba(10,22,40,0.18);'>
  <div style='font-size:0.68rem;font-weight:700;color:#4a90c4;letter-spacing:0.22em;
              text-transform:uppercase;margin-bottom:0.9rem;'>
    OCO Global &nbsp;·&nbsp; AUB MSBA Capstone &nbsp;·&nbsp; Spring 2026
  </div>
  <div class='hero-title'>
    GCC Export<br>Opportunity Engine
  </div>
  <p style='font-size:1.05rem;color:#b8cde0;max-width:680px;line-height:1.75;margin:0 0 2rem;'>
    A decision-support platform that identifies, scores, and ranks the highest-potential
    non-fuel export opportunities for all six GCC member states — powered by demand
    forecasting, machine learning classifiers, and a multi-criteria composite scoring framework.
  </p>
  <div style='display:flex;gap:2.5rem;flex-wrap:wrap;'>
    <div style='text-align:center;'>
      <div style='font-size:2rem;font-weight:800;color:#ffffff;line-height:1;'>6</div>
      <div style='font-size:0.72rem;color:#7aabcf;text-transform:uppercase;
                  letter-spacing:0.1em;margin-top:0.3rem;'>GCC Exporters</div>
    </div>
    <div style='width:1px;background:rgba(255,255,255,0.12);'></div>
    <div style='text-align:center;'>
      <div style='font-size:2rem;font-weight:800;color:#ffffff;line-height:1;'>34</div>
      <div style='font-size:0.72rem;color:#7aabcf;text-transform:uppercase;
                  letter-spacing:0.1em;margin-top:0.3rem;'>Destination Markets</div>
    </div>
    <div style='width:1px;background:rgba(255,255,255,0.12);'></div>
    <div style='text-align:center;'>
      <div style='font-size:2rem;font-weight:800;color:#ffffff;line-height:1;'>10 yrs</div>
      <div style='font-size:0.72rem;color:#7aabcf;text-transform:uppercase;
                  letter-spacing:0.1em;margin-top:0.3rem;'>Trade History</div>
    </div>
    <div style='width:1px;background:rgba(255,255,255,0.12);'></div>
    <div style='text-align:center;'>
      <div style='font-size:2rem;font-weight:800;color:#ffffff;line-height:1;'>4 yrs</div>
      <div style='font-size:0.72rem;color:#7aabcf;text-transform:uppercase;
                  letter-spacing:0.1em;margin-top:0.3rem;'>Demand Forecast</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Module navigator cards ────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:700;color:#0f2847;margin-bottom:1rem;'>"
        "Dashboard Modules</div>",
        unsafe_allow_html=True,
    )

    modules = [
        ("#2e86de", "🔍", "Opportunity Finder",
         "Core analytical tool",
         "Select a GCC exporter and commodity to surface the top 15 ranked destination markets, "
         "a choropleth map, and a full score-components breakdown with transport recommendation."),
        ("#1B9AAA", "📊", "Executive Summary",
         "Strategic overview",
         "Headline KPIs, the highest-scored opportunity per GCC member state, and macroeconomic "
         "trend charts covering 2015–2024 import demand and GCC export performance."),
        ("#8338EC", "🌍", "Market Demand",
         "Global import landscape",
         "Top 20 commodity sectors by import value, dual-axis annual demand chart with YoY growth, "
         "and a per-commodity historical demand trend viewer."),
        ("#FF6B35", "📈", "GCC Penetration",
         "Whitespace analysis",
         "Sectors where GCC already dominates versus high-demand sectors where GCC holds less than "
         "5% market share — the clearest signal of untapped diversification potential."),
        ("#D62828", "📉", "Demand Forecasts",
         "4-year projections (2025–2028)",
         "Holt-Winters forecasts with 95% confidence intervals per commodity, filterable by GCC "
         "exporter. Includes a projected demand table and a ranking of top forecast opportunities."),
    ]

    col_a, col_b = st.columns(2)
    for i, (color, icon, title, subtitle, desc) in enumerate(modules):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            st.markdown(
                f"<div style='background:#ffffff;border:1px solid #e4eaf2;border-radius:12px;"
                f"padding:1.4rem 1.6rem;margin-bottom:0.4rem;"
                f"border-left:4px solid {color};'>"
                f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;'>"
                f"<span style='font-size:1.4rem;'>{icon}</span>"
                f"<div>"
                f"<div style='font-size:0.95rem;font-weight:700;color:#0f2847;'>{title}</div>"
                f"<div style='font-size:0.72rem;font-weight:600;color:{color};text-transform:uppercase;"
                f"letter-spacing:0.08em;'>{subtitle}</div>"
                f"</div></div>"
                f"<div style='font-size:0.85rem;color:#4a5568;line-height:1.6;margin-bottom:0.8rem;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Open {title} →", key=f"nav_{title}",
                         use_container_width=True):
                st.session_state["nav_page"] = title
                st.session_state["sidebar_nav"] = title
                st.rerun()

    st.divider()

    # ── Analytical pipeline ───────────────────────────────────────────────────
    st.markdown("""
<div style='background:#f4f7fc;border-radius:12px;padding:1.4rem 2rem;margin-bottom:0.2rem;'>
  <div style='font-size:0.68rem;font-weight:700;color:#6b7a8d;letter-spacing:0.15em;
              text-transform:uppercase;margin-bottom:1rem;'>Analytical Pipeline</div>
  <div style='display:flex;align-items:center;gap:0;flex-wrap:wrap;'>
    <div style='background:#ffffff;border:1px solid #dde3ee;border-radius:8px;
                padding:0.7rem 1.1rem;text-align:center;min-width:110px;'>
      <div style='font-size:1.2rem;margin-bottom:0.2rem;'>📦</div>
      <div style='font-size:0.78rem;font-weight:700;color:#0f2847;'>UN Comtrade</div>
      <div style='font-size:0.68rem;color:#8a9ab0;'>HS4 · 2015–2024</div>
    </div>
    <div style='color:#b0baca;font-size:1.4rem;padding:0 0.5rem;'>→</div>
    <div style='background:#ffffff;border:1px solid #dde3ee;border-radius:8px;
                padding:0.7rem 1.1rem;text-align:center;min-width:110px;'>
      <div style='font-size:1.2rem;margin-bottom:0.2rem;'>📈</div>
      <div style='font-size:0.78rem;font-weight:700;color:#0f2847;'>Demand Forecast</div>
      <div style='font-size:0.68rem;color:#8a9ab0;'>Holt-Winters ETS</div>
    </div>
    <div style='color:#b0baca;font-size:1.4rem;padding:0 0.5rem;'>→</div>
    <div style='background:#ffffff;border:1px solid #dde3ee;border-radius:8px;
                padding:0.7rem 1.1rem;text-align:center;min-width:110px;'>
      <div style='font-size:1.2rem;margin-bottom:0.2rem;'>🤖</div>
      <div style='font-size:0.78rem;font-weight:700;color:#0f2847;'>ML Classifier</div>
      <div style='font-size:0.68rem;color:#8a9ab0;'>RF + XGBoost</div>
    </div>
    <div style='color:#b0baca;font-size:1.4rem;padding:0 0.5rem;'>→</div>
    <div style='background:#ffffff;border:1px solid #dde3ee;border-radius:8px;
                padding:0.7rem 1.1rem;text-align:center;min-width:110px;'>
      <div style='font-size:1.2rem;margin-bottom:0.2rem;'>⚖️</div>
      <div style='font-size:0.78rem;font-weight:700;color:#0f2847;'>Composite Score</div>
      <div style='font-size:0.68rem;color:#8a9ab0;'>6 weighted criteria</div>
    </div>
    <div style='color:#b0baca;font-size:1.4rem;padding:0 0.5rem;'>→</div>
    <div style='background:#0f2847;border:1px solid #0f2847;border-radius:8px;
                padding:0.7rem 1.1rem;text-align:center;min-width:110px;'>
      <div style='font-size:1.2rem;margin-bottom:0.2rem;'>🏆</div>
      <div style='font-size:0.78rem;font-weight:700;color:#ffffff;'>Ranked Markets</div>
      <div style='font-size:0.68rem;color:#7aabcf;'>Per GCC country</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Scoring methodology ───────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:700;color:#0f2847;margin-bottom:0.3rem;'>"
        "Composite Scoring Methodology</div>"
        "<div style='font-size:0.88rem;color:#6b7a8d;margin-bottom:1.2rem;'>"
        "Each opportunity is scored 0–1 by combining six independently min-max normalised "
        "sub-scores. Weights are fixed and expert-defined; no component can dominate due to "
        "unit differences.</div>",
        unsafe_allow_html=True,
    )

    score_rows = [
        ("#2e86de", "📈", "Demand Forecast",      "25",
         "4-year total import demand (2025–2028) projected by Holt-Winters ETS. Captures structural market size."),
        ("#1B9AAA", "🎯", "Penetration Gap",       "20",
         "Inverse of current GCC share in the destination (1 − pen %). High gap = large untapped headroom."),
        ("#8338EC", "🏛️", "Country Viability",     "20",
         "World Bank composite (2021–2023) across economic performance, governance, and infrastructure readiness."),
        ("#FF6B35", "🚢", "Landing Cost Index",    "15",
         "Blends inverted MFN tariff rate (50%) and World Bank LPI (50%). Lower cost, better logistics = higher score."),
        ("#D62828", "🤖", "ML Growth Signal",      "10",
         "Ensemble probability from Random Forest + XGBoost predicting above-median structural export growth."),
        ("#0a7a4e", "💰", "Price Quality",          "10",
         "Weighted blend of GCC unit export value level (70%) and its CAGR (30%). Rewards premium-priced exports."),
    ]

    for color, icon, comp, wt_str, desc in score_rows:
        wt = int(wt_str)
        bar_width = wt * 3.6   # scale 25% → 90px max on a nominal 360px bar
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #e8ecf2;border-radius:10px;"
            f"padding:1rem 1.4rem;margin-bottom:0.6rem;display:flex;align-items:center;gap:1.2rem;'>"
            # icon + label
            f"<div style='min-width:170px;display:flex;align-items:center;gap:0.55rem;'>"
            f"<span style='font-size:1.1rem;'>{icon}</span>"
            f"<span style='font-size:0.9rem;font-weight:700;color:#0f2847;'>{comp}</span>"
            f"</div>"
            # weight badge + bar
            f"<div style='min-width:160px;display:flex;align-items:center;gap:0.8rem;'>"
            f"<span style='background:{color}1a;color:{color};border-radius:20px;"
            f"padding:2px 10px;font-weight:800;font-size:0.82rem;white-space:nowrap;'>{wt}%</span>"
            f"<div style='background:#f0f4f8;border-radius:4px;height:6px;width:120px;'>"
            f"<div style='background:{color};border-radius:4px;height:6px;width:{bar_width}px;'></div>"
            f"</div></div>"
            # description
            f"<div style='font-size:0.84rem;color:#4a5568;line-height:1.55;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Data sources ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:700;color:#0f2847;margin-bottom:1rem;'>"
        "Data Sources</div>",
        unsafe_allow_html=True,
    )
    sources = [
        ("📦", "UN Comtrade",
         "HS4-level annual trade flows, 2015–2024, across 40 reporting countries."),
        ("🏦", "World Bank LPI 2023",
         "Logistics Performance Index scores used in the landing cost sub-component."),
        ("📋", "WITS / UNCTAD TRAINS",
         "MFN applied tariff rates at HS6, aggregated to HS2 by destination country."),
        ("🌐", "World Bank Open Data",
         "Country viability indicators: governance, economics, and infrastructure (2021–2023)."),
        ("📍", "CEPII GeoDist",
         "Capital-to-capital and population-weighted geographic distance matrix."),
        ("🏢", "OCO Global",
         "Analytical framework, sector prioritisation criteria, and strategic scope definition."),
    ]
    s1, s2, s3 = st.columns(3)
    src_cols_list = [s1, s2, s3]
    for i, (ico, src, desc) in enumerate(sources):
        with src_cols_list[i % 3]:
            st.markdown(
                f"<div style='background:#f8f9fb;border:1px solid #e4eaf2;border-radius:10px;"
                f"padding:1rem 1.2rem;margin-bottom:0.8rem;'>"
                f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;'>"
                f"<span style='font-size:1rem;'>{ico}</span>"
                f"<span style='font-weight:700;color:#0f2847;font-size:0.88rem;'>{src}</span>"
                f"</div>"
                f"<div style='color:#6b7a8d;font-size:0.81rem;line-height:1.55;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='border-top:1px solid #e8ecf2;margin-top:1.5rem;padding-top:1.2rem;"
        "display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;'>"
        "<div style='font-size:0.78rem;color:#8a9ab0;'>"
        "Georges Elkassouf &amp; Joseph Hobeika &nbsp;·&nbsp; AUB MSBA &nbsp;·&nbsp; Spring 2026"
        "</div>"
        "<div style='font-size:0.78rem;color:#8a9ab0;'>"
        "In partnership with <strong style='color:#0f2847;'>OCO Global</strong>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# OPPORTUNITY FINDER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Opportunity Finder":
    st.title("Opportunity Finder")
    st.markdown(
        "Pick a **GCC exporter** and a **commodity** — the dashboard surfaces "
        "the highest-potential destination markets ranked by a composite opportunity score."
    )
    st.markdown(
        "<p style='font-size:1rem;font-weight:700;color:#1a3a5c;margin:4px 0 14px;line-height:1.8;'>"
        "Score &nbsp;=&nbsp; Demand Forecast (25%) &nbsp;+&nbsp; Penetration Gap (20%) "
        "&nbsp;+&nbsp; Country Viability (20%) &nbsp;+&nbsp; Landing Cost (15%) "
        "&nbsp;+&nbsp; ML Growth Signal (10%) &nbsp;+&nbsp; Price Quality (10%)"
        "</p>",
        unsafe_allow_html=True,
    )

    opp = load_opp()
    if opp is None:
        st.error("Missing: **opportunity_rankings_full.csv**. Place it into `data/`.")
        st.stop()

    col_gcc, col_cmd = st.columns([1, 2])
    with col_gcc:
        gcc_sel = st.selectbox("GCC Exporter", sorted(opp["gcc_country"].unique()))

    df_gcc = opp[opp["gcc_country"] == gcc_sel].copy()

    # Only show commodities where this GCC country has actual exports.
    exported_cmds = (
        df_gcc[df_gcc["penetration_pct"] > 0]["cmdCode"].unique()
    )
    df_gcc_exported = df_gcc[df_gcc["cmdCode"].isin(exported_cmds)]

    cmd_scores = (
        df_gcc_exported.groupby(["cmdCode", "commodity"])["opportunity_score"]
        .max().reset_index().sort_values("opportunity_score", ascending=False)
    )
    cmd_labels = cmd_scores.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
    cmd_code_map = dict(zip(cmd_labels, cmd_scores["cmdCode"]))

    if not cmd_labels:
        st.warning("No commodities found for this GCC country.")
        st.stop()

    with col_cmd:
        cmd_sel = st.selectbox(
            "🔍 Search or select commodity — sorted by opportunity score ↓",
            cmd_labels,
            index=0,
            key=f"cmd_sel_{gcc_sel}",
        )

    sel_code = cmd_code_map[cmd_sel]
    df = df_gcc_exported[df_gcc_exported["cmdCode"] == sel_code].sort_values("opportunity_score", ascending=False).copy()
    if df.empty:
        st.info("No scored opportunities for this combination.")
        st.stop()

    commodity_name = df["commodity"].iloc[0]

    # KPIs
    st.divider()
    n_markets = df["dest_country"].nunique()
    k1, k2, k3 = st.columns(3)
    k1.metric(
        f"Destination Markets Ranked — {gcc_sel}",
        f"{n_markets}",
        "scored markets",
    )

    # GCC Exports for this country + commodity
    # Strategy: try gcc_export_penetration.csv filtered by gcc_country first;
    # if that column doesn't exist or yields 0, estimate from penetration_pct * demand
    pen_data = load("gcc_export_penetration.csv")
    gcc_exp_val = 0
    exp_label = "—"
    exp_year = ""
    if pen_data is not None and "gcc_exports" in pen_data.columns:
        # pen_scored has no year column — it is a 2022-2023 average.
        # Filter by gcc_country and cmdCode directly without year.
        exp_year = "2022–2023 avg"
        if "gcc_country" in pen_data.columns:
            cmd_pen = pen_data[
                (pen_data["gcc_country"] == gcc_sel) &
                (pen_data["cmdCode"] == sel_code)
            ]
        else:
            # Fallback: old format without gcc_country column
            cmd_pen = pen_data[pen_data["cmdCode"] == sel_code]
        gcc_exp_val = float(cmd_pen["gcc_exports"].sum()) if not cmd_pen.empty else 0

    # If still 0, estimate from penetration_pct × demand_4y_total / 4 (annual proxy)
    if gcc_exp_val == 0 and "penetration_pct" in df.columns and "demand_4y_total" in df.columns:
        avg_pen = float(df["penetration_pct"].mean()) / 100.0
        total_demand_annual = float(df["demand_4y_total"].sum()) / 4.0
        gcc_exp_val = avg_pen * total_demand_annual
        exp_label = f"est. from pen. gap"
        exp_year = "2024 est."
    else:
        exp_label = f"{gcc_sel} exports, {exp_year}" if exp_year else "GCC combined"

    k2.metric(
        f"{gcc_sel} Exports — {commodity_name[:28]}",
        fmt_usd(gcc_exp_val),
        exp_label,
    )

    if "demand_4y_total" in df.columns:
        k3.metric(
            f"4-Year Forecast Demand — {commodity_name[:28]}",
            fmt_usd(df["demand_4y_total"].sum()),
            f"across {n_markets} markets (2025–2028)",
        )

    # Top 15 bar chart
    st.divider()
    st.subheader(f"Top 15 Target Markets for {gcc_sel} — {commodity_name[:50]}")
    st.caption("Ranked by composite opportunity score (demand forecast · penetration gap · country viability · landing cost · ML signal · price quality)")
    df_top = df.head(15)
    fig = hbar(
        df_top["dest_country"], df_top["opportunity_score"],
        colorscale=BLUE_SCALE,
        text_fmt=[f"{v:.3f}" for v in df_top["opportunity_score"]],
        x_title="Opportunity Score",
    )
    # Zoom x-axis: start just below the lowest score so differences are legible
    _scores = df_top["opportunity_score"]
    _spread = float(_scores.max() - _scores.min())
    _pad = max(_spread * 0.5, 0.01)
    _x_min = max(0.0, float(_scores.min()) - _pad)
    _x_max = float(_scores.max()) + _pad * 0.4
    fig.update_xaxes(range=[_x_min, _x_max])
    st.plotly_chart(fig, use_container_width=True)

    # Choropleth map
    st.subheader(f"Market Attractiveness Map — {gcc_sel} · {commodity_name[:45]}")
    st.caption("Color intensity reflects the composite opportunity score across all scored destination markets. Grey = outside the 40-market universe.")
    fig_map = go.Figure(go.Choropleth(
        locations=df["dest_country"],
        locationmode="country names",
        z=df["opportunity_score"],
        colorscale=BLUE_SCALE,
        zmin=float(df["opportunity_score"].min()),
        zmax=float(df["opportunity_score"].max()),
        hovertemplate="<b>%{location}</b><br>Opportunity Score: %{z:.3f}<extra></extra>",
        marker_line_color="white",
        marker_line_width=0.5,
        colorbar=dict(
            title=dict(text="Score", font=dict(size=12)),
            thickness=14, len=0.7,
            tickformat=".2f",
        ),
    ))
    fig_map.update_layout(
        **CHART_LAYOUT,
        geo=dict(
            showframe=False,
            showcoastlines=True, coastlinecolor="rgba(0,0,0,0.08)",
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
            showland=True, landcolor="#f0f4f8",
            showocean=True, oceancolor="#dce8f5",
            showcountries=True, countrycolor="rgba(0,0,0,0.08)",
            lataxis_range=[-55, 80],
        ),
        margin=dict(t=10, b=10, l=0, r=0),
        height=440,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Detail table
    st.divider()
    st.subheader(f"Score Components — Top 15 Target Markets for {gcc_sel} ({commodity_name[:40]})")
    # Column legend
    st.markdown("""
<div style="background:#f4f7fc;border-radius:10px;padding:14px 18px;margin-bottom:14px;font-size:0.82rem;color:#1a3a5c;line-height:1.9;">
<b>Column guide:</b> &nbsp;
<b>Score</b> — composite opportunity score (0–1, higher = more attractive). &nbsp;·&nbsp;
<b>Grade</b> — country viability tier (A+/A/B/C/D) based on World Bank governance & economic indicators. &nbsp;·&nbsp;
<b>4Y Demand</b> — total forecasted import demand for this commodity in that market over 2025–2028. &nbsp;·&nbsp;
<b>ML Growth P</b> — Random Forest + XGBoost probability (0–1) that this market will show above-median structural growth. &nbsp;·&nbsp;
<b>LPI</b> — World Bank Logistics Performance Index score (1–5) for the destination; higher = easier to ship to. &nbsp;·&nbsp;
<b>Tariff %</b> — MFN applied tariff rate (%) faced by GCC exporters; lower = cheaper market entry. &nbsp;·&nbsp;
<b>Transport</b> — recommended shipping mode based on distance, LPI, and commodity weight.
</div>
    """, unsafe_allow_html=True)

    col_map = {
        "dest_country": "Target Market", "opportunity_score": "Score", "grade": "Grade",
        "demand_4y_total": "4Y Demand",
        "ml_growth_prob": "ML Growth P",
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
    for c, r in [("Score", 3), ("GCC Pen %", 1), ("UV ($/kg)", 2),
                  ("Price CAGR %", 1), ("ML Growth P", 2), ("LPI", 2), ("Tariff %", 1)]:
        if c in table.columns:
            table[c] = table[c].round(r)
    if "Distance (km)" in table.columns:
        table["Distance (km)"] = table["Distance (km)"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
    if "Transport" in table.columns:
        table["Transport"] = table["Transport"].apply(
            lambda x: f"{TRANSPORT_ICONS.get(str(x), '')} {x}" if pd.notna(x) else "—")
    if "Rationale" in table.columns:
        table["Rationale"] = (
            table["Rationale"]
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str[:90]
        )
    st.dataframe(table, use_container_width=True, hide_index=True, height=min(620, len(df_top) * 42 + 50))

    # Commodity-level metrics (same across all destinations — shown as cards, not table columns)
    st.markdown("#### Commodity-Level Characteristics *(constant across all destination markets)*")
    st.caption("These values describe the commodity itself rather than a specific destination, so they are the same for every row in the table above.")
    ci1, ci2, ci3 = st.columns(3)
    if "penetration_pct" in df_top.columns:
        avg_pen = float(df_top["penetration_pct"].mean())
        ci1.metric(
            "Avg GCC Penetration (across top 15)",
            f"{avg_pen:.1f}%",
            "share of destination import demand captured by GCC",
        )
    if "uv_mean" in df_top.columns:
        uv_val = float(df_top["uv_mean"].iloc[0]) if not df_top["uv_mean"].isna().all() else None
        ci2.metric(
            f"Unit Value — {commodity_name[:30]}",
            f"${uv_val:,.2f} /kg" if uv_val else "—",
            "avg GCC export price per kg (price quality signal)",
        )
    if "uv_cagr" in df_top.columns:
        cagr_val = float(df_top["uv_cagr"].iloc[0]) if not df_top["uv_cagr"].isna().all() else None
        ci3.metric(
            "Unit Value CAGR",
            f"{cagr_val:.1f}%" if cagr_val is not None else "—",
            "compound annual growth rate of GCC export price",
        )

    # Rationale expander
    if "opportunity_rationale" in df_top.columns:
        with st.expander("📋 Full scoring rationale for each market"):
            for _, row in df_top.iterrows():
                transport = row.get("recommended_transport", "")
                icon = TRANSPORT_ICONS.get(str(transport), "")
                st.markdown(
                    f"**{row['dest_country']}** · Score **{row['opportunity_score']:.3f}** · "
                    f"Grade {row.get('grade', '—')} · {icon} {transport}")
                raw_rat = row.get("opportunity_rationale", "—") or "—"
                clean_rat = re.sub(r'\s*\([^)]*\)', '', str(raw_rat)).strip()
                st.caption(clean_rat)
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

    latest_yr = int(yearly["year"].max()) if yearly is not None else "—"
    total_demand = yearly.loc[yearly["year"] == latest_yr, "total_demand"].iloc[0] if yearly is not None else 0
    total_gcc = yearly.loc[yearly["year"] == latest_yr, "total_gcc_exports"].iloc[0] if yearly is not None else 0

    c1, c2 = st.columns(2)
    c1.metric(
        f"Total Import Demand — 40 Destinations",
        fmt_usd(total_demand),
        f"across all HS2 sectors, {latest_yr}",
    )
    c2.metric(
        "Combined GCC Non-Fuel Exports",
        fmt_usd(total_gcc),
        f"all 6 GCC members combined, {latest_yr}",
    )

    st.divider()
    st.subheader("Highest-Scored Export Opportunity per GCC Member State")
    st.caption("For each GCC country, the single commodity-destination combination with the highest composite opportunity score.")
    top1 = (
        opp.sort_values("opportunity_score", ascending=False)
        .groupby("gcc_country").first().reset_index()
    )
    display = top1[["gcc_country", "dest_country", "commodity"]].copy()
    display.columns = ["GCC Exporter", "Best Target Market", "Top Export Commodity"]
    display["Top Export Commodity"] = display["Top Export Commodity"].str[:60]
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Trend charts
    st.divider()
    col_l, col_r = st.columns(2)
    if yearly is not None:
        with col_l:
            st.subheader("Annual Import Demand Across 40 Destination Markets (2015–2024)")
            st.caption("Combined import value of all non-GCC destination countries across all non-fuel HS2 sectors.")
            yearly["d_T"] = yearly["total_demand"] / 1e12
            st.plotly_chart(
                area_chart(yearly["year"], yearly["d_T"], "#0F4C75", "USD (Trillions)"),
                use_container_width=True,
            )
        with col_r:
            st.subheader("Annual Combined GCC Non-Fuel Export Value (2015–2024)")
            st.caption("Total non-fuel exports from all 6 GCC member states to the 40 destination markets.")
            _max_gcc = float(yearly["total_gcc_exports"].max())
            if _max_gcc >= 1e12:
                yearly["g_scaled"] = yearly["total_gcc_exports"] / 1e12
                _gcc_unit = "USD (Trillions)"
            elif _max_gcc >= 1e9:
                yearly["g_scaled"] = yearly["total_gcc_exports"] / 1e9
                _gcc_unit = "USD (Billions)"
            else:
                yearly["g_scaled"] = yearly["total_gcc_exports"] / 1e6
                _gcc_unit = "USD (Millions)"
            st.plotly_chart(
                area_chart(yearly["year"], yearly["g_scaled"], "#FF6B35", _gcc_unit),
                use_container_width=True,
            )






# ═══════════════════════════════════════════════════════════════════════════
# MARKET DEMAND
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Market Demand":
    st.title("Destination Market Demand")
    st.markdown("What do the 40 destination countries import — and which commodities matter most?")

    pen = require("gcc_export_penetration.csv")

    # pen_scored (new notebook output) is a 2022-2023 average — no year column.
    # Year-level historical trend charts require per-year rows; skip gracefully.
    if "year" not in pen.columns:
        st.info(
            "ℹ️ Year-level historical demand trend is not available in the current "
            "data version. `gcc_export_penetration.csv` is now a 2022–2023 average "
            "(`pen_scored`) with no `year` column. The top-commodity ranking and "
            "per-commodity trend selector below are still fully functional."
        )
    else:
        # Yearly trend
        yearly = pen.groupby("year")["world_demand"].sum().reset_index()
        yearly["d_B"] = yearly["world_demand"] / 1e9

        st.subheader("Annual Combined Import Demand Across 40 Destination Markets (2015–2024)")
        st.caption("Bar = total import value (left axis, USD Trillions). Line = year-on-year growth rate (right axis, %). The 2019→2020 dip reflects COVID-19 trade contraction.")
        yearly["d_T"] = yearly["world_demand"] / 1e12
        yearly = yearly.sort_values("year").reset_index(drop=True)
        yearly["yoy_growth"] = yearly["d_T"].pct_change() * 100

        fig_demand = go.Figure()
        # Bars coloured by value
        fig_demand.add_trace(go.Bar(
            x=yearly["year"], y=yearly["d_T"],
            name="Import Demand (USD T)",
            marker=dict(color=yearly["d_T"], colorscale=BLUE_SCALE, cornerradius=4, line=dict(width=0)),
            text=[f"${v:.2f}T" for v in yearly["d_T"]], textposition="outside",
            textfont=dict(size=12, color="#0f2847", family="system-ui, sans-serif"),
            hovertemplate="%{x}<br>Demand: $%{y:.2f}T<extra></extra>",
            yaxis="y1",
        ))
        # YoY growth line on secondary axis
        fig_demand.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["yoy_growth"],
            name="YoY Growth %",
            mode="lines+markers+text",
            line=dict(color="#FF6B35", width=2.5, dash="dot"),
            marker=dict(size=7, color="#FF6B35", line=dict(color="white", width=1.5)),
            text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in yearly["yoy_growth"]],
            textposition="top center", textfont=dict(size=9, color="#c4420a"),
            hovertemplate="%{x}<br>YoY Growth: %{y:+.1f}%<extra></extra>",
            yaxis="y2",
        ))
        fig_demand.update_layout(
            **CHART_LAYOUT,
            margin=dict(t=30, b=40, l=10, r=80), height=420,
            yaxis=dict(
                title="USD (Trillions)", gridcolor="rgba(0,0,0,0.05)", side="left",
                tickformat=".1f", ticksuffix="T",
                tickfont=dict(size=12, color="#1a3a5c"),
                title_font=dict(size=12),
            ),
            yaxis2=dict(
                title="YoY Growth (%)", overlaying="y", side="right",
                showgrid=False, zeroline=True, zerolinecolor="rgba(0,0,0,0.2)",
                zerolinewidth=1.5,
                ticksuffix="%", tickformat="+.1f",
                tickfont=dict(size=12, color="#c4420a"),
                title_font=dict(size=12, color="#c4420a"),
            ),
            xaxis=dict(
                title="", showgrid=False,
                tickfont=dict(size=12), dtick=1,
            ),
            legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12)),
            bargap=0.28,
        )
        st.plotly_chart(fig_demand, use_container_width=True)

    # Top 20 commodities + connected trend
    st.divider()
    st.subheader("Top 20 Most-Imported Commodity Sectors — Click to Explore Trend")
    st.caption("Ranked by total import value (2015–2024). Select a commodity from the list below the chart to see its historical demand trend.")

    top_cmd = (
        pen.groupby(["cmdCode", "commodity"])["world_demand"]
        .sum().reset_index().sort_values("world_demand", ascending=False).head(20)
    )
    top_cmd["label"] = top_cmd["cmdCode"].astype(str) + " — " + top_cmd["commodity"].str[:45]
    top_cmd["d_B"] = top_cmd["world_demand"] / 1e9

    # Commodity selector — pre-filled with the top 20 in the same order as the chart
    sel_trend_label = st.selectbox(
        "🔍 Select commodity to view its demand trend (defaults to #1)",
        top_cmd["label"].tolist(),
        index=0,
        key="md_trend_sel",
    )
    sel_trend_code = top_cmd.loc[top_cmd["label"] == sel_trend_label, "cmdCode"].iloc[0]
    sel_trend_name = top_cmd.loc[top_cmd["label"] == sel_trend_label, "commodity"].iloc[0]

    # Colour the selected bar in orange, others in teal
    bar_colors = [
        "#FF6B35" if lbl == sel_trend_label else "#1B9AAA"
        for lbl in top_cmd["label"]
    ]
    fig_top20 = go.Figure()
    fig_top20.add_trace(go.Bar(
        y=top_cmd["label"], x=top_cmd["d_B"], orientation="h",
        marker=dict(color=bar_colors, cornerradius=4, line=dict(width=0)),
        text=[f"${v:.1f}B" for v in top_cmd["d_B"]], textposition="outside",
        textfont=dict(size=11, color="#1a3a5c"),
        hovertemplate="<b>%{y}</b><br>$%{x:.1f}B<extra></extra>",
    ))
    h20 = max(380, len(top_cmd) * 34)
    fig_top20.update_layout(
        **CHART_LAYOUT,
        margin=dict(t=10, b=30, l=10, r=70), height=h20,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12), showgrid=False),
        xaxis=dict(title="Total Import Value 2015–2024 (USD Billions)",
                   gridcolor="rgba(0,0,0,0.05)"),
    )
    st.plotly_chart(fig_top20, use_container_width=True)

    # Trend for selected commodity
    st.markdown(f"#### Import Demand Trend — {sel_trend_name}")
    if "year" in pen.columns:
        trend = pen[pen["cmdCode"] == sel_trend_code].groupby("year")["world_demand"].sum().reset_index()
        trend["d_B"] = trend["world_demand"] / 1e9
        trend = trend.sort_values("year")
        trend["yoy"] = trend["d_B"].pct_change() * 100

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend["year"], y=trend["d_B"], mode="lines+markers",
            name="Import Demand",
            line=dict(color="#0F4C75", width=2.5),
            marker=dict(size=7, color="#0F4C75", line=dict(color="white", width=1.5)),
            hovertemplate="%{x}<br>Demand: $%{y:.1f}B<extra></extra>",
            yaxis="y1",
        ))
        fig_trend.add_trace(go.Bar(
            x=trend["year"], y=trend["yoy"],
            name="YoY Growth %",
            marker=dict(
                color=["#D62828" if (v < 0) else "#2e86de" for v in trend["yoy"].fillna(0)],
                opacity=0.35, cornerradius=3, line=dict(width=0),
            ),
            hovertemplate="%{x}<br>Growth: %{y:+.1f}%<extra></extra>",
            yaxis="y2",
        ))
        fig_trend.update_layout(
            **CHART_LAYOUT,
            margin=dict(t=20, b=30, l=10, r=60), height=320,
            yaxis=dict(title="USD (Billions)", gridcolor="rgba(0,0,0,0.05)"),
            yaxis2=dict(title="YoY Growth (%)", overlaying="y", side="right",
                        showgrid=False, zeroline=True, zerolinecolor="rgba(0,0,0,0.2)",
                        ticksuffix="%"),
            xaxis=dict(title="", showgrid=False),
            legend=dict(orientation="h", y=1.08, x=0),
            barmode="overlay",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Year-level per-commodity trend not available in current data version.")


# ═══════════════════════════════════════════════════════════════════════════
# GCC PENETRATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "GCC Penetration":
    st.title("GCC Export Penetration")
    st.markdown("**Penetration %** = Combined GCC exports / destination import demand. Low penetration + high demand = opportunity. Figures aggregate all 6 GCC member states.")

    pen = require("gcc_export_penetration.csv")

    # pen_scored is a 2022-2023 average per gcc_country × dest_country × cmdCode.
    # To show combined GCC penetration (all 6 countries, all destinations) we:
    #   1. Sum gcc_exports across all gcc_countries per (dest_country, cmdCode)
    #   2. Take world_demand once per (dest_country, cmdCode) — not duplicated per GCC country
    #   3. Sum across destinations to get commodity-level totals
    #   4. Re-derive penetration_pct from the aggregated figures
    if "gcc_country" in pen.columns and "dest_country" in pen.columns:
        by_dest = (
            pen.groupby(["dest_country", "cmdCode", "commodity"])
            .agg(
                gcc_exports=("gcc_exports", "sum"),
                world_demand=("world_demand", "first"),  # same value per dest — not per GCC country
            )
            .reset_index()
        )
        snap = (
            by_dest.groupby(["cmdCode", "commodity"])
            .agg(gcc_exports=("gcc_exports", "sum"), world_demand=("world_demand", "sum"))
            .reset_index()
        )
        snap["penetration_pct"] = (
            snap["gcc_exports"] / snap["world_demand"] * 100
        ).clip(0, 100).round(2)
        data_label = "2022–2023 avg"
    else:
        # Fallback: old format with year column
        latest_yr = int(pen["year"].max())
        snap = pen[pen["year"] == latest_yr].copy()
        data_label = str(latest_yr)

    _, col_c, _ = st.columns([0.5, 9, 0.5])

    with col_c:
        st.subheader(f"Sectors Where GCC Has the Strongest Market Presence ({data_label})")
        st.caption(f"Top 15 commodity sectors by GCC export share of destination import demand. A high % means GCC suppliers already dominate that market.")
        high = snap.sort_values("penetration_pct", ascending=False).head(15).copy()
        high["label"] = high["cmdCode"].astype(str) + " — " + high["commodity"].str[:50]
        fig2 = hbar(
            high["label"], high["penetration_pct"], colorscale=PURPLE_SCALE,
            text_fmt=[f"{v:.1f}%" for v in high["penetration_pct"]],
            x_title="GCC Penetration %", height=600,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    _, col_c2, _ = st.columns([0.5, 9, 0.5])

    with col_c2:
        st.subheader(f"High-Demand Sectors With Low GCC Penetration — Untapped Opportunities ({data_label})")
        st.caption("Commodity sectors in the top 50% of global import demand where GCC supplies less than 5% of total imports. These represent the largest addressable whitespace for GCC exporters.")
        gaps = snap[(snap["penetration_pct"] < 5) &
                    (snap["world_demand"] > snap["world_demand"].quantile(0.5))]
        gaps = gaps.sort_values("world_demand", ascending=False).head(15).copy()
        if not gaps.empty:
            gaps["label"] = gaps["cmdCode"].astype(str) + " — " + gaps["commodity"].str[:50]
            gaps["d_B"] = gaps["world_demand"] / 1e9
            fig3 = hbar(
                gaps["label"], gaps["d_B"], colorscale=RED_SCALE,
                text_fmt=[f"${v:.1f}B" for v in gaps["d_B"]],
                x_title="Import Demand (USD B)", height=600,
            )
            st.plotly_chart(fig3, use_container_width=True)



# ═══════════════════════════════════════════════════════════════════════════
# DEMAND FORECASTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Demand Forecasts":
    st.title("4-Year Import Demand Forecasts (2025–2028)")
    st.markdown("Holt-Winters exponential smoothing forecasts of global import demand per commodity sector, trained on 2015–2024 historical data. The shaded band shows the 95% confidence interval.")

    files = require("demand_forecast_global.csv", "gcc_export_penetration.csv")
    fc = files["demand_forecast_global.csv"]
    pen = files["gcc_export_penetration.csv"]
    hist = pen.groupby(["cmdCode", "commodity", "year"])["world_demand"].sum().reset_index()

    # GCC exporter filter
    opp_fc = load_opp()
    col_gcc_fc, col_spacer = st.columns([1, 2])
    with col_gcc_fc:
        gcc_options = ["All GCC"] + (sorted(opp_fc["gcc_country"].unique().tolist()) if opp_fc is not None else [])
        gcc_fc_sel = st.selectbox("Filter by GCC Exporter", gcc_options, key="fc_gcc")

    # If a specific GCC country is selected, restrict commodity list to its top-ranked commodities
    if gcc_fc_sel != "All GCC" and opp_fc is not None:
        top_cmds = (
            opp_fc[opp_fc["gcc_country"] == gcc_fc_sel]
            .groupby(["cmdCode", "commodity"])["opportunity_score"]
            .max().reset_index().sort_values("opportunity_score", ascending=False)
        )
        fc_filtered = fc[fc["cmdCode"].isin(top_cmds["cmdCode"])]
        fc_totals = (
            fc_filtered.groupby(["cmdCode", "commodity"])["demand_ensemble"]
            .sum().reset_index()
            .merge(top_cmds[["cmdCode", "opportunity_score"]], on="cmdCode")
            .sort_values("opportunity_score", ascending=False)
        )
    else:
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
        title=dict(text=f"Historical & Projected Global Import Demand — {cname} (2015–2028)", font=dict(size=14, color="#1a3a5c")),
        yaxis=dict(title="Import Demand (USD)", gridcolor="rgba(0,0,0,0.05)"),
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0.05)"),
        margin=dict(t=50, b=30), height=430,
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    if not f.empty:
        st.subheader(f"Projected Annual Import Demand — {cname} (2025–2028)")
        tbl = f[["year", "demand_ensemble", "ci_lower", "ci_upper"]].copy()
        tbl.columns = ["Year", "Forecast", "Lower Bound", "Upper Bound"]
        for c in ["Forecast", "Lower Bound", "Upper Bound"]:
            tbl[c] = tbl[c].apply(fmt_usd)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Top forecasted markets
    st.divider()
    st.subheader(f"Top 15 Commodity Sectors by Total Projected Import Demand (2025–2028){' — ' + gcc_fc_sel if gcc_fc_sel != 'All GCC' else ''}")
    st.caption("Ranked by the sum of 4-year Holt-Winters demand forecasts across all destination markets." + (f" Filtered to commodities relevant for {gcc_fc_sel} based on opportunity scores." if gcc_fc_sel != "All GCC" else ""))
    top_fc = fc_totals.head(15).copy()
    top_fc["label"] = top_fc["cmdCode"].astype(str) + " — " + top_fc["commodity"].str[:45]
    top_fc["d_B"] = top_fc["demand_ensemble"] / 1e9
    fig2 = hbar(
        top_fc["label"], top_fc["d_B"], colorscale=RED_SCALE,
        text_fmt=[f"${v:.1f}B" for v in top_fc["d_B"]],
        height=430, x_title="4-Year Forecast (USD B)",
    )
    st.plotly_chart(fig2, use_container_width=True)
