"""
GCC Export Opportunity Dashboard
Predicting Export Market Growth & Price Potential for Strategic Trade Diversification
OCO Global x AUB MSBA Capstone

Works with the 10 CSV files already exported by Section 45 of the notebook:
  1. demand_forecast_global.csv
  2. demand_forecast_by_dest_country.csv
  3. gcc_export_unit_value_forecast.csv
  4. destination_country_viability.csv
  5. opportunity_rankings_full.csv
  6. top20_per_gcc_country.csv
  7. gcc_export_penetration.csv
  8. backtest_results.csv
  9. ml_growth_probabilities.csv
  10. reexport_flags.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GCC Export Opportunities",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA = Path("data")

GRADE_COLORS = {
    "A+": "#0B6E4F", "A": "#21A179", "B": "#F4A236",
    "C": "#E76F51", "D": "#D62828",
}
GCC_COLORS = {
    "Bahrain": "#1B9AAA", "Kuwait": "#06D6A0", "Oman": "#EF476F",
    "Qatar": "#8338EC", "Saudi Arabia": "#0F4C75", "United Arab Emirates": "#FF6B35",
}
TRANSPORT_ICONS = {"Sea": "🚢", "Air": "✈️", "Land": "🚛"}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load(name):
    path = DATA / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def require(*names):
    frames = {}
    missing = []
    for n in names:
        df = load(n)
        if df is None:
            missing.append(n)
        else:
            frames[n] = df
    if missing:
        st.error(
            f"Missing data file(s): **{', '.join(missing)}**. "
            "Place the CSVs exported by Section 45 of the notebook into the `data/` folder."
        )
        st.stop()
    return frames if len(names) > 1 else frames[names[0]]


def fmt_usd(val, decimals=1):
    if pd.isna(val) or val == 0:
        return "—"
    if abs(val) >= 1e12:
        return f"${val/1e12:.{decimals}f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.{decimals}f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.{decimals}f}M"
    if abs(val) >= 1e3:
        return f"${val/1e3:.{decimals}f}K"
    return f"${val:,.0f}"


# ---------------------------------------------------------------------------
# DERIVED DATA — computed from existing files, no extra exports needed
# ---------------------------------------------------------------------------
@st.cache_data
def derive_yearly_from_penetration():
    pen = load("gcc_export_penetration.csv")
    if pen is None:
        return None
    return (
        pen.groupby("year")
        .agg(total_demand=("world_demand", "sum"), total_gcc_exports=("gcc_exports", "sum"))
        .reset_index()
    )


@st.cache_data
def derive_backtest_metrics():
    bt = load("backtest_results.csv")
    if bt is None:
        return {}
    bt = bt.dropna(subset=["actual", "predicted"]).query("actual > 0")
    if bt.empty:
        return {}
    mae = np.mean(np.abs(bt["actual"] - bt["predicted"]))
    rmse = np.sqrt(np.mean((bt["actual"] - bt["predicted"]) ** 2))
    ss_res = np.sum((bt["actual"] - bt["predicted"]) ** 2)
    ss_tot = np.sum((bt["actual"] - bt["actual"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mape_raw = np.mean(np.abs((bt["actual"] - bt["predicted"]) / bt["actual"])) * 100
    bt_sig = bt[bt["actual"] > 1_000_000]
    mape_sig = (
        np.mean(np.abs((bt_sig["actual"] - bt_sig["predicted"]) / bt_sig["actual"])) * 100
        if len(bt_sig) > 0
        else mape_raw
    )
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "mape_all": mape_raw, "mape_sig": mape_sig,
        "n_total": len(bt), "n_sig": len(bt_sig),
    }


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 0.5rem 0 1rem 0;'>"
        "<div style='font-size:1.6rem; font-weight:700; color:#0F4C75;'>"
        "🌍 GCC Export<br>Opportunities</div>"
        "<div style='font-size:0.75rem; color:#888; margin-top:0.25rem;'>"
        "OCO Global · AUB MSBA Capstone</div></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Executive Summary",
            "Market Demand",
            "GCC Penetration",
            "Demand Forecasts",
            "Price Outlook",
            "Opportunity Finder",
            "Country Profiles",
            "Model Confidence",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: UN Comtrade · ITC Trade Map")
    st.caption("HS27 / HS71 / HS93 / HS99 excluded")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.title("Executive Summary")
    st.markdown(
        "A data-driven view of where GCC countries should focus "
        "non-fuel export efforts over the next 3-5 years."
    )

    opp = require("opportunity_rankings_full.csv")
    yearly = derive_yearly_from_penetration()
    bt_metrics = derive_backtest_metrics()

    latest_yr = int(yearly["year"].max()) if yearly is not None else "—"
    total_demand = yearly.loc[yearly["year"] == latest_yr, "total_demand"].iloc[0] if yearly is not None else 0
    total_gcc_exp = yearly.loc[yearly["year"] == latest_yr, "total_gcc_exports"].iloc[0] if yearly is not None else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Addressable Import Demand", fmt_usd(total_demand), f"{latest_yr}")
    c2.metric("GCC Non-Fuel Exports", fmt_usd(total_gcc_exp), f"{latest_yr}")
    c3.metric("Scored Opportunities", f"{len(opp):,}", f"{opp['dest_country'].nunique()} markets")
    c4.metric("Best Opportunity Score", f"{opp['opportunity_score'].max():.3f}", "out of 1.000")

    st.divider()

    st.subheader("Top Opportunity per GCC Country")
    top1 = (
        opp.sort_values("opportunity_score", ascending=False)
        .groupby("gcc_country").first().reset_index()
    )
    cols_show = [c for c in ["gcc_country", "dest_country", "commodity", "opportunity_score", "grade"] if c in top1.columns]
    top1_disp = top1[cols_show].copy()
    top1_disp.columns = ["GCC Exporter", "Best Target Market", "Top Commodity", "Score", "Market Grade"][:len(cols_show)]
    if "Top Commodity" in top1_disp.columns:
        top1_disp["Top Commodity"] = top1_disp["Top Commodity"].str[:55]
    st.dataframe(
        top1_disp.style.format({"Score": "{:.3f}"}).background_gradient(subset=["Score"], cmap="YlGn"),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    col_l, col_r = st.columns(2)
    if yearly is not None:
        with col_l:
            st.subheader("Import Demand Trend — 40 Destinations")
            yearly["demand_B"] = yearly["total_demand"] / 1e9
            fig = px.area(yearly, x="year", y="demand_B", labels={"demand_B": "USD (Billions)", "year": ""}, color_discrete_sequence=["#0F4C75"])
            fig.update_layout(margin=dict(t=10, b=30), height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.subheader("GCC Non-Fuel Exports (Aggregate)")
            yearly["gcc_B"] = yearly["total_gcc_exports"] / 1e9
            fig2 = px.area(yearly, x="year", y="gcc_B", labels={"gcc_B": "USD (Billions)", "year": ""}, color_discrete_sequence=["#FF6B35"])
            fig2.update_layout(margin=dict(t=10, b=30), height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    if bt_metrics:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Demand Forecast MAPE", f"{bt_metrics['mape_sig']:.1f}%", "on $1M+ series")
        c2.metric("Forecast R²", f"{bt_metrics['r2']:.3f}", f"N = {bt_metrics['n_total']:,}")
        c3.metric("Back-test MAE", fmt_usd(bt_metrics["mae"]))


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — MARKET DEMAND
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Market Demand":
    st.title("Destination Market Demand")
    st.markdown("What do the 40 destination countries import — and which commodities drive the most value?")

    pen = require("gcc_export_penetration.csv")

    yearly_demand = pen.groupby("year")["world_demand"].sum().reset_index()
    yearly_demand["demand_B"] = yearly_demand["world_demand"] / 1e9

    st.subheader("Total Addressable Import Demand Over Time")
    fig = px.bar(yearly_demand, x="year", y="demand_B", labels={"demand_B": "Import Value (USD Billions)", "year": ""}, color_discrete_sequence=["#0F4C75"])
    fig.update_layout(margin=dict(t=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Top 20 Commodities by Total Import Demand")
    top_cmd = (
        pen.groupby(["cmdCode", "commodity"])["world_demand"]
        .sum().reset_index().sort_values("world_demand", ascending=False).head(20)
    )
    top_cmd["label"] = top_cmd["cmdCode"].astype(str) + " — " + top_cmd["commodity"].str[:45]
    top_cmd["demand_B"] = top_cmd["world_demand"] / 1e9
    fig2 = px.bar(top_cmd, y="label", x="demand_B", orientation="h", labels={"demand_B": "USD (Billions)", "label": ""}, color_discrete_sequence=["#1B9AAA"])
    fig2.update_layout(margin=dict(t=10, l=10), height=540, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Commodity Demand Trend")
    cmd_options = pen.groupby(["cmdCode", "commodity"])["world_demand"].sum().reset_index().sort_values("world_demand", ascending=False)
    labels = cmd_options.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
    code_map = dict(zip(labels, cmd_options["cmdCode"]))
    selected = st.selectbox("Select a commodity", labels[:80])
    sel_code = code_map[selected]

    cmd_trend = pen[pen["cmdCode"] == sel_code].groupby("year")["world_demand"].sum().reset_index().sort_values("year")
    cmd_trend["demand_B"] = cmd_trend["world_demand"] / 1e9
    cname = pen.loc[pen["cmdCode"] == sel_code, "commodity"].iloc[0]
    fig3 = px.line(cmd_trend, x="year", y="demand_B", markers=True, labels={"demand_B": "Import Demand (USD B)", "year": ""}, color_discrete_sequence=["#0F4C75"], title=cname)
    fig3.update_layout(margin=dict(t=40), height=320)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — GCC PENETRATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "GCC Penetration":
    st.title("GCC Export Penetration")
    st.markdown(
        "**Penetration %** = GCC total exports / addressable import demand from 40 countries. "
        "Low penetration on large-demand commodities = strategic opportunity."
    )

    pen = require("gcc_export_penetration.csv")
    latest_yr = int(pen["year"].max())
    pen_snap = pen[pen["year"] == latest_yr].copy()

    st.subheader("GCC Non-Fuel Export Aggregate Over Time")
    gcc_yr = pen.groupby("year")["gcc_exports"].sum().reset_index()
    gcc_yr["gcc_B"] = gcc_yr["gcc_exports"] / 1e9
    fig = px.area(gcc_yr, x="year", y="gcc_B", labels={"gcc_B": "GCC Exports (USD B)", "year": ""}, color_discrete_sequence=["#FF6B35"])
    fig.update_layout(margin=dict(t=10), height=280, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(f"Highest GCC Penetration ({latest_yr})")
        st.caption("Commodities where GCC already holds significant market share.")
        high_pen = pen_snap.sort_values("penetration_pct", ascending=False).head(15).copy()
        high_pen["label"] = high_pen["cmdCode"].astype(str) + " — " + high_pen["commodity"].str[:40]
        fig2 = px.bar(high_pen, y="label", x="penetration_pct", orientation="h", labels={"penetration_pct": "GCC Penetration %", "label": ""}, color_discrete_sequence=["#8338EC"])
        fig2.update_layout(margin=dict(t=10, l=10), height=420, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.subheader(f"Biggest Gaps ({latest_yr})")
        st.caption("Large import markets where GCC has barely entered.")
        gaps = pen_snap[(pen_snap["penetration_pct"] < 5) & (pen_snap["world_demand"] > pen_snap["world_demand"].quantile(0.5))].sort_values("world_demand", ascending=False).head(15).copy()
        if not gaps.empty:
            gaps["label"] = gaps["cmdCode"].astype(str) + " — " + gaps["commodity"].str[:40]
            gaps["demand_B"] = gaps["world_demand"] / 1e9
            fig3 = px.bar(gaps, y="label", x="demand_B", orientation="h", labels={"demand_B": "Import Demand (USD B)", "label": ""}, color_discrete_sequence=["#D62828"])
            fig3.update_layout(margin=dict(t=10, l=10), height=420, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Penetration Trend Over Time")
    cmd_opts = pen.groupby(["cmdCode", "commodity"])["world_demand"].sum().reset_index().sort_values("world_demand", ascending=False)
    labels = cmd_opts.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
    code_map = dict(zip(labels, cmd_opts["cmdCode"]))
    selected = st.selectbox("Select a commodity", labels[:80])
    sel_code = code_map[selected]
    pen_trend = pen[pen["cmdCode"] == sel_code].sort_values("year")
    fig4 = px.line(pen_trend, x="year", y="penetration_pct", markers=True, labels={"penetration_pct": "GCC Penetration %", "year": ""}, color_discrete_sequence=["#8338EC"])
    fig4.update_layout(margin=dict(t=10), height=300)
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — DEMAND FORECASTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Demand Forecasts":
    st.title("Demand Forecasts (2025-2028)")
    st.markdown("Holt-Winters forecasts of global import demand per commodity. Confidence band widens from +/-5% (year 1) to +/-20% (year 4).")

    files = require("demand_forecast_global.csv", "gcc_export_penetration.csv")
    fc = files["demand_forecast_global.csv"]
    pen = files["gcc_export_penetration.csv"]
    hist = pen.groupby(["cmdCode", "commodity", "year"])["world_demand"].sum().reset_index()

    fc_totals = fc.groupby(["cmdCode", "commodity"])["demand_ensemble"].sum().reset_index().sort_values("demand_ensemble", ascending=False)
    labels = fc_totals.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:55]}", axis=1).tolist()
    code_map = dict(zip(labels, fc_totals["cmdCode"]))
    selected = st.selectbox("Select a commodity", labels[:80])
    sel_code = code_map[selected]

    h = hist[hist["cmdCode"] == sel_code].sort_values("year")
    f = fc[fc["cmdCode"] == sel_code].sort_values("year")
    commodity_name = f["commodity"].iloc[0] if len(f) > 0 else sel_code

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["year"], y=h["world_demand"], mode="lines+markers", name="Historical Demand", line=dict(color="#0F4C75", width=2.5), marker=dict(size=6)))

    if not f.empty and not h.empty:
        bridge_yr = h["year"].max()
        bridge_val = h.loc[h["year"] == bridge_yr, "world_demand"]
        if not bridge_val.empty:
            f_ext = pd.concat([
                pd.DataFrame({"year": [bridge_yr], "demand_ensemble": [bridge_val.iloc[0]], "ci_lower": [bridge_val.iloc[0]], "ci_upper": [bridge_val.iloc[0]]}),
                f[["year", "demand_ensemble", "ci_lower", "ci_upper"]],
            ], ignore_index=True)
        else:
            f_ext = f[["year", "demand_ensemble", "ci_lower", "ci_upper"]]

        fig.add_trace(go.Scatter(x=f_ext["year"], y=f_ext["demand_ensemble"], mode="lines+markers", name="Forecast", line=dict(color="#D62828", width=2.5, dash="dash"), marker=dict(size=6)))
        fig.add_trace(go.Scatter(
            x=pd.concat([f_ext["year"], f_ext["year"][::-1]]),
            y=pd.concat([f_ext["ci_upper"], f_ext["ci_lower"][::-1]]),
            fill="toself", fillcolor="rgba(214,40,40,0.12)", line=dict(width=0), name="Confidence Band",
        ))

    fig.update_layout(title=f"Global Demand Forecast — {commodity_name}", yaxis_title="Import Demand (USD)", xaxis_title="", margin=dict(t=50), height=430, legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, use_container_width=True)

    if not f.empty:
        st.subheader("Forecast Values")
        summary = f[["year", "demand_ensemble", "ci_lower", "ci_upper"]].copy()
        summary.columns = ["Year", "Forecast", "Lower Bound", "Upper Bound"]
        for c in ["Forecast", "Lower Bound", "Upper Bound"]:
            summary[c] = summary[c].apply(fmt_usd)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Largest Forecasted Markets (4-Year Total)")
    top_fc = fc_totals.head(15).copy()
    top_fc["label"] = top_fc["cmdCode"].astype(str) + " — " + top_fc["commodity"].str[:45]
    top_fc["demand_B"] = top_fc["demand_ensemble"] / 1e9
    fig2 = px.bar(top_fc, y="label", x="demand_B", orientation="h", labels={"demand_B": "4-Year Forecast (USD B)", "label": ""}, color_discrete_sequence=["#D62828"])
    fig2.update_layout(margin=dict(t=10, l=10), height=430, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRICE OUTLOOK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Price Outlook":
    st.title("Price (Unit Value) Outlook")
    st.markdown("Forecasted GCC export unit values (USD/kg) identify commodities with rising or falling price trajectories.")

    uv_fc = require("gcc_export_unit_value_forecast.csv")

    uv_options = (
        uv_fc.groupby(["cmdCode", "commodity"])
        .agg(avg_uv=("uv_forecast", "mean"), cagr=("uv_cagr_pct", "first"), last_uv=("last_uv", "first"))
        .reset_index().sort_values("avg_uv", ascending=False)
    )
    labels = uv_options.apply(lambda r: f"{r['cmdCode']} — {r['commodity'][:50]}  (CAGR {r['cagr']:+.1f}%)", axis=1).tolist()
    code_map = dict(zip(labels, uv_options["cmdCode"]))
    selected = st.selectbox("Select a commodity", labels[:60])
    sel_code = code_map[selected]

    f = uv_fc[uv_fc["cmdCode"] == sel_code].sort_values("year")
    cname = f["commodity"].iloc[0] if len(f) > 0 else sel_code
    cagr_val = f["uv_cagr_pct"].iloc[0] if len(f) > 0 else 0
    last_uv_val = f["last_uv"].iloc[0] if len(f) > 0 and "last_uv" in f.columns else None

    fig = go.Figure()
    if last_uv_val is not None and not np.isnan(last_uv_val):
        first_fc_yr = int(f["year"].min())
        last_hist_yr = first_fc_yr - 1
        fig.add_trace(go.Scatter(x=[last_hist_yr], y=[last_uv_val], mode="markers", name=f"Last Historical ({last_hist_yr})", marker=dict(size=10, color="#0F4C75", symbol="diamond")))
        f_plot = pd.concat([pd.DataFrame({"year": [last_hist_yr], "uv_forecast": [last_uv_val]}), f[["year", "uv_forecast"]]], ignore_index=True)
    else:
        f_plot = f[["year", "uv_forecast"]]

    fig.add_trace(go.Scatter(x=f_plot["year"], y=f_plot["uv_forecast"], mode="lines+markers", name="Forecast", line=dict(color="#D62828", width=2.5, dash="dash"), marker=dict(size=6)))
    trend_text = "Rising" if cagr_val > 2 else "Falling" if cagr_val < -2 else "Stable"
    fig.update_layout(title=f"{cname}  |  Historical CAGR: {cagr_val:+.1f}%  ({trend_text})", yaxis_title="USD / kg", xaxis_title="", margin=dict(t=60), height=400, legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Fastest-Rising Prices")
        top_up = uv_options.sort_values("cagr", ascending=False).head(15).copy()
        top_up["commodity"] = top_up["commodity"].str[:50]
        st.dataframe(
            top_up[["cmdCode", "commodity", "avg_uv", "cagr"]].rename(columns={"commodity": "Commodity", "avg_uv": "Avg UV ($/kg)", "cagr": "CAGR %"})
            .style.format({"Avg UV ($/kg)": "${:.2f}", "CAGR %": "{:+.1f}%"}).background_gradient(subset=["CAGR %"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )
    with col_r:
        st.subheader("Fastest-Falling Prices")
        top_dn = uv_options.sort_values("cagr", ascending=True).head(15).copy()
        top_dn["commodity"] = top_dn["commodity"].str[:50]
        st.dataframe(
            top_dn[["cmdCode", "commodity", "avg_uv", "cagr"]].rename(columns={"commodity": "Commodity", "avg_uv": "Avg UV ($/kg)", "cagr": "CAGR %"})
            .style.format({"Avg UV ($/kg)": "${:.2f}", "CAGR %": "{:+.1f}%"}).background_gradient(subset=["CAGR %"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — OPPORTUNITY FINDER (main tool)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Opportunity Finder":
    st.title("Opportunity Finder")
    st.markdown(
        "Ranked GCC export opportunities scored by **demand (25%)**, **penetration gap (20%)**, "
        "**country viability (20%)**, **landing cost (15%)**, **ML growth signal (10%)**, and **price quality (10%)**. "
        "Re-exports and saturated markets (>40% penetration) are excluded."
    )

    opp = require("opportunity_rankings_full.csv")

    f1, f2, f3 = st.columns(3)
    gcc_sel = f1.selectbox("GCC Exporter", ["All"] + sorted(opp["gcc_country"].unique()))
    dest_sel = f2.selectbox("Target Market", ["All"] + sorted(opp["dest_country"].unique()))
    grade_list = sorted(opp["grade"].dropna().unique()) if "grade" in opp.columns else []
    grade_sel = f3.multiselect("Market Grade", grade_list, default=grade_list)

    search = st.text_input("🔍 Search commodity (keyword)", "")

    df = opp.copy()
    if gcc_sel != "All":
        df = df[df["gcc_country"] == gcc_sel]
    if dest_sel != "All":
        df = df[df["dest_country"] == dest_sel]
    if grade_sel and "grade" in df.columns:
        df = df[df["grade"].isin(grade_sel)]
    if search.strip():
        df = df[df["commodity"].str.contains(search.strip(), case=False, na=False)]

    df = df.sort_values("opportunity_score", ascending=False)
    st.caption(f"Showing {len(df):,} opportunities")

    max_n = min(300, len(df)) if len(df) > 10 else max(10, len(df))
    top_n = st.slider("Results to display", 10, max_n, min(30, max_n), 10)
    df_show = df.head(top_n)

    display_cols = {"gcc_country": "GCC Exporter", "dest_country": "Target Market", "commodity": "Commodity", "opportunity_score": "Score", "grade": "Grade", "demand_4y_total": "4Y Demand", "penetration_pct": "GCC Pen %", "uv_mean": "UV ($/kg)", "uv_cagr": "Price CAGR %", "ml_growth_prob": "ML Growth P", "recommended_transport": "Transport"}
    available = {k: v for k, v in display_cols.items() if k in df_show.columns}
    table = df_show[list(available.keys())].rename(columns=available).copy()

    fmt_map = {}
    if "Score" in table.columns: fmt_map["Score"] = "{:.3f}"
    if "GCC Pen %" in table.columns: fmt_map["GCC Pen %"] = "{:.1f}"
    if "UV ($/kg)" in table.columns: fmt_map["UV ($/kg)"] = "${:.2f}"
    if "Price CAGR %" in table.columns: fmt_map["Price CAGR %"] = "{:+.1f}"
    if "ML Growth P" in table.columns: fmt_map["ML Growth P"] = "{:.2f}"
    if "4Y Demand" in table.columns:
        table["4Y Demand"] = table["4Y Demand"].apply(lambda x: fmt_usd(x) if pd.notna(x) else "—")
    if "Transport" in table.columns:
        table["Transport"] = table["Transport"].map(lambda x: f"{TRANSPORT_ICONS.get(str(x), '')} {x}" if pd.notna(x) else "—")
    if "Commodity" in table.columns:
        table["Commodity"] = table["Commodity"].str[:55]

    styled = table.style.format(fmt_map)
    if "Score" in table.columns:
        styled = styled.background_gradient(subset=["Score"], cmap="YlGn")
    st.dataframe(styled, use_container_width=True, hide_index=True, height=520)

    if "opportunity_rationale" in df_show.columns:
        with st.expander("View scoring rationale for top results"):
            for _, row in df_show.head(10).iterrows():
                transport = row.get("recommended_transport", "")
                icon = TRANSPORT_ICONS.get(str(transport), "")
                st.markdown(f"**{row['gcc_country']} → {row['dest_country']}** | _{row['commodity'][:50]}_ | Score: **{row['opportunity_score']:.3f}** | {icon} {transport}")
                st.caption(row.get("opportunity_rationale", "—"))

    st.divider()
    st.subheader("Opportunity Matrix — Demand vs Penetration Gap")
    st.caption("Bubble size = score · Color = grade · Best opportunities in top-right.")

    if len(df) > 0 and all(c in df.columns for c in ["demand_4y_total", "pen_opportunity", "opportunity_score"]):
        bubble_df = df.head(150).copy()
        d_min, d_max = bubble_df["demand_4y_total"].min(), bubble_df["demand_4y_total"].max()
        bubble_df["demand_norm"] = (bubble_df["demand_4y_total"] - d_min) / (d_max - d_min + 1e-9)
        bubble_df["label"] = bubble_df["dest_country"] + " — " + bubble_df["commodity"].str[:30]

        fig = px.scatter(
            bubble_df, x="demand_norm", y="pen_opportunity", size="opportunity_score",
            color="grade" if "grade" in bubble_df.columns else None, hover_name="label",
            hover_data={"opportunity_score": ":.3f", "demand_norm": False, "pen_opportunity": ":.2f"},
            color_discrete_map=GRADE_COLORS,
            labels={"demand_norm": "Demand Score (normalised)", "pen_opportunity": "Penetration Opportunity (1 - pen%)"},
            size_max=22,
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="grey", opacity=0.4)
        fig.add_vline(x=0.5, line_dash="dash", line_color="grey", opacity=0.4)
        fig.add_annotation(x=0.85, y=0.97, text="SWEET SPOT", showarrow=False, font=dict(color="#0B6E4F", size=13, family="Arial Black"), xref="paper", yref="paper")
        fig.update_layout(margin=dict(t=10), height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    download_cols = [c for c in list(display_cols.keys()) + ["opportunity_rationale"] if c in df.columns]
    st.download_button("Download filtered results as CSV", df[download_cols].to_csv(index=False).encode("utf-8"), "gcc_opportunities_filtered.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 — COUNTRY PROFILES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Country Profiles":
    st.title("Destination Country Profiles")
    st.markdown("Each country is graded on World Bank indicators: **Economic Performance (45%)**, **Governance & Stability (30%)**, **Financial & Infrastructure (25%)**.")

    viab = require("destination_country_viability.csv")
    opp = load("opportunity_rankings_full.csv")

    viab_sorted = viab.sort_values("viability_score", ascending=False).copy()
    fig = px.bar(viab_sorted, y="country", x="viability_score", orientation="h", color="grade", color_discrete_map=GRADE_COLORS, labels={"viability_score": "Viability Score (0-1)", "country": "", "grade": "Grade"})
    fig.update_layout(margin=dict(t=10, l=10), height=max(400, len(viab_sorted) * 24), yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.06))
    st.plotly_chart(fig, use_container_width=True)

    if opp is not None and "lpi_score" in opp.columns:
        st.divider()
        st.subheader("Logistics & Trade Barriers")
        st.caption("Landing cost score = 40% inverted tariff + 35% LPI + 25% inverted distance. Higher = easier to reach.")
        landing = opp.groupby("dest_country").agg(lpi_score=("lpi_score", "first"), avg_tariff=("mfn_tariff_rate", "mean"), avg_dist_km=("dist_km", "mean"), landing_cost_n=("landing_cost_n", "mean")).reset_index().sort_values("landing_cost_n", ascending=False)
        landing.columns = ["Market", "LPI Score", "Avg MFN Tariff %", "Avg Distance (km)", "Landing Cost Score"]
        st.dataframe(
            landing.style.format({"LPI Score": "{:.2f}", "Avg MFN Tariff %": "{:.1f}%", "Avg Distance (km)": "{:,.0f}", "Landing Cost Score": "{:.3f}"}).background_gradient(subset=["Landing Cost Score"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

    if opp is not None and "recommended_transport" in opp.columns:
        st.divider()
        st.subheader("Transport Mode Recommendations")
        transport_summary = opp.groupby(["gcc_country", "recommended_transport"]).size().reset_index(name="count")
        fig2 = px.bar(transport_summary, x="gcc_country", y="count", color="recommended_transport", labels={"gcc_country": "", "count": "Opportunities", "recommended_transport": "Mode"}, color_discrete_map={"Sea": "#0F4C75", "Air": "#EF476F", "Land": "#06D6A0"}, barmode="stack")
        fig2.update_layout(margin=dict(t=10), height=350, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 8 — MODEL CONFIDENCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Model Confidence":
    st.title("Model Confidence & Evaluation")
    st.markdown("How reliable are the demand forecasts and ML growth predictions?")

    bt = require("backtest_results.csv")
    metrics = derive_backtest_metrics()

    st.subheader("Demand Forecast — Holt-Winters Back-test (2023-2024 Holdout)")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{metrics['r2']:.3f}")
        c2.metric("MAPE (>$1M series)", f"{metrics['mape_sig']:.1f}%")
        c3.metric("MAE", fmt_usd(metrics["mae"]))
        c4.metric("Series Tested", f"{metrics['n_total']:,}")

        mape_val = metrics["mape_sig"]
        if mape_val < 20:
            st.success("MAPE under 20% — forecasts are reliable for prioritisation.")
        elif mape_val < 40:
            st.info("MAPE 20-40% — directional rankings are valid; point estimates are approximate.")
        else:
            st.warning("MAPE above 40% — use rankings directionally; point estimates are order-of-magnitude.")

    st.subheader("Actual vs Predicted")
    bt_clean = bt.dropna(subset=["actual", "predicted"]).query("actual > 0")
    bt_plot = bt_clean[bt_clean["actual"] <= bt_clean["actual"].quantile(0.99)]
    fig = px.scatter(bt_plot, x="actual", y="predicted", opacity=0.35, labels={"actual": "Actual (USD)", "predicted": "Forecast (USD)"}, color_discrete_sequence=["#0F4C75"], log_x=True, log_y=True)
    mx = max(bt_plot["actual"].max(), bt_plot["predicted"].max())
    fig.add_trace(go.Scatter(x=[bt_plot["actual"].min(), mx], y=[bt_plot["actual"].min(), mx], mode="lines", line=dict(color="red", dash="dash", width=1.5), name="Perfect"))
    fig.update_layout(margin=dict(t=10), height=440)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Forecast Error Distribution (>$1M Series)")
    bt_sig = bt_clean[bt_clean["actual"] > 1_000_000].copy()
    if not bt_sig.empty:
        bt_sig["pct_error"] = ((bt_sig["predicted"] - bt_sig["actual"]) / bt_sig["actual"]) * 100
        bt_sig["pct_error"] = bt_sig["pct_error"].clip(-200, 200)
        fig2 = px.histogram(bt_sig, x="pct_error", nbins=60, labels={"pct_error": "% Forecast Error"}, color_discrete_sequence=["#1B9AAA"])
        fig2.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
        fig2.update_layout(margin=dict(t=10), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("ML Classification — P(Structural Growth)")
    st.markdown(
        "The ML model predicts whether a market-commodity pair is **structurally growing** "
        "(next-year import growth above the training-set median). "
        "Probabilities are **isotonic-calibrated** so a predicted 70% reflects ~70% true likelihood."
    )
    ml_probs = load("ml_growth_probabilities.csv")
    if ml_probs is not None and not ml_probs.empty:
        col1, col2 = st.columns(2)
        col1.metric("Avg ML Growth Probability", f"{ml_probs['ml_growth_prob'].mean():.2f}")
        col2.metric("Pairs with P > 60%", f"{(ml_probs['ml_growth_prob'] > 0.6).sum():,} of {len(ml_probs):,}")
