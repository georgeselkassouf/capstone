# Trade Opportunity Engine

**Predicting Export Market Growth & Price Potential to Support Strategic Trade Diversification**

An interactive Streamlit dashboard that helps consultants identify the most attractive non-fuel export opportunities for GCC countries across 34+ destination markets and hundreds of HS2 commodities.

Built for **OCO Global** as part of the AUB MSBA Capstone Project by Georges Elkassouf & Joseph Hobeika.

---

## Setup

### 1. Run the Colab notebook

Open `capstone_code.ipynb` in Google Colab and run all cells through the final export section. This produces the CSV files in your Drive's outputs folder.

### 2. Copy data files

The app requires **three CSV files** in the `data/` directory. All scoring components (viability grades, ML probabilities, unit values, tariffs, LPI, rationale strings) are pre-computed by the notebook and embedded in `opportunity_rankings_full.csv`.

```
data/
  gcc_export_penetration.csv          # GCC-aggregate penetration panel (all 6 members combined)
  demand_forecast_global.csv          # Holt-Winters 4-year global demand forecasts
  opportunity_rankings_full.csv       # Full composite opportunity scores (main dashboard feed)
```

> **Optional:** `backtest_results.csv` — if present in `data/`, the app will compute back-test metrics (R², MAPE) automatically. The app runs fine without it.

> **Pipeline-only outputs** (produced by the notebook but not loaded directly by the app): `demand_forecast_by_dest_country.csv`, `gcc_export_unit_value_forecast.csv`, `destination_country_viability.csv`, `top20_per_gcc_country.csv`, `ml_growth_probabilities.csv`, `reexport_flags.csv`. These feed the scoring pipeline in the notebook and their outputs are embedded in `opportunity_rankings_full.csv`.

### 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 4. Deploy on Streamlit Cloud

1. Push this repo to GitHub (with the `data/` folder included).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repo.
4. Set the main file path to `app.py`.
5. Deploy.

---

## Dashboard Pages

| Page | What it answers |
| --- | --- |
| **Opportunity Finder** | **Main tool** — pick a GCC country + commodity, see top destination markets ranked by composite score |
| **Executive Summary** | Total addressable demand, combined GCC export size, best opportunity per country |
| **Market Demand** | Which commodities have the highest import demand across the 40 destination countries? |
| **GCC Penetration** | Where does GCC already penetrate? Where are the whitespace gaps? (aggregate across all 6 GCC members) |
| **Demand Forecasts** | Holt-Winters 4-year demand projections with confidence bands per commodity; filterable by GCC exporter to surface the most relevant commodities |

---

## Scoring Formula

The composite opportunity score is a weighted sum of six sub-scores, each normalised to [0, 1]:

| Component | Weight | Source |
| --- | --- | --- |
| Demand Forecast (4-year total) | **25%** | Pre-computed by notebook — `demand_4y_total` column in `opportunity_rankings_full.csv` |
| Penetration Gap (1 − current GCC share) | **20%** | `pen_opportunity` column in `opportunity_rankings_full.csv` |
| Country Viability (World Bank composite) | **20%** | `grade` / viability score columns in `opportunity_rankings_full.csv` |
| Landing Cost Index (MFN tariff + LPI) | **15%** | `mfn_tariff_rate` + `lpi_score` columns in `opportunity_rankings_full.csv` |
| ML Growth Signal | **10%** | Random Forest + XGBoost ensemble — `ml_growth_prob` column in `opportunity_rankings_full.csv` |
| Price Quality (unit value level + CAGR) | **10%** | `uv_mean` + `uv_cagr` columns in `opportunity_rankings_full.csv` |

Exclusions applied before scoring: re-exports, fuels (HS27), precious stones (HS71), arms (HS93), unclassified (HS99).

---

## Key Data Schemas

### `gcc_export_penetration.csv`
Aggregated across all 6 GCC member states (UAE, Saudi Arabia, Qatar, Kuwait, Oman, Bahrain).

| Column | Description |
| --- | --- |
| `cmdCode` | HS2 commodity code |
| `year` | Reporting year (2015–2024) |
| `world_demand` | Total import demand across destination markets (USD) |
| `commodity` | HS2 commodity description |
| `gcc_exports` | Combined GCC exports to destination markets (USD) |
| `penetration_pct` | `gcc_exports / world_demand × 100` |

### `demand_forecast_global.csv`

| Column | Description |
| --- | --- |
| `cmdCode` | HS2 commodity code |
| `commodity` | HS2 commodity description |
| `year` | Forecast year (2025–2028) |
| `demand_ensemble` | Holt-Winters point forecast (USD) |
| `ci_lower` | Lower confidence bound |
| `ci_upper` | Upper confidence bound |

### `opportunity_rankings_full.csv`
One row per GCC exporter × destination country × HS2 commodity combination. All scoring sub-components are pre-computed and embedded here.

Key columns: `gcc_country`, `cmdCode`, `commodity`, `dest_country`, `opportunity_score`, `grade`, `demand_4y_total`, `penetration_pct`, `pen_opportunity`, `ml_growth_prob`, `uv_mean`, `uv_cagr`, `weighted_dist_km`, `dist_km`, `lpi_score`, `mfn_tariff_rate`, `recommended_transport`, `opportunity_rationale`.

---

## Data Sources

| Source | Usage |
| --- | --- |
| **UN Comtrade** | Trade flows (HS2, 2015–2024, annual) |
| **World Bank LPI 2023** | Logistics Performance Index scores |
| **WITS / UNCTAD TRAINS** | MFN applied tariff rates (HS6 → HS2) |
| **World Bank Open Data** | Country viability indicators (2021–2023) |
| **CEPII** | Geographic distance matrix (capital-to-capital) |

---

## Transport Mode Logic

Rule-based four-filter recommendation (no ML — Comtrade has no mode-level labels):

1. **Geography** — landlocked pairs → Land
2. **Perishability + Long-haul** — short shelf-life + distance > 3 000 km → Air
3. **Unit value** — high-value / low-bulk commodities → Air
4. **Default** → Sea

---

## About

AUB MSBA Capstone · Spring 2026 · in partnership with OCO Global  
Students: Georges Elkassouf (202574690) · Joseph Hobeika (202574794)  
Advisor: Kinan Morad, OCO Global
