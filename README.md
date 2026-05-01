# GCC Export Opportunity Dashboard

**Predicting Export Market Growth & Price Potential to Support Strategic Trade Diversification**

An interactive Streamlit dashboard that helps consultants identify the most attractive non-fuel export opportunities for GCC countries across 34+ destination markets and hundreds of HS2 commodities.

Built for **OCO Global** as part of the AUB MSBA Capstone Project by Georges Elkassouf & Joseph Hobeika.

---

## Setup

### 1. Run the Colab notebook

Open `capstone_code.ipynb` in Google Colab and run all cells through the final export section. This produces the CSV files in your Drive's outputs folder.

### 2. Copy data files

Download the CSV files and place them into the `data/` directory of this repo:

```
data/
  gcc_export_penetration.csv          # GCC-aggregate penetration panel (all 6 members combined)
  demand_forecast_global.csv          # Holt-Winters 4-year global demand forecasts
  demand_forecast_by_dest_country.csv # Destination-level demand forecasts
  gcc_export_unit_value_forecast.csv  # Unit value (price) forecasts per GCC country
  destination_country_viability.csv   # Country viability grades & scores
  opportunity_rankings_full.csv       # Full composite opportunity scores (main dashboard feed)
  top20_per_gcc_country.csv           # Top 20 opportunities per GCC exporter
  ml_growth_probabilities.csv         # ML structural growth probabilities (country × cmdCode)
  backtest_results.csv                # Forecast backtest actual vs predicted
  reexport_flags.csv                  # Re-export filter flags per GCC country × cmdCode
```

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
| **Market Demand** | Which commodities have the highest import demand across the 34 destination markets? |
| **GCC Penetration** | Where does GCC already penetrate? Where are the whitespace gaps? (aggregate across all 6 GCC members) |
| **Demand Forecasts** | Holt-Winters 4-year demand projections with confidence bands per commodity |

---

## Scoring Formula

The composite opportunity score is a weighted sum of six sub-scores, each normalised to [0, 1]:

| Component | Weight | Source |
| --- | --- | --- |
| Demand Forecast (4-year total) | **30%** | Holt-Winters forecasts — `demand_forecast_by_dest_country.csv` |
| Penetration Gap (1 − current GCC share) | **20%** | `gcc_export_penetration.csv` |
| Country Viability (World Bank composite) | **20%** | `destination_country_viability.csv` |
| ML Structural Growth Probability | **15%** | Random Forest + XGBoost ensemble — `ml_growth_probabilities.csv` |
| Price Quality (unit value level + CAGR) | **10%** | `gcc_export_unit_value_forecast.csv` |
| Landing Cost Index (MFN tariff + LPI) | **5%** | WITS/TRAINS tariffs + World Bank LPI 2023 |

Exclusions applied before scoring: re-exports (`reexport_flags.csv`), fuels (HS27), precious stones (HS71), arms (HS93), unclassified (HS99).

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

### `ml_growth_probabilities.csv`

| Column | Description |
| --- | --- |
| `country` | Destination country name |
| `cmdCode` | HS2 commodity code |
| `ml_growth_prob` | Calibrated probability of structural growth (0–1) |

### `opportunity_rankings_full.csv`
One row per GCC exporter × destination country × HS2 commodity combination.

Key columns: `gcc_country`, `cmdCode`, `commodity`, `dest_country`, `opportunity_score`, `grade`, `demand_4y_total`, `penetration_pct`, `pen_opportunity`, `ml_growth_prob`, `uv_mean`, `uv_cagr`, `weighted_dist_km`, `lpi_score`, `mfn_tariff_rate`, `recommended_transport`, `opportunity_rationale`.

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
