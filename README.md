# GCC Export Opportunity Dashboard

**Predicting Export Market Growth & Price Potential to Support Strategic Trade Diversification**

An interactive Streamlit dashboard that helps consultants identify the most attractive non-fuel export opportunities for GCC countries across 40 destination markets and hundreds of HS4 commodities.

Built for **OCO Global** as part of the AUB MSBA Capstone Project.

---

## Setup

### 1. Run the Colab notebook

Open `capstone_code.ipynb` in Google Colab and run all cells through **Section 45** (the export cell). This produces 10 CSV files in your Drive's `outputs/` folder.

### 2. Copy data files

Download the 10 CSV files from your Google Drive `outputs/` folder and place them into the `data/` directory of this repo:

```
data/
  demand_forecast_global.csv
  demand_forecast_by_dest_country.csv
  gcc_export_unit_value_forecast.csv
  destination_country_viability.csv
  opportunity_rankings_full.csv
  top20_per_gcc_country.csv
  gcc_export_penetration.csv
  backtest_results.csv
  ml_growth_probabilities.csv
  reexport_flags.csv
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
|---|---|
| **Executive Summary** | Total addressable demand, GCC export size, best opportunity per country |
| **Market Demand** | Which commodities have the highest import demand across 40 destinations? |
| **GCC Penetration** | Where does GCC already penetrate? Where are the whitespace gaps? |
| **Demand Forecasts** | Holt-Winters 4-year demand projections with confidence bands per commodity |
| **Opportunity Finder** | **Main tool** — pick a GCC country + commodity, see top destination markets ranked by score |

---

## Scoring Formula

- **25%** — Forecasted demand (4-year total)
- **20%** — Penetration gap (1 - current GCC share)
- **20%** — Country viability (World Bank indicators)
- **15%** — Landing cost (tariff + LPI + distance, commodity-adjusted)
- **10%** — ML structural growth probability (calibrated)
- **10%** — Price quality (unit value level + CAGR)

Exclusions: re-exports, fuels (HS27), precious stones (HS71), arms (HS93), unclassified (HS99).

---

## Data Sources

- **UN Comtrade** & **ITC Trade Map** — bilateral trade data (HS4, 2015-2024)
- **World Bank** — LPI 2023, governance & development indicators
- **WTO** — MFN applied tariff rates
- **CEPII** — geographic distance matrix
