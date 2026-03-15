# SovWatch — Global Sovereign Debt Early Warning System

Predicts sovereign debt crises 2 years in advance using a Hybrid BiLSTM + GJR-GARCH + FinBERT model across 188 countries (2000–2025).

## Architecture
- **BiLSTM** — reads 5-year economic sequences bidirectionally
- **GJR-GARCH** — bond market volatility with skip connection
- **FinBERT** — financial news sentiment scoring

## Features
- 188 countries across all income groups
- 40+ macroeconomic, volatility, and sentiment features
- Interactive Streamlit dashboard with country selector and year slider
- AUROC 0.79 — outperforms XGBoost, Random Forest, Logistic Regression

## Project Structure
```
sovereign_debt/
├── data_collection/
│   ├── 01_macro_data.py       # World Bank + FRED data
│   ├── 02_news_sentiment.py   # GDELT + FinBERT
│   ├── 03_garch_volatility.R  # GJR-GARCH models
│   └── 04_build_features.py   # Feature engineering
├── models/
│   ├── 05_lstm_model.py       # BiLSTM training
│   └── 06_baselines.py        # Comparison models
├── app.py                     # Streamlit dashboard
└── requirements.txt
```

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the pipeline (in order)
```bash
python data_collection/01_macro_data.py
python data_collection/02_news_sentiment.py
Rscript data_collection/03_garch_volatility.R
python data_collection/04_build_features.py
python models/05_lstm_model.py
python models/06_baselines.py
```

### Launch dashboard
```bash
streamlit run app.py
```

## Data Sources
- World Bank WDI — https://databank.worldbank.org
- FRED — https://fred.stlouisfed.org
- GDELT — https://www.gdeltproject.org
- IMF DSA — https://www.imf.org/en/Publications/DSA

## Requirements
- Python 3.10+
- R 4.0+ with rugarch package
- FRED API key (free) — https://fred.stlouisfed.org/docs/api/api_key.html