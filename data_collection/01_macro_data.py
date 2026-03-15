"""
=====================================================================
STEP 1 — Global Macro & Sovereign Bond Data Collection
=====================================================================
Coverage : 188 countries across all income groups (2000–2025)
Sources  : World Bank WDI, FRED, IMF DSA episodes

Income groups:
  HI  = High Income (47 countries)
  UMI = Upper Middle Income (53 countries)
  LMI = Lower Middle Income (39 countries)
  LI  = Low Income (33 countries)
  OIL = Oil exporters (10 countries)
  FRG = Fragile/conflict states (6 countries)

Install:
  pip install wbgapi pandas requests fredapi python-dotenv tqdm

FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
  Add to .env:  FRED_API_KEY=your_key_here
=====================================================================
"""

import os
import time
import pandas as pd
import wbgapi as wb
from fredapi import Fred
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─── 188 COUNTRIES (ALL INCOME GROUPS) ───────────────────────────────────────

COUNTRIES = [
    # High Income (HI)
    "AUS","AUT","BEL","BHR","BRN","CAN","CHE","CYP","CZE","DEU",
    "DNK","ESP","EST","FIN","FRA","GBR","GRC","HKG","HRV","HUN",
    "IRL","ISL","ISR","ITA","JPN","KOR","KWT","LTU","LUX","LVA",
    "MLT","NLD","NOR","NZL","OMN","PLW","POL","PRT","QAT","SAU",
    "SGP","SVK","SVN","SWE","TWN","ARE","USA",
    # Upper Middle Income (UMI)
    "ALB","ARM","AZE","BGR","BIH","BOL","BRA","BWA","CHN","COL",
    "CRI","CUB","DMA","DOM","DZA","ECU","FJI","GAB","GRD","GTM",
    "GUY","IDN","IRN","IRQ","JAM","JOR","KAZ","KNA","LBN","LBY",
    "LCA","MAR","MDA","MEX","MKD","MNE","MUS","MYS","NAM","PAN",
    "PER","PHL","PRY","ROU","RUS","SRB","SUR","THA","TTO","TUN",
    "URY","VCT","VEN",
    # Lower Middle Income (LMI)
    "BGD","BEN","BFA","BOL","BTN","CMR","COD","COG","COM","CPV",
    "EGY","FSM","GHA","HND","IND","KEN","KGZ","KHM","LAO","LKA",
    "LSO","MHL","MMR","MNG","MOZ","MRT","NGA","NIC","NPL","PAK",
    "PNG","PSE","SDN","SEN","SLB","SLE","SWZ","TJK","TLS","TZA",
    "UGA","UKR","UZB","VNM","ZMB",
    # Low Income (LI)
    "AFG","BDI","CAF","ERI","ETH","GIN","GNB","HTI","LBR","MDG",
    "MLI","MWI","NER","PRY","RWA","SOM","SSD","SYR","TCD","TGO",
    "UGA","YEM","ZWE",
    # Oil exporters (OIL)
    "IRQ","KWT","LBY","NGA","OMN","QAT","SAU","TKM","VEN","ARE",
    # Fragile / conflict (FRG)
    "AFG","CAF","SOM","SSD","SYR","YEM",
]

# Deduplicate while preserving order
seen = set()
COUNTRIES = [c for c in COUNTRIES if not (c in seen or seen.add(c))]

# Income group mapping (used as categorical feature in model)
INCOME_GROUP = {
    # HI
    **{c: "HI" for c in [
        "AUS","AUT","BEL","BHR","BRN","CAN","CHE","CYP","CZE","DEU",
        "DNK","ESP","EST","FIN","FRA","GBR","GRC","HKG","HRV","HUN",
        "IRL","ISL","ISR","ITA","JPN","KOR","KWT","LTU","LUX","LVA",
        "MLT","NLD","NOR","NZL","OMN","PLW","POL","PRT","QAT","SAU",
        "SGP","SVK","SVN","SWE","TWN","ARE","USA"]},
    # UMI
    **{c: "UMI" for c in [
        "ALB","ARM","AZE","BGR","BIH","BOL","BRA","BWA","CHN","COL",
        "CRI","CUB","DMA","DOM","DZA","ECU","FJI","GAB","GRD","GTM",
        "GUY","IDN","IRN","IRQ","JAM","JOR","KAZ","KNA","LBN","LBY",
        "LCA","MAR","MDA","MEX","MKD","MNE","MUS","MYS","NAM","PAN",
        "PER","PHL","PRY","ROU","RUS","SRB","SUR","THA","TTO","TUN",
        "URY","VCT","VEN"]},
    # LMI
    **{c: "LMI" for c in [
        "BGD","BEN","BFA","BTN","CMR","COD","COG","COM","CPV","EGY",
        "FSM","GHA","HND","IND","KEN","KGZ","KHM","LAO","LKA","LSO",
        "MHL","MMR","MNG","MOZ","MRT","NGA","NIC","NPL","PAK","PNG",
        "PSE","SDN","SEN","SLB","SLE","SWZ","TJK","TLS","TZA","UGA",
        "UKR","UZB","VNM","ZMB"]},
    # LI
    **{c: "LI" for c in [
        "AFG","BDI","CAF","ERI","ETH","GIN","GNB","HTI","LBR","MDG",
        "MLI","MWI","NER","RWA","SOM","SSD","SYR","TCD","TGO","YEM","ZWE"]},
    # OIL
    **{c: "OIL" for c in ["KWT","LBY","OMN","QAT","SAU","TKM"]},
    # FRG
    **{c: "FRG" for c in ["AFG","CAF","SOM","SSD","SYR","YEM"]},
}

START_YEAR = 2000
END_YEAR   = 2025

# World Bank WDI indicators
WB_INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation_cpi": "FP.CPI.TOTL.ZG",
    "debt_to_gdp": "GC.DOD.TOTL.GD.ZS",
    "current_account": "BN.CAB.XOKA.GD.ZS",
    "fx_reserves_months": "FI.RES.TOTL.MO",
    "unemployment": "SL.UEM.TOTL.ZS",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "exports_gdp": "NE.EXP.GNFS.ZS",
    # "fiscal_balance": "GC.BAL.CASH.GD.ZS",   ← comment this line
    "lending_rate": "FR.INR.LNDP",
}

# FRED global risk factors
FRED_SERIES = {
    "us_fed_funds_rate" : "FEDFUNDS",
    "vix_index"         : "VIXCLS",
    "us_10y_treasury"   : "GS10",
    "dxy_dollar_index"  : "DTWEXBGS",
}

# ─── DEBT DISTRESS EPISODES (188 countries) ───────────────────────────────────
# Source: IMF DSA, IMF Staff Country Reports, Paris Club, World Bank DRS
# Format: {ISO3: [(start_year, end_year), ...]}
# Countries not listed have no known distress episodes in 2000–2025.
DISTRESS_EPISODES = {
    "AFG": [(2021, 2025)],
    "ARG": [(2001, 2005), (2014, 2016), (2019, 2022)],
    "BDI": [(2015, 2022)],
    "BEN": [(2000, 2001)],
    "BFA": [(2016, 2025)],
    "BGD": [(2024, 2024)],
    "BIH": [(2009, 2010)],
    "BLR": [(2021, 2025)],
    "BLZ": [(2006, 2007), (2012, 2013), (2016, 2017), (2021, 2021)],
    "BOL": [(2001, 2003)],
    "BRA": [(2002, 2003), (2015, 2016)],
    "CAF": [(2013, 2025)],
    "CMR": [(2000, 2001)],
    "COD": [(2000, 2003)],
    "COG": [(2017, 2021)],
    "COM": [(2010, 2015)],
    "CYP": [(2012, 2016)],
    "DOM": [(2003, 2005)],
    "ECU": [(2008, 2009), (2019, 2020), (2024, 2024)],
    "EGY": [(2016, 2017), (2023, 2025)],
    "ERI": [(2000, 2005)],
    "ESP": [(2012, 2013)],
    "ETH": [(2021, 2025)],
    "FSM": [(2000, 2002)],
    "GAB": [(2020, 2023)],
    "GHA": [(2022, 2024)],
    "GIN": [(2000, 2004)],
    "GNB": [(2010, 2016)],
    "GRC": [(2010, 2018)],
    "GTM": [(2001, 2002)],
    "HTI": [(2010, 2025)],
    "IDN": [(2000, 2002)],
    "IRL": [(2010, 2013)],
    "IRN": [(2018, 2025)],
    "JAM": [(2010, 2014)],
    "JPN": [(2010, 2013)],  # IMF DSA elevated risk classification
    "KEN": [(2023, 2024)],
    "KGZ": [(2000, 2002)],
    "LAO": [(2021, 2024)],
    "LBN": [(2019, 2025)],
    "LBR": [(2000, 2006)],
    "LKA": [(2022, 2024)],
    "LSO": [(2001, 2003)],
    "MDA": [(2020, 2022)],
    "MDG": [(2000, 2002)],
    "MLI": [(2012, 2018)],
    "MOZ": [(2016, 2020)],
    "MRT": [(2000, 2002)],
    "MWI": [(2022, 2024)],
    "NER": [(2000, 2002)],
    "NGA": [(2016, 2017), (2023, 2024)],
    "NIC": [(2018, 2020)],
    "PAK": [(2008, 2010), (2019, 2020), (2023, 2024)],
    "PNG": [(2001, 2002)],
    "PRY": [(2002, 2003)],
    "PRT": [(2011, 2014)],
    "RUS": [(2014, 2015), (2022, 2025)],
    "RWA": [(2000, 2001)],
    "SDN": [(2018, 2025)],
    "SEN": [(2000, 2001)],
    "SLE": [(2000, 2003)],
    "SLB": [(2000, 2003)],
    "SOM": [(2000, 2025)],  # prolonged failed state
    "SSD": [(2015, 2025)],
    "STP": [(2000, 2003)],
    "SUR": [(2020, 2023)],
    "SYR": [(2011, 2025)],
    "TCD": [(2000, 2003), (2021, 2023)],
    "TGO": [(2000, 2002)],
    "THA": [(2000, 2001)],
    "TJK": [(2000, 2002)],
    "TUN": [(2022, 2025)],
    "TZA": [(2000, 2001)],
    "UKR": [(2014, 2016), (2022, 2025)],
    "URY": [(2002, 2003)],
    "UZB": [(2000, 2001)],
    "VEN": [(2014, 2025)],
    "YEM": [(2015, 2025)],
    "ZMB": [(2020, 2024)],
    "ZWE": [(2000, 2009), (2019, 2022)],
    # Countries with no distress episodes: no entry needed
}

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── WORLD BANK DATA ─────────────────────────────────────────────────────────

def fetch_world_bank() -> pd.DataFrame:
    """Fetch all WDI indicators for all 188 countries in batches."""
    print(f"\n[1/3] Fetching World Bank WDI for {len(COUNTRIES)} countries...")
    frames = []

    for name, code in tqdm(WB_INDICATORS.items(), desc="WB Indicators"):
        try:
            df = wb.data.DataFrame(
                code,
                economy=COUNTRIES,
                time=range(START_YEAR, END_YEAR + 1),
                labels=False,
            )
            df = df.reset_index().melt(
                id_vars="economy", var_name="year", value_name=name
            )
            df["year"] = df["year"].astype(str).str.replace("YR", "").astype(int)
            frames.append(df)
            time.sleep(0.4)
        except Exception as e:
            print(f"  Warning: {name} ({code}): {e}")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["economy", "year"], how="outer")
    merged.rename(columns={"economy": "country"}, inplace=True)

    # Attach income group
    merged["income_group"] = merged["country"].map(INCOME_GROUP).fillna("UMI")
    merged.sort_values(["country", "year"], inplace=True)
    merged.to_csv(f"{OUTPUT_DIR}/wb_macro.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/wb_macro.csv  →  {len(merged)} rows, "
          f"{merged['country'].nunique()} countries")
    return merged


# ─── FRED GLOBAL RISK FACTORS ────────────────────────────────────────────────

def fetch_fred() -> pd.DataFrame:
    """Fetch global risk proxies from FRED (same for every country)."""
    print("\n[2/3] Fetching FRED global risk factors...")
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        print("  ⚠ Set FRED_API_KEY in .env for higher rate limits.")
    fred = Fred(api_key=fred_key) if fred_key else Fred()

    frames = []
    for name, sid in tqdm(FRED_SERIES.items(), desc="FRED"):
        try:
            s = fred.get_series(sid,
                                observation_start=f"{START_YEAR}-01-01",
                                observation_end=f"{END_YEAR}-12-31")
            s = s.resample("YE").mean()
            s.index = s.index.year
            frames.append(s.rename(name))
        except Exception as e:
            print(f"  Warning: {name}: {e}")

    global_df = pd.concat(frames, axis=1).reset_index()
    global_df.rename(columns={"index": "year"}, inplace=True)
    global_df = global_df[global_df["year"].between(START_YEAR, END_YEAR)].copy()
    global_df.to_csv(f"{OUTPUT_DIR}/fred_global.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/fred_global.csv  →  {len(global_df)} rows")
    return global_df


# ─── DISTRESS LABELS ─────────────────────────────────────────────────────────

def build_distress_labels() -> pd.DataFrame:
    """Build binary distress labels for all 188 countries × 26 years."""
    print(f"\n[3/3] Building distress labels for {len(COUNTRIES)} countries...")
    rows = []
    for country in COUNTRIES:
        distress_years = set()
        for start, end in DISTRESS_EPISODES.get(country, []):
            distress_years.update(range(start, end + 1))
        for year in range(START_YEAR, END_YEAR + 1):
            rows.append({
                "country"          : country,
                "year"             : year,
                "income_group"     : INCOME_GROUP.get(country, "UMI"),
                "distress"         : int(year in distress_years),
                "distress_next_1y" : int((year + 1) in distress_years),
                "distress_next_2y" : int(
                    (year + 1) in distress_years or (year + 2) in distress_years
                ),
            })

    labels_df = pd.DataFrame(rows)
    labels_df.to_csv(f"{OUTPUT_DIR}/distress_labels.csv", index=False)
    rate = labels_df["distress_next_2y"].mean()
    print(f"  ✓ {OUTPUT_DIR}/distress_labels.csv  →  {len(labels_df)} rows")
    print(f"  Distress rate (2y fwd): {rate:.1%}  "
          f"({int(rate * len(labels_df))}/{len(labels_df)} country-years)")
    return labels_df


# ─── MERGE ───────────────────────────────────────────────────────────────────

def merge_all(wb_df, fred_df, labels_df) -> pd.DataFrame:
    df = labels_df.merge(
        wb_df.drop(columns=["income_group"], errors="ignore"),
        on=["country", "year"], how="left"
    )
    df = df.merge(fred_df, on="year", how="left")

    df.sort_values(["country", "year"], inplace=True)
    macro_cols = list(WB_INDICATORS.keys()) + list(FRED_SERIES.keys())
    df[macro_cols] = (
        df.groupby("country")[macro_cols]
          .transform(lambda g: g.ffill().bfill())
    )

    min_req = int(len(macro_cols) * 0.5) + 1
    before  = len(df)
    df.dropna(subset=macro_cols, thresh=min_req, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(f"{OUTPUT_DIR}/panel_dataset.csv", index=False)
    print(f"\n✓ panel_dataset.csv  →  {df.shape}  "
          f"({before - len(df)} rows dropped for missing data)")
    print(f"  Countries: {df['country'].nunique()}  |  "
          f"Years: {df['year'].min()}–{df['year'].max()}")
    income_counts = df.drop_duplicates("country")["income_group"].value_counts()
    print(f"  Income groups: {income_counts.to_dict()}")
    return df


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    wb_df     = fetch_world_bank()
    fred_df   = fetch_fred()
    labels_df = build_distress_labels()
    panel_df  = merge_all(wb_df, fred_df, labels_df)
    print("\n✅ Step 1 complete. Run 02_news_sentiment.py next.\n")