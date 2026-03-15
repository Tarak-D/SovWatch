"""
=====================================================================
STEP 2 — Global News Sentiment Pipeline (GDELT + FinBERT)
=====================================================================
Coverage : 188 countries (2000–2025)
Sources  : GDELT DOC 2.0 API + ProsusAI/finbert

Notes:
  - GDELT free tier: ~250 requests/day. For full 188-country coverage
    use GDELT BigQuery (1 TB/month free via GCP):
    https://console.cloud.google.com/bigquery
  - FinBERT downloads ~440 MB on first run (cached by HuggingFace)

Install:
  pip install pandas requests transformers torch tqdm
=====================================================================
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# ─── CONFIG ──────────────────────────────────────────────────────────────────

START_YEAR = 2000
END_YEAR   = 2025

SENTIMENT_DIR = "data/sentiment"
os.makedirs(SENTIMENT_DIR, exist_ok=True)

# All 188 countries — common English news names for GDELT search
COUNTRY_NAMES = {
    "AFG": ["Afghanistan"],
    "ALB": ["Albania"],
    "ARE": ["UAE", "United Arab Emirates"],
    "ARG": ["Argentina"],
    "ARM": ["Armenia"],
    "ATG": ["Antigua"],
    "AUS": ["Australia"],
    "AUT": ["Austria"],
    "AZE": ["Azerbaijan"],
    "BDI": ["Burundi"],
    "BEL": ["Belgium"],
    "BEN": ["Benin"],
    "BFA": ["Burkina Faso"],
    "BGD": ["Bangladesh"],
    "BGR": ["Bulgaria"],
    "BHR": ["Bahrain"],
    "BIH": ["Bosnia", "Bosnia and Herzegovina"],
    "BLR": ["Belarus"],
    "BLZ": ["Belize"],
    "BOL": ["Bolivia"],
    "BRA": ["Brazil"],
    "BRB": ["Barbados"],
    "BRN": ["Brunei"],
    "BTN": ["Bhutan"],
    "BWA": ["Botswana"],
    "CAF": ["Central African Republic", "CAR"],
    "CAN": ["Canada"],
    "CHE": ["Switzerland"],
    "CHL": ["Chile"],
    "CHN": ["China"],
    "CIV": ["Ivory Coast", "Cote d'Ivoire"],
    "CMR": ["Cameroon"],
    "COD": ["Congo DRC", "Democratic Republic of Congo"],
    "COG": ["Republic of Congo", "Congo Brazzaville"],
    "COL": ["Colombia"],
    "COM": ["Comoros"],
    "CPV": ["Cape Verde"],
    "CRI": ["Costa Rica"],
    "CUB": ["Cuba"],
    "CYP": ["Cyprus"],
    "CZE": ["Czech Republic", "Czechia"],
    "DEU": ["Germany"],
    "DJI": ["Djibouti"],
    "DMA": ["Dominica"],
    "DNK": ["Denmark"],
    "DOM": ["Dominican Republic"],
    "DZA": ["Algeria"],
    "ECU": ["Ecuador"],
    "EGY": ["Egypt"],
    "ERI": ["Eritrea"],
    "ESP": ["Spain"],
    "EST": ["Estonia"],
    "ETH": ["Ethiopia"],
    "FIN": ["Finland"],
    "FJI": ["Fiji"],
    "FRA": ["France"],
    "FSM": ["Micronesia"],
    "GAB": ["Gabon"],
    "GBR": ["United Kingdom", "UK"],
    "GEO": ["Georgia"],
    "GHA": ["Ghana"],
    "GIN": ["Guinea"],
    "GMB": ["Gambia"],
    "GNB": ["Guinea-Bissau"],
    "GNQ": ["Equatorial Guinea"],
    "GRC": ["Greece"],
    "GRD": ["Grenada"],
    "GTM": ["Guatemala"],
    "GUY": ["Guyana"],
    "HKG": ["Hong Kong"],
    "HND": ["Honduras"],
    "HRV": ["Croatia"],
    "HTI": ["Haiti"],
    "HUN": ["Hungary"],
    "IDN": ["Indonesia"],
    "IND": ["India"],
    "IRL": ["Ireland"],
    "IRN": ["Iran"],
    "IRQ": ["Iraq"],
    "ISL": ["Iceland"],
    "ISR": ["Israel"],
    "ITA": ["Italy"],
    "JAM": ["Jamaica"],
    "JOR": ["Jordan"],
    "JPN": ["Japan"],
    "KAZ": ["Kazakhstan"],
    "KEN": ["Kenya"],
    "KGZ": ["Kyrgyzstan"],
    "KHM": ["Cambodia"],
    "KIR": ["Kiribati"],
    "KNA": ["Saint Kitts", "St Kitts"],
    "KOR": ["South Korea"],
    "KWT": ["Kuwait"],
    "LAO": ["Laos"],
    "LBN": ["Lebanon"],
    "LBR": ["Liberia"],
    "LBY": ["Libya"],
    "LCA": ["Saint Lucia", "St Lucia"],
    "LKA": ["Sri Lanka"],
    "LSO": ["Lesotho"],
    "LTU": ["Lithuania"],
    "LUX": ["Luxembourg"],
    "LVA": ["Latvia"],
    "MAR": ["Morocco"],
    "MDA": ["Moldova"],
    "MDG": ["Madagascar"],
    "MDV": ["Maldives"],
    "MEX": ["Mexico"],
    "MHL": ["Marshall Islands"],
    "MKD": ["North Macedonia", "Macedonia"],
    "MLI": ["Mali"],
    "MLT": ["Malta"],
    "MMR": ["Myanmar", "Burma"],
    "MNE": ["Montenegro"],
    "MNG": ["Mongolia"],
    "MOZ": ["Mozambique"],
    "MRT": ["Mauritania"],
    "MUS": ["Mauritius"],
    "MWI": ["Malawi"],
    "MYS": ["Malaysia"],
    "NAM": ["Namibia"],
    "NER": ["Niger"],
    "NGA": ["Nigeria"],
    "NIC": ["Nicaragua"],
    "NLD": ["Netherlands"],
    "NOR": ["Norway"],
    "NPL": ["Nepal"],
    "NRU": ["Nauru"],
    "NZL": ["New Zealand"],
    "OMN": ["Oman"],
    "PAK": ["Pakistan"],
    "PAN": ["Panama"],
    "PER": ["Peru"],
    "PHL": ["Philippines"],
    "PLW": ["Palau"],
    "PNG": ["Papua New Guinea"],
    "POL": ["Poland"],
    "PRT": ["Portugal"],
    "PRY": ["Paraguay"],
    "PSE": ["Palestine", "Palestinian"],
    "QAT": ["Qatar"],
    "ROU": ["Romania"],
    "RUS": ["Russia"],
    "RWA": ["Rwanda"],
    "SAU": ["Saudi Arabia"],
    "SDN": ["Sudan"],
    "SEN": ["Senegal"],
    "SGP": ["Singapore"],
    "SLB": ["Solomon Islands"],
    "SLE": ["Sierra Leone"],
    "SLV": ["El Salvador"],
    "SOM": ["Somalia"],
    "SRB": ["Serbia"],
    "SSD": ["South Sudan"],
    "STP": ["Sao Tome"],
    "SUR": ["Suriname"],
    "SVK": ["Slovakia"],
    "SVN": ["Slovenia"],
    "SWE": ["Sweden"],
    "SWZ": ["Eswatini", "Swaziland"],
    "SYC": ["Seychelles"],
    "SYR": ["Syria"],
    "TCD": ["Chad"],
    "TGO": ["Togo"],
    "THA": ["Thailand"],
    "TJK": ["Tajikistan"],
    "TKM": ["Turkmenistan"],
    "TLS": ["Timor-Leste", "East Timor"],
    "TON": ["Tonga"],
    "TTO": ["Trinidad and Tobago"],
    "TUN": ["Tunisia"],
    "TWN": ["Taiwan"],
    "TUR": ["Turkey", "Türkiye"],
    "TZA": ["Tanzania"],
    "UGA": ["Uganda"],
    "UKR": ["Ukraine"],
    "URY": ["Uruguay"],
    "USA": ["United States", "US"],
    "UZB": ["Uzbekistan"],
    "VCT": ["Saint Vincent"],
    "VEN": ["Venezuela"],
    "VNM": ["Vietnam"],
    "VUT": ["Vanuatu"],
    "WSM": ["Samoa"],
    "XKX": ["Kosovo"],
    "YEM": ["Yemen"],
    "ZAF": ["South Africa"],
    "ZMB": ["Zambia"],
    "ZWE": ["Zimbabwe"],
}

DEBT_KEYWORDS = [
    "debt", "default", "IMF", "bond", "sovereign", "fiscal",
    "bailout", "restructur", "credit rating", "downgrad",
    "currency crisis", "inflation", "deficit", "GDP",
    "economic crisis", "financial crisis", "loan", "repayment",
    "forex", "reserves", "devaluation", "austerity",
]

SEARCH_TERMS = ["debt", "IMF", "default", "bond", "fiscal crisis"]


# ─── GDELT DOC API ───────────────────────────────────────────────────────────

def fetch_gdelt_headlines(iso3: str, year: int,
                          max_articles: int = 200,
                          retries: int = 3) -> list[dict]:
    """Fetch debt-related headlines for one country-year from GDELT."""
    names    = COUNTRY_NAMES.get(iso3, [iso3])
    headlines = []

    for name in names[:2]:  # use first 2 name variants to stay within rate limit
        for term in SEARCH_TERMS:
            query = f'"{name}" "{term}"'
            url = (
                "https://api.gdeltproject.org/api/v2/doc/doc"
                f"?query={requests.utils.quote(query)}"
                "&mode=artlist&maxrecords=25"
                f"&startdatetime={year}0101000000"
                f"&enddatetime={year}1231235959"
                "&format=json&sourcelang=english"
            )
            for attempt in range(retries):
                try:
                    r = requests.get(url, timeout=20)
                    if r.status_code == 200:
                        for a in r.json().get("articles", []):
                            title = a.get("title", "")
                            if title and any(kw.lower() in title.lower()
                                             for kw in DEBT_KEYWORDS):
                                headlines.append({
                                    "country": iso3,
                                    "year"   : year,
                                    "title"  : title,
                                    "url"    : a.get("url", ""),
                                    "date"   : a.get("seendate", ""),
                                })
                    break
                except requests.exceptions.Timeout:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                except Exception:
                    break
            time.sleep(0.5)
            if len(headlines) >= max_articles:
                break

    return headlines[:max_articles]


def collect_all_headlines(countries: list, years) -> pd.DataFrame:
    print(f"\n[1/2] Collecting GDELT headlines for {len(countries)} countries...")
    all_rows = []
    for country in tqdm(countries, desc="Countries"):
        for year in years:
            all_rows.extend(fetch_gdelt_headlines(country, year))
            time.sleep(0.2)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["title"])
    df.to_csv(f"{SENTIMENT_DIR}/raw_headlines.csv", index=False)
    print(f"  ✓ {len(df):,} unique headlines  →  {SENTIMENT_DIR}/raw_headlines.csv")
    return df


# ─── FINBERT SCORING ─────────────────────────────────────────────────────────

class FinBERTScorer:
    LABEL_ORDER = ["positive", "negative", "neutral"]

    def __init__(self, batch_size=32):
        print("\nLoading FinBERT (~440 MB on first run)...")
        self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        self.model     = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"  ✓ FinBERT on {self.device}")

    def score(self, texts: list[str]) -> list[dict]:
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc="Scoring", unit="batch"):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.model(**inputs).logits, dim=-1).cpu().numpy()
            for p in probs:
                results.append({
                    "prob_positive" : float(p[0]),
                    "prob_negative" : float(p[1]),
                    "prob_neutral"  : float(p[2]),
                    "sentiment"     : self.LABEL_ORDER[p.argmax()],
                    "distress_score": float(p[1]),
                })
        return results


def score_headlines(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2/2] Scoring {len(df):,} headlines with FinBERT...")
    scorer = FinBERTScorer(batch_size=32)
    scores = scorer.score(df["title"].tolist())
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)


# ─── AGGREGATE TO COUNTRY-YEAR ────────────────────────────────────────────────

def aggregate_sentiment(scored_df: pd.DataFrame,
                        all_countries: list,
                        years: range) -> pd.DataFrame:
    """
    Aggregate to country-year. Countries with no headlines get neutral priors.
    This is critical for 188-country coverage — many small countries have
    minimal English-language financial news coverage.
    """
    agg = scored_df.groupby(["country", "year"]).agg(
        mean_distress_score  = ("distress_score", "mean"),
        negative_ratio       = ("sentiment", lambda x: (x == "negative").mean()),
        positive_ratio       = ("sentiment", lambda x: (x == "positive").mean()),
        sentiment_volatility = ("distress_score", "std"),
        article_count        = ("title", "count"),
    ).reset_index()
    agg["log_article_count"]    = np.log1p(agg["article_count"])
    agg["sentiment_volatility"] = agg["sentiment_volatility"].fillna(0.0)

    # Build complete grid and fill missing with neutral priors
    grid = pd.MultiIndex.from_product(
        [all_countries, list(years)], names=["country", "year"]
    ).to_frame(index=False)
    agg = grid.merge(agg, on=["country", "year"], how="left")
    agg["mean_distress_score"]  = agg["mean_distress_score"].fillna(1/3)
    agg["negative_ratio"]       = agg["negative_ratio"].fillna(1/3)
    agg["positive_ratio"]       = agg["positive_ratio"].fillna(1/3)
    agg["sentiment_volatility"] = agg["sentiment_volatility"].fillna(0.0)
    agg["log_article_count"]    = agg["log_article_count"].fillna(0.0)
    agg["article_count"]        = agg["article_count"].fillna(0)

    agg.to_csv(f"{SENTIMENT_DIR}/sentiment_features.csv", index=False)
    print(f"\n✓ {SENTIMENT_DIR}/sentiment_features.csv  →  {agg.shape}")
    covered = (agg["article_count"] > 0).mean()
    print(f"  Country-years with real data: {covered:.1%}  "
          f"(rest filled with neutral priors)")
    return agg


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    COUNTRIES = [
"ARG","GHA","PAK","LKA","TUR","EGY","KEN","NGA","ZMB","ETH","BGD",
"UKR","BLR","TUN","ECU","VEN","JAM","BLZ","MOZ","MWI","IND",
"AUS","AUT","BEL","CAN","CHE","CZE","DEU","DNK","ESP","EST","FIN","FRA",
"GBR","GRC","HKG","HRV","HUN","IRL","ISL","ISR","ITA","JPN","KOR","LUX",
"LVA","MLT","NLD","NOR","NZL","POL","PRT","SGP","SVK","SVN","SWE","TWN",
"USA","BRA","CHN","COL","CRI","DOM","GTM","IDN","IRN","JOR","KAZ","KEN",
"MEX","MYS","PAN","PER","PHL","ROU","RUS","SRB","THA","TUR","URY","ZAF"
]
   # or paste COUNTRIES list here
    years = range(START_YEAR, END_YEAR + 1)

    headlines_df = collect_all_headlines(COUNTRIES, years)
    scored_df    = score_headlines(headlines_df)
    scored_df.to_csv(f"{SENTIMENT_DIR}/scored_headlines.csv", index=False)

    sentiment_features = aggregate_sentiment(scored_df, COUNTRIES, years)
    print("\n✅ Step 2 complete. Run 03_garch_volatility.R next.\n")