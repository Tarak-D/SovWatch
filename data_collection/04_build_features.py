"""
=====================================================================
STEP 4 — Feature Engineering & Dataset Assembly (188 Countries)
=====================================================================
Merges: macro panel + GARCH features + sentiment features
Outputs: model-ready numpy arrays for LSTM training

Key differences vs 21-country version:
  - Income group encoded as 6 binary columns (one-hot)
  - Country-grouped split stratified by income group to ensure
    each income group appears in train, val, and test
  - ~4,880 sequences (vs ~350 in 21-country version)
  - Handles much larger share of countries with sparse coverage
    (many LI/FRG countries have significant data gaps → robust imputation)
=====================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import pickle
import os

RAW_DIR       = "data/raw"
SENTIMENT_DIR = "data/sentiment"
GARCH_DIR     = "data/garch"
FEATURES_DIR  = "data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

SEQUENCE_LEN = 5
TARGET_COL   = "distress_next_2y"

INCOME_GROUPS = ["HI", "UMI", "LMI", "LI", "OIL", "FRG"]

# ─── FEATURE LISTS ───────────────────────────────────────────────────────────

MACRO_FEATURES = [
    "gdp_growth", "inflation_cpi", "debt_to_gdp",
    "current_account", "fx_reserves_months", "unemployment",
    "gdp_per_capita", "exports_gdp", "fiscal_balance", "lending_rate",
    "gross_savings", "external_debt_gdp", "short_term_debt",
]

GLOBAL_FEATURES = [
    "us_fed_funds_rate", "vix_index", "us_10y_treasury", "dxy_dollar_index",
]

GARCH_FEATURES = [
    "mean_cond_vol", "max_cond_vol", "vol_of_vol",
    "pct_high_vol", "mean_spread", "spread_change",
    "persistence", "garch_asymmetry",
]

SENTIMENT_FEATURES = [
    "mean_distress_score", "negative_ratio", "positive_ratio",
    "sentiment_volatility", "log_article_count",
]

# Income group one-hot columns (added during engineer_features)
INCOME_FEATURES = [f"ig_{g.lower()}" for g in INCOME_GROUPS]

ALL_FEATURES = (MACRO_FEATURES + GLOBAL_FEATURES + GARCH_FEATURES
                + SENTIMENT_FEATURES + INCOME_FEATURES)


# ─── LOAD & MERGE ────────────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    panel  = pd.read_csv(f"{RAW_DIR}/panel_dataset.csv")
    garch  = pd.read_csv(f"{GARCH_DIR}/garch_features_annual.csv")
    sentim = pd.read_csv(f"{SENTIMENT_DIR}/sentiment_features.csv")

    df = panel.merge(garch,  on=["country", "year"], how="left")
    df = df.merge(sentim, on=["country", "year"], how="left")

    # Neutral fills
    df["mean_distress_score"]  = df["mean_distress_score"].fillna(1/3)
    df["negative_ratio"]       = df["negative_ratio"].fillna(1/3)
    df["positive_ratio"]       = df["positive_ratio"].fillna(1/3)
    df["sentiment_volatility"] = df["sentiment_volatility"].fillna(0.0)
    df["log_article_count"]    = df["log_article_count"].fillna(0.0)
    df["garch_asymmetry"]      = df["garch_asymmetry"].fillna(0.0)

    print(f"Merged dataset: {df.shape}")
    print(f"  Countries   : {df['country'].nunique()}")
    print(f"  Years       : {df['year'].min()}–{df['year'].max()}")
    print(f"  Distress    : {df[TARGET_COL].mean():.1%}")
    if "income_group" in df.columns:
        print(f"  Income groups: {df.groupby('income_group')['country'].nunique().to_dict()}")
    return df


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country", "year"]).copy()

    # Lag features
    for col in ["gdp_growth", "inflation_cpi", "debt_to_gdp",
                "mean_cond_vol", "external_debt_gdp", "fiscal_balance"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("country")[col].shift(1)
            df[f"{col}_lag2"] = df.groupby("country")[col].shift(2)

    # YoY changes
    for col in ["debt_to_gdp", "fx_reserves_months", "mean_spread",
                "external_debt_gdp", "short_term_debt"]:
        if col in df.columns:
            df[f"{col}_yoy"] = df.groupby("country")[col].diff()

    # Interaction terms
    if "debt_to_gdp" in df.columns and "mean_cond_vol" in df.columns:
        d = df["debt_to_gdp"] / (df["debt_to_gdp"].max() + 1e-9)
        v = df["mean_cond_vol"] / (df["mean_cond_vol"].max() + 1e-9)
        df["debt_vol_interaction"] = d * v

    if "negative_ratio" in df.columns and "vix_index" in df.columns:
        vn = df["vix_index"] / (df["vix_index"].max() + 1e-9)
        df["sentiment_vix_interaction"] = df["negative_ratio"] * vn

    if "external_debt_gdp" in df.columns and "short_term_debt" in df.columns:
        e = df["external_debt_gdp"] / (df["external_debt_gdp"].max() + 1e-9)
        s = df["short_term_debt"] / (df["short_term_debt"].max() + 1e-9)
        df["rollover_risk"] = e * s

    # Income group one-hot encoding (static per country but informative for model)
    if "income_group" in df.columns:
        for g in INCOME_GROUPS:
            df[f"ig_{g.lower()}"] = (df["income_group"] == g).astype(float)
    else:
        for g in INCOME_GROUPS:
            df[f"ig_{g.lower()}"] = 0.0

    return df


# ─── BUILD SEQUENCES ─────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, feature_cols: list) -> tuple:
    X_list, y_list, groups_list = [], [], []

    for country, grp in df.groupby("country"):
        grp = grp.sort_values("year").reset_index(drop=True)
        grp[feature_cols] = grp[feature_cols].apply(
            lambda col: col.fillna(col.median() if col.notna().any() else 0.0)
        )
        for i in range(SEQUENCE_LEN, len(grp)):
            window = grp.iloc[i - SEQUENCE_LEN : i][feature_cols].values
            label  = grp.iloc[i][TARGET_COL]
            if not np.isnan(window).any() and not np.isnan(label):
                X_list.append(window)
                y_list.append(int(label))
                groups_list.append(country)

    X      = np.array(X_list,    dtype=np.float32)
    y      = np.array(y_list,    dtype=np.float32)
    groups = np.array(groups_list)

    print(f"\nSequences: X={X.shape}  y={y.shape}")
    print(f"  Positive rate : {y.mean():.1%}")
    print(f"  Unique countries in sequences: {np.unique(groups).shape[0]}")
    return X, y, groups


# ─── STRATIFIED COUNTRY-GROUPED SPLIT ────────────────────────────────────────

def split_data(X, y, groups, df: pd.DataFrame):
    """
    Country-grouped split stratified by income group.
    Ensures each income tier appears in train, val, and test.
    This matters for generalization: a model trained only on LI countries
    shouldn't be tested on HI countries and vice versa.
    """
    # Map country → income group
    ig_map = df.drop_duplicates("country").set_index("country")["income_group"].to_dict() \
               if "income_group" in df.columns else {}

    # Stratified: split within each income group, then combine
    train_idx_all, val_idx_all, test_idx_all = [], [], []

    for ig in INCOME_GROUPS + ["unknown"]:
        if ig == "unknown":
            mask = np.array([ig_map.get(g, "unknown") not in INCOME_GROUPS
                             for g in groups])
        else:
            mask = np.array([ig_map.get(g, "unknown") == ig for g in groups])

        idx = np.where(mask)[0]
        if len(idx) < 10:
            train_idx_all.extend(idx.tolist())
            continue

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tv_idx, te_idx = next(gss.split(X[idx], y[idx], groups[idx]))
        train_val = idx[tv_idx]
        test_idx_all.extend(idx[te_idx].tolist())

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.187, random_state=0)
        tr_idx, v_idx = next(gss2.split(X[train_val], y[train_val], groups[train_val]))
        train_idx_all.extend(train_val[tr_idx].tolist())
        val_idx_all.extend(train_val[v_idx].tolist())

    ti = np.array(train_idx_all)
    vi = np.array(val_idx_all)
    te = np.array(test_idx_all)

    print(f"\nStratified country-grouped split:")
    print(f"  Train : {len(ti):>5}  ({y[ti].mean():.1%} positive)")
    print(f"  Val   : {len(vi):>5}  ({y[vi].mean():.1%} positive)")
    print(f"  Test  : {len(te):>5}  ({y[te].mean():.1%} positive)")
    return X[ti], X[vi], X[te], y[ti], y[vi], y[te]


# ─── SCALING ─────────────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    N, T, F = X_train.shape
    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N, T, F)
    X_val_s   = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_test_s  = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
    with open(f"{FEATURES_DIR}/scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    return X_train_s, X_val_s, X_test_s, scaler


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_merge()
    df = engineer_features(df)

    lag_cols  = [c for c in df.columns if "_lag" in c]
    yoy_cols  = [c for c in df.columns if "_yoy" in c]
    int_cols  = [c for c in df.columns
                 if "_interaction" in c or c in ("rollover_risk",)]

    feature_cols = [c for c in ALL_FEATURES + lag_cols + yoy_cols + int_cols
                    if c in df.columns]
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    print(f"\nTotal features: {len(feature_cols)}")

    X, y, groups       = build_sequences(df, feature_cols)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, groups, df)
    X_tr, X_v, X_te, _ = scale_features(X_tr, X_v, X_te)

    for name, arr in [("X_train", X_tr), ("X_val", X_v), ("X_test", X_te),
                      ("y_train", y_tr), ("y_val", y_v), ("y_test", y_te)]:
        np.save(f"{FEATURES_DIR}/{name}.npy", arr)

    with open(f"{FEATURES_DIR}/feature_cols.pkl", "wb") as fh:
        pickle.dump(feature_cols, fh)

    print(f"\n✓ Arrays → {FEATURES_DIR}/")
    print(f"  X_train={X_tr.shape}  X_val={X_v.shape}  X_test={X_te.shape}")
    print("\n✅ Step 4 complete. Run 05_lstm_model.py next.\n")