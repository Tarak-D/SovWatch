"""
=====================================================================
STEP 6 — Baseline Models & Comparison Table (188 Countries)
=====================================================================
Baselines:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. LSTM-only ablation   (no GARCH, no sentiment)
  5. GARCH threshold rule
  6. Hybrid BiLSTM-GARCH-Sentiment (ours) ← auto-read from config

Extra output vs 21-country version:
  - Per-income-group breakdown (HI / UMI / LMI / LI / OIL / FRG)
  - Tells you if the model works better for emerging markets vs developed
=====================================================================
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, brier_score_loss, classification_report,
)
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

FEATURES_DIR = "data/features"
MODELS_DIR   = "models"
RAW_DIR      = "data/raw"

INCOME_GROUPS = ["HI", "UMI", "LMI", "LI", "OIL", "FRG"]


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_data():
    X_train = np.load(f"{FEATURES_DIR}/X_train.npy")
    X_test  = np.load(f"{FEATURES_DIR}/X_test.npy")
    y_train = np.load(f"{FEATURES_DIR}/y_train.npy").astype(int)
    y_test  = np.load(f"{FEATURES_DIR}/y_test.npy").astype(int)

    with open(f"{FEATURES_DIR}/feature_cols.pkl", "rb") as fh:
        feature_cols = pickle.load(fh)

    X_train_flat = X_train[:, -1, :]
    X_test_flat  = X_test[:,  -1, :]
    X_train_full = X_train.reshape(len(X_train), -1)
    X_test_full  = X_test.reshape(len(X_test),   -1)

    print(f"Test set: {len(X_test)} samples  |  "
          f"Positive rate: {y_test.mean():.1%}")
    return {
        "flat" : (X_train_flat, X_test_flat, y_train, y_test, feature_cols),
        "full" : (X_train_full, X_test_full, y_train, y_test, feature_cols),
        "seq"  : (X_train,      X_test,      y_train, y_test, feature_cols),
    }


def load_income_group_labels() -> dict:
    """Load income group for each country for subgroup analysis."""
    panel_path = f"{RAW_DIR}/panel_dataset.csv"
    if os.path.exists(panel_path):
        df = pd.read_csv(panel_path)
        if "income_group" in df.columns:
            return df.drop_duplicates("country").set_index("country")["income_group"].to_dict()
    return {}


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def evaluate_model(name, y_true, y_prob, threshold=0.40) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    result = {
        "model" : name,
        "AUROC" : roc_auc_score(y_true, y_prob),
        "AUPRC" : average_precision_score(y_true, y_prob),
        "F1"    : f1_score(y_true, y_pred, zero_division=0),
        "Brier" : brier_score_loss(y_true, y_prob),
    }
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"  AUROC={result['AUROC']:.4f}  AUPRC={result['AUPRC']:.4f}  "
          f"F1={result['F1']:.4f}  Brier={result['Brier']:.4f}")
    print(classification_report(y_true, y_pred,
          target_names=["No Distress","Distress"], zero_division=0))
    return result


# ─── BASELINES ───────────────────────────────────────────────────────────────

def run_logistic(data):
    X_tr, X_te, y_tr, y_te, _ = data["flat"]
    clf = LogisticRegression(class_weight="balanced", C=0.1,
                             max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    return evaluate_model("Logistic Regression",
                          y_te, clf.predict_proba(X_te)[:, 1])


def run_random_forest(data):
    X_tr, X_te, y_tr, y_te, _ = data["full"]
    clf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                 class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return evaluate_model("Random Forest",
                          y_te, clf.predict_proba(X_te)[:, 1])


def run_xgboost(data):
    X_tr, X_te, y_tr, y_te, _ = data["full"]
    clf = xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        eval_metric      = "auc",
        random_state     = 42,
        verbosity        = 0,
    )
    clf.fit(X_tr, y_tr)
    return evaluate_model("XGBoost",
                          y_te, clf.predict_proba(X_te)[:, 1])


class _SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def run_lstm_ablation(data):
    """BiLSTM without GARCH or sentiment — ablation to prove those add value."""
    X_tr, X_te, y_tr, y_te, _ = data["seq"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SimpleModel(nn.Module):
        def __init__(self, n_feat):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, 128, 2, batch_first=True,
                                bidirectional=True, dropout=0.3)
            self.head = nn.Linear(256, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    tr_ld = DataLoader(_SimpleDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    te_ld = DataLoader(_SimpleDataset(X_te, y_te), batch_size=64)

    model = SimpleModel(X_tr.shape[2]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pw    = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                         dtype=torch.float32).to(device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_loss, best_state = float("inf"), None
    for _ in range(30):
        model.train()
        for Xb, yb in tr_ld:
            opt.zero_grad()
            crit(model(Xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(Xb.to(device)), yb.to(device)).item()
                     for Xb, yb in te_ld) / len(te_ld)
        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    probs = np.concatenate([
        torch.sigmoid(model(Xb.to(device))).cpu().detach().numpy()
        for Xb, _ in te_ld
    ])
    return evaluate_model("LSTM-only (ablation)", y_te, probs)


def run_garch_threshold(data):
    X_tr, X_te, y_tr, y_te, feature_cols = data["flat"]
    try:
        idx = list(feature_cols).index("mean_cond_vol")
        thr = np.percentile(X_tr[:, idx], 75)
        prb = (X_te[:, idx] >= thr).astype(float)
        prb += np.random.default_rng(42).uniform(0, 0.01, len(prb))
        return evaluate_model("GARCH Threshold Rule", y_te, prb)
    except (ValueError, IndexError):
        print("  Skipping GARCH baseline: mean_cond_vol not in features")
        return None


def load_hybrid_results() -> dict:
    path = f"{MODELS_DIR}/model_config.json"
    if os.path.exists(path):
        with open(path) as fh:
            cfg = json.load(fh)
        return {
            "model" : "Hybrid BiLSTM-GARCH-Sentiment (Ours)",
            "AUROC" : cfg.get("test_auroc", 0.0),
            "AUPRC" : cfg.get("test_auprc", 0.0),
            "F1"    : cfg.get("test_f1",    0.0),
            "Brier" : cfg.get("test_brier", 0.0),
        }
    print("  ⚠ Run 05_lstm_model.py first to populate model_config.json")
    return {"model": "Hybrid BiLSTM-GARCH-Sentiment (Ours)",
            "AUROC": 0.0, "AUPRC": 0.0, "F1": 0.0, "Brier": 0.0}


# ─── PER-INCOME-GROUP BREAKDOWN ──────────────────────────────────────────────

def income_group_breakdown(y_true, y_prob, feature_cols, X_test):
    """
    Break down test performance by income group.
    The income group columns are one-hot encoded in the feature set.
    """
    print("\n── Per-income-group breakdown ──")
    rows = []
    for ig in INCOME_GROUPS:
        col = f"ig_{ig.lower()}"
        if col not in feature_cols:
            continue
        idx = feature_cols.index(col)
        mask = X_test[:, -1, idx] > 0.5   # last timestep, binary flag
        if mask.sum() < 5:
            continue
        try:
            auroc = roc_auc_score(y_true[mask], y_prob[mask])
            auprc = average_precision_score(y_true[mask], y_prob[mask])
            n_pos = y_true[mask].sum()
            rows.append({
                "Income Group": ig,
                "N Samples"   : int(mask.sum()),
                "N Distress"  : int(n_pos),
                "AUROC"       : round(auroc, 4),
                "AUPRC"       : round(auprc, 4),
            })
        except Exception:
            pass

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        df.to_csv(f"{MODELS_DIR}/income_group_breakdown.csv", index=False)
        print(f"\n  Saved → {MODELS_DIR}/income_group_breakdown.csv")


# ─── COMPARISON TABLE & CHART ─────────────────────────────────────────────────

def build_comparison_table(results: list) -> pd.DataFrame:
    rows = [r for r in results if r is not None]
    df   = pd.DataFrame(rows).set_index("model").sort_values("AUROC", ascending=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE — 188 Countries Global Dataset")
    print("=" * 70)
    print(df.round(4).to_string())

    # LaTeX booktabs table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model comparison on the global sovereign debt distress dataset "
        r"(188 countries, 2000--2025, country-grouped test split)}",
        r"\label{tab:global_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & AUROC & AUPRC & F1 & Brier \\",
        r"\midrule",
    ]
    for name, row in df.iterrows():
        bold = r"\textbf{(ours)}" if "Ours" in name else ""
        lines.append(f"{name} {bold} & "
                     f"{row['AUROC']:.4f} & {row['AUPRC']:.4f} & "
                     f"{row['F1']:.4f} & {row['Brier']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(f"{MODELS_DIR}/comparison_table.tex", "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n  LaTeX → {MODELS_DIR}/comparison_table.tex")

    # Horizontal bar chart per metric
    metrics = ["AUROC", "AUPRC", "F1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(df) * 0.6 + 1)))
    colors = ["#185FA5", "#D85A30", "#1D9E75"]
    for ax, m, c in zip(axes, metrics, colors):
        vals   = df[m].values
        labels = [lb[:35] for lb in df.index.tolist()]  # truncate long names
        bars   = ax.barh(labels, vals, color=c, alpha=0.85)
        ax.set_xlim(0, 1.05)
        ax.axvline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_title(m, fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
    plt.suptitle("Global Sovereign Debt Model Comparison (188 Countries)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/model_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart  → {MODELS_DIR}/model_comparison.png")
    return df


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()

    results = [
        run_logistic(data),
        run_random_forest(data),
        run_xgboost(data),
        run_lstm_ablation(data),
        run_garch_threshold(data),
        load_hybrid_results(),
    ]

    comparison = build_comparison_table(results)

    # Per-income-group breakdown for the best model (Hybrid)
    X_test, y_test = data["seq"][1], data["seq"][3]
    feature_cols   = list(data["seq"][4])
    hybrid_path    = f"{MODELS_DIR}/model_config.json"
    if os.path.exists(hybrid_path):
        try:
            from models.lstm_model import SovereignDistressLSTM, DistressDataset
            cfg   = json.load(open(hybrid_path))
            garch_names = ["mean_cond_vol","max_cond_vol","vol_of_vol",
                           "pct_high_vol","mean_spread","spread_change",
                           "persistence","garch_asymmetry"]
            garch_idx = [i for i, f in enumerate(feature_cols) if f in garch_names]
            model = SovereignDistressLSTM(
                cfg["n_features"], max(cfg["n_garch_feats"], 1)
            )
            model.load_state_dict(torch.load(f"{MODELS_DIR}/best_model.pt",
                                             weights_only=True))
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            ld = DataLoader(DistressDataset(X_test, y_test, garch_idx),
                            batch_size=64)
            probs = []
            with torch.no_grad():
                for Xb, gb, _ in ld:
                    probs.append(torch.sigmoid(
                        model(Xb.to(device), gb.to(device))[0]
                    ).cpu().numpy())
            probs = np.concatenate(probs)
            income_group_breakdown(y_test, probs, feature_cols, X_test)
        except Exception as e:
            print(f"  Could not run income-group breakdown: {e}")

    print("\n✅ Step 6 complete. All outputs in models/\n")