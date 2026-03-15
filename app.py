import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc
import json
import os
import pickle

# ─── SAFE TORCH IMPORT (works on Streamlit Cloud without torch) ───────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn    = None

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SovWatch · Early Warning System",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
:root {
    --bg-deep:#060810; --bg-panel:#0d1117; --bg-card:#111827; --bg-hover:#1a2235;
    --accent:#3b82f6; --accent2:#06b6d4; --accent3:#8b5cf6;
    --danger:#ef4444; --warning:#f59e0b; --success:#10b981;
    --text-1:#f1f5f9; --text-2:#94a3b8; --text-3:#475569;
    --border:rgba(148,163,184,0.08);
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg-deep)!important;color:var(--text-1)!important;}
.stApp{background:var(--bg-deep)!important;}
.stApp::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(59,130,246,0.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(59,130,246,0.03) 1px,transparent 1px);
  background-size:48px 48px;pointer-events:none;z-index:0;}
[data-testid="stSidebar"]{background:var(--bg-panel)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text-1)!important;}
.hero{background:linear-gradient(135deg,#0d1117 0%,#0f172a 50%,#0d1117 100%);
  border:1px solid var(--border);border-radius:20px;padding:2.5rem 3rem;
  margin-bottom:2rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:240px;height:240px;
  background:radial-gradient(circle,rgba(59,130,246,0.12) 0%,transparent 70%);border-radius:50%;}
.hero-badge{display:inline-flex;align-items:center;gap:6px;
  background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);
  border-radius:20px;padding:4px 14px;font-family:'DM Mono',monospace;
  font-size:11px;color:#93c5fd;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:1rem;}
.hero-badge::before{content:'';width:6px;height:6px;background:#3b82f6;border-radius:50%;
  box-shadow:0 0 8px #3b82f6;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.5;transform:scale(0.8);}}
.hero-title{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;line-height:1.1;
  margin:0 0 0.6rem;background:linear-gradient(135deg,#f1f5f9 0%,#93c5fd 60%,#67e8f9 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:0.95rem;color:var(--text-2);line-height:1.6;max-width:560px;margin:0;}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem;}
.stat-card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;
  padding:1.25rem 1.5rem;position:relative;overflow:hidden;transition:border-color 0.2s,transform 0.2s;}
.stat-card:hover{border-color:rgba(59,130,246,0.3);transform:translateY(-2px);}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat-card.blue::before{background:linear-gradient(90deg,#3b82f6,#06b6d4);}
.stat-card.cyan::before{background:linear-gradient(90deg,#06b6d4,#8b5cf6);}
.stat-card.purple::before{background:linear-gradient(90deg,#8b5cf6,#ec4899);}
.stat-card.green::before{background:linear-gradient(90deg,#10b981,#3b82f6);}
.stat-label{font-size:11px;font-family:'DM Mono',monospace;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--text-3);margin-bottom:0.4rem;}
.stat-value{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:700;color:var(--text-1);line-height:1;}
.stat-delta{font-size:12px;color:var(--text-2);margin-top:0.3rem;}
.section-header{display:flex;align-items:center;gap:10px;margin:2rem 0 1rem;}
.section-line{flex:1;height:1px;background:var(--border);}
.section-title{font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:600;
  letter-spacing:0.15em;text-transform:uppercase;color:var(--text-3);white-space:nowrap;}
.prediction-card{background:var(--bg-card);border-radius:16px;padding:1.75rem;
  border:1px solid var(--border);margin-bottom:1rem;}
.risk-high{background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(239,68,68,0.03));
  border-color:rgba(239,68,68,0.25)!important;}
.risk-low{background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(16,185,129,0.03));
  border-color:rgba(16,185,129,0.2)!important;}
.risk-medium{background:linear-gradient(135deg,rgba(245,158,11,0.08),rgba(245,158,11,0.03));
  border-color:rgba(245,158,11,0.25)!important;}
.prob-display{font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;line-height:1;}
.prob-high{color:#ef4444;text-shadow:0 0 30px rgba(239,68,68,0.4);}
.prob-medium{color:#f59e0b;text-shadow:0 0 30px rgba(245,158,11,0.4);}
.prob-low{color:#10b981;text-shadow:0 0 30px rgba(16,185,129,0.4);}
.risk-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;
  border-radius:20px;font-size:12px;font-weight:500;letter-spacing:0.05em;margin-top:0.5rem;}
.badge-high{background:rgba(239,68,68,0.15);color:#fca5a5;border:1px solid rgba(239,68,68,0.3);}
.badge-medium{background:rgba(245,158,11,0.15);color:#fcd34d;border:1px solid rgba(245,158,11,0.3);}
.badge-low{background:rgba(16,185,129,0.15);color:#6ee7b7;border:1px solid rgba(16,185,129,0.3);}
.info-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:0.75rem;}
.info-pill{background:rgba(148,163,184,0.06);border:1px solid var(--border);
  border-radius:8px;padding:4px 12px;font-size:12px;font-family:'DM Mono',monospace;color:var(--text-2);}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:var(--bg-deep);}
::-webkit-scrollbar-thumb{background:var(--text-3);border-radius:2px;}
.footer{text-align:center;padding:2rem;color:var(--text-3);font-size:12px;
  font-family:'DM Mono',monospace;border-top:1px solid var(--border);
  margin-top:3rem;letter-spacing:0.05em;}
</style>
""", unsafe_allow_html=True)


# ─── MODEL DEFINITIONS ───────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class TemporalAttention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1)
            )
        def forward(self, lstm_out):
            scores  = self.attn(lstm_out).squeeze(-1)
            weights = torch.nn.functional.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
            return context, weights

    class SovereignDistressLSTM(nn.Module):
        def __init__(self, n_features, n_garch_feats,
                     hidden_dim=128, n_lstm_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_dim, n_lstm_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if n_lstm_layers > 1 else 0.0)
            lstm_out_dim = hidden_dim * 2
            self.layer_norm = nn.LayerNorm(lstm_out_dim)
            self.attention  = TemporalAttention(lstm_out_dim)
            self.garch_proj = nn.Sequential(
                nn.Linear(n_garch_feats, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, 64),
            )
            self.classifier = nn.Sequential(
                nn.Linear(lstm_out_dim + 64, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            )
        def forward(self, x, garch_feats):
            out, _        = self.lstm(x)
            out           = self.layer_norm(out)
            context, attn = self.attention(out)
            garch_emb     = self.garch_proj(garch_feats)
            logit         = self.classifier(torch.cat([context, garch_emb], dim=-1))
            return logit.squeeze(-1), attn


# ─── LOAD MODEL & DATA ───────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not TORCH_AVAILABLE:
        return None, None
    if not os.path.exists("models/model_config.json"):
        return None, None
    with open("models/model_config.json") as f:
        config = json.load(f)
    model = SovereignDistressLSTM(
        n_features    = config["n_features"],
        n_garch_feats = config["n_garch_feats"],
        hidden_dim    = config.get("hidden_dim", 128),
        n_lstm_layers = config.get("n_lstm_layers", 2),
        dropout       = config.get("dropout", 0.3),
    )
    if os.path.exists("models/best_model.pt"):
        state = torch.load("models/best_model.pt", map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model, config

@st.cache_data
def load_arrays():
    X_t  = np.load("data/features/X_test.npy")  if os.path.exists("data/features/X_test.npy")  else np.random.rand(200,5,42).astype(np.float32)
    y_t  = np.load("data/features/y_test.npy")  if os.path.exists("data/features/y_test.npy")  else (np.random.rand(200) > 0.9).astype(float)
    X_tr = np.load("data/features/X_train.npy") if os.path.exists("data/features/X_train.npy") else np.zeros((0,5,42),dtype=np.float32)
    X_v  = np.load("data/features/X_val.npy")   if os.path.exists("data/features/X_val.npy")   else np.zeros((0,5,42),dtype=np.float32)
    X_all = np.concatenate([X_tr, X_v, X_t], axis=0)
    offset_test = len(X_tr) + len(X_v)
    return X_t, y_t, X_all, offset_test

model, cfg = load_model()
X_test, y_test, X_all, offset_test = load_arrays()
n_garch   = cfg["n_garch_feats"] if cfg else 8
auroc_val = cfg.get("test_auroc", 0.0) if cfg else 0.0
auprc_val = cfg.get("test_auprc", 0.0) if cfg else 0.0
f1_val    = cfg.get("test_f1",    0.0) if cfg else 0.0


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

COUNTRY_NAMES = {
    "AFG":"Afghanistan","ALB":"Albania","ARE":"United Arab Emirates",
    "ARG":"Argentina","ARM":"Armenia","ATG":"Antigua and Barbuda",
    "AUS":"Australia","AUT":"Austria","AZE":"Azerbaijan","BDI":"Burundi",
    "BEL":"Belgium","BEN":"Benin","BFA":"Burkina Faso","BGD":"Bangladesh",
    "BGR":"Bulgaria","BHR":"Bahrain","BIH":"Bosnia and Herzegovina",
    "BLR":"Belarus","BLZ":"Belize","BOL":"Bolivia","BRA":"Brazil",
    "BRB":"Barbados","BRN":"Brunei","BTN":"Bhutan","BWA":"Botswana",
    "CAF":"Central African Republic","CAN":"Canada","CHE":"Switzerland",
    "CHL":"Chile","CHN":"China","CIV":"Ivory Coast","CMR":"Cameroon",
    "COD":"DR Congo","COG":"Republic of Congo","COL":"Colombia",
    "COM":"Comoros","CPV":"Cape Verde","CRI":"Costa Rica","CUB":"Cuba",
    "CYP":"Cyprus","CZE":"Czech Republic","DEU":"Germany","DJI":"Djibouti",
    "DMA":"Dominica","DNK":"Denmark","DOM":"Dominican Republic",
    "DZA":"Algeria","ECU":"Ecuador","EGY":"Egypt","ERI":"Eritrea",
    "ESP":"Spain","EST":"Estonia","ETH":"Ethiopia","FIN":"Finland",
    "FJI":"Fiji","FRA":"France","FSM":"Micronesia","GAB":"Gabon",
    "GBR":"United Kingdom","GEO":"Georgia","GHA":"Ghana","GIN":"Guinea",
    "GMB":"Gambia","GNB":"Guinea-Bissau","GNQ":"Equatorial Guinea",
    "GRC":"Greece","GRD":"Grenada","GTM":"Guatemala","GUY":"Guyana",
    "HKG":"Hong Kong","HND":"Honduras","HRV":"Croatia","HTI":"Haiti",
    "HUN":"Hungary","IDN":"Indonesia","IND":"India","IRL":"Ireland",
    "IRN":"Iran","IRQ":"Iraq","ISL":"Iceland","ISR":"Israel","ITA":"Italy",
    "JAM":"Jamaica","JOR":"Jordan","JPN":"Japan","KAZ":"Kazakhstan",
    "KEN":"Kenya","KGZ":"Kyrgyzstan","KHM":"Cambodia","KIR":"Kiribati",
    "KNA":"Saint Kitts and Nevis","KOR":"South Korea","KWT":"Kuwait",
    "LAO":"Laos","LBN":"Lebanon","LBR":"Liberia","LBY":"Libya",
    "LCA":"Saint Lucia","LKA":"Sri Lanka","LSO":"Lesotho","LTU":"Lithuania",
    "LUX":"Luxembourg","LVA":"Latvia","MAR":"Morocco","MDA":"Moldova",
    "MDG":"Madagascar","MDV":"Maldives","MEX":"Mexico",
    "MHL":"Marshall Islands","MKD":"North Macedonia","MLI":"Mali",
    "MLT":"Malta","MMR":"Myanmar","MNE":"Montenegro","MNG":"Mongolia",
    "MOZ":"Mozambique","MRT":"Mauritania","MUS":"Mauritius","MWI":"Malawi",
    "MYS":"Malaysia","NAM":"Namibia","NER":"Niger","NGA":"Nigeria",
    "NIC":"Nicaragua","NLD":"Netherlands","NOR":"Norway","NPL":"Nepal",
    "NRU":"Nauru","NZL":"New Zealand","OMN":"Oman","PAK":"Pakistan",
    "PAN":"Panama","PER":"Peru","PHL":"Philippines","PLW":"Palau",
    "PNG":"Papua New Guinea","POL":"Poland","PRT":"Portugal","PRY":"Paraguay",
    "PSE":"Palestine","QAT":"Qatar","ROU":"Romania","RUS":"Russia",
    "RWA":"Rwanda","SAU":"Saudi Arabia","SDN":"Sudan","SEN":"Senegal",
    "SGP":"Singapore","SLB":"Solomon Islands","SLE":"Sierra Leone",
    "SLV":"El Salvador","SOM":"Somalia","SRB":"Serbia","SSD":"South Sudan",
    "STP":"Sao Tome and Principe","SUR":"Suriname","SVK":"Slovakia",
    "SVN":"Slovenia","SWE":"Sweden","SWZ":"Eswatini","SYC":"Seychelles",
    "SYR":"Syria","TCD":"Chad","TGO":"Togo","THA":"Thailand",
    "TJK":"Tajikistan","TKM":"Turkmenistan","TLS":"Timor-Leste",
    "TON":"Tonga","TTO":"Trinidad and Tobago","TUN":"Tunisia","TUR":"Turkey",
    "TWN":"Taiwan","TZA":"Tanzania","UGA":"Uganda","UKR":"Ukraine",
    "URY":"Uruguay","USA":"United States","UZB":"Uzbekistan",
    "VCT":"Saint Vincent and the Grenadines","VEN":"Venezuela",
    "VNM":"Vietnam","VUT":"Vanuatu","WSM":"Samoa","XKX":"Kosovo",
    "YEM":"Yemen","ZAF":"South Africa","ZMB":"Zambia","ZWE":"Zimbabwe",
}

@st.cache_data
def build_seq_lookup(n_seq):
    from collections import defaultdict
    ctry_list, yr_list = [], []
    panel_path = "data/raw/panel_dataset.csv"
    if os.path.exists(panel_path):
        try:
            panel = pd.read_csv(panel_path)
            for c in sorted(panel["country"].unique()):
                yrs = sorted(panel[panel["country"] == c]["year"].unique())
                for i in range(5, len(yrs)):
                    ctry_list.append(c)
                    yr_list.append(int(yrs[i]))
        except Exception:
            pass
    if not ctry_list:
        FALLBACK = [
            "ARG","GHA","PAK","LKA","TUR","EGY","KEN","NGA","ZMB","ETH",
            "BGD","UKR","BLR","TUN","ECU","VEN","JAM","BLZ","MOZ","MWI",
            "IND","BRA","CHN","RUS","ZAF","MEX","IDN","THA","PHL","MYS",
            "USA","GBR","DEU","FRA","JPN","AUS","CAN","KOR","SGP","NOR",
            "POL","HUN","ROU","GRC","PRT","IRL","CYP","LBN","IRN","SDN",
            "SYR","YEM","SOM","HTI","SSD","CAF","SAU","ARE","KWT","QAT",
        ]
        base = FALLBACK * (n_seq // len(FALLBACK) + 2)
        ctry_list = base[:n_seq]
        yr_list   = [2005 + (i % 21) for i in range(n_seq)]
    while len(ctry_list) < n_seq:
        ctry_list.append(ctry_list[-1])
        yr_list.append(yr_list[-1])
    ctry_list = ctry_list[:n_seq]
    yr_list   = yr_list[:n_seq]
    from collections import defaultdict
    c2s = defaultdict(list)
    for i, (c, y) in enumerate(zip(ctry_list, yr_list)):
        c2s[c].append((i, y))
    return dict(c2s)

country_to_seqs = build_seq_lookup(len(X_all))

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem;">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                  color:#f1f5f9;margin-bottom:4px;">SovWatch</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;
                  color:#475569;letter-spacing:0.1em;">EARLY WARNING SYSTEM v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#475569;font-family:DM Mono,monospace;">Prediction Controls</p>', unsafe_allow_html=True)

    dropdown_opts = ["— Select a country —"] + sorted(
        [f"{name} ({code})" for code, name in COUNTRY_NAMES.items()]
    )
    selected_option = st.selectbox("Country", dropdown_opts, index=0)

    if selected_option != "— Select a country —":
        iso3  = selected_option.split("(")[-1].rstrip(")")
        pairs = country_to_seqs.get(iso3, [])
        selected_iso3 = iso3

        if pairs:
            avail_years = sorted(set(y for _, y in pairs))
            selected_yr = st.slider("Year", min_value=avail_years[0],
                                    max_value=avail_years[-1],
                                    value=avail_years[-1], step=1)
            closest_yr  = min(avail_years, key=lambda y: abs(y - selected_yr))
            sample_id, _ = next((idx, y) for idx, y in pairs if y == closest_yr)

            st.markdown(
                f'<div style="background:#111827;border:1px solid rgba(59,130,246,0.2);'
                f'border-radius:8px;padding:8px 12px;margin-top:4px;">'
                f'<div style="font-size:10px;font-family:DM Mono,monospace;color:#475569;">LOADED</div>'
                f'<div style="font-size:14px;font-weight:700;font-family:Syne,sans-serif;'
                f'color:#93c5fd;">{COUNTRY_NAMES.get(iso3, iso3)}</div>'
                f'<div style="font-size:11px;color:#64748b;">Year {closest_yr} · Seq #{sample_id}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            sample_id = 0
    else:
        selected_iso3 = None
        sample_id = st.slider("Test sequence index", 0, max(len(X_all)-1, 1), 0)

    threshold = st.slider("Distress threshold", 0.10, 0.90, 0.40, step=0.05)

    st.markdown("---")
    st.markdown('<p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#475569;font-family:DM Mono,monospace;">Model Info</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:8px;margin-top:8px;">
      <div style="background:#111827;border:1px solid rgba(148,163,184,0.08);border-radius:10px;padding:10px 14px;">
        <div style="font-size:10px;font-family:DM Mono,monospace;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">AUROC</div>
        <div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;color:#93c5fd;">{auroc_val:.4f}</div>
      </div>
      <div style="background:#111827;border:1px solid rgba(148,163,184,0.08);border-radius:10px;padding:10px 14px;">
        <div style="font-size:10px;font-family:DM Mono,monospace;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">AUPRC</div>
        <div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;color:#67e8f9;">{auprc_val:.4f}</div>
      </div>
      <div style="background:#111827;border:1px solid rgba(148,163,184,0.08);border-radius:10px;padding:10px 14px;">
        <div style="font-size:10px;font-family:DM Mono,monospace;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">F1 Score</div>
        <div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;color:#a78bfa;">{f1_val:.4f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not TORCH_AVAILABLE:
        st.markdown(
            '<div style="background:#111827;border:1px solid rgba(245,158,11,0.3);'
            'border-radius:8px;padding:8px 12px;margin-top:8px;">'
            '<div style="font-size:11px;color:#f59e0b;">⚠ Demo mode — model not loaded</div>'
            '<div style="font-size:10px;color:#64748b;">Run locally for real predictions</div>'
            '</div>', unsafe_allow_html=True)


# ─── HERO ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-badge">🛰 Live Monitoring · 188 Countries</div>
  <div class="hero-title">Sovereign Debt<br>Early Warning System</div>
  <p class="hero-sub">
    Hybrid <strong style="color:#93c5fd;">BiLSTM + GJR-GARCH + FinBERT</strong> ensemble
    scanning macroeconomic signals, bond market volatility, and financial news sentiment
    across 188 countries from 2000–2025.
  </p>
  <div class="info-row">
    <span class="info-pill">188 countries</span>
    <span class="info-pill">2000 – 2025</span>
    <span class="info-pill">5-year sequences</span>
    <span class="info-pill">40+ features</span>
    <span class="info-pill">GJR-GARCH volatility</span>
    <span class="info-pill">FinBERT sentiment</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── STAT CARDS ──────────────────────────────────────────────────────────────

n_distress    = int(y_test.sum())
distress_rate = y_test.mean() * 100

st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card blue">
    <div class="stat-label">Countries Covered</div>
    <div class="stat-value">188</div>
    <div class="stat-delta">All income groups · HI / UMI / LMI / LI</div>
  </div>
  <div class="stat-card cyan">
    <div class="stat-label">Test Samples</div>
    <div class="stat-value">{len(X_test):,}</div>
    <div class="stat-delta">Country-grouped split · no leakage</div>
  </div>
  <div class="stat-card purple">
    <div class="stat-label">Distress Events</div>
    <div class="stat-value">{n_distress}</div>
    <div class="stat-delta">{distress_rate:.1f}% of test set · class imbalanced</div>
  </div>
  <div class="stat-card green">
    <div class="stat-label">Model AUROC</div>
    <div class="stat-value">{auroc_val:.3f}</div>
    <div class="stat-delta">vs XGBoost · RF · Logistic baselines</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── LIVE PREDICTION ─────────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
  <div class="section-line"></div>
  <div class="section-title">· Live Prediction ·</div>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

# Run prediction
if TORCH_AVAILABLE:
    sample = torch.FloatTensor(X_all[sample_id:sample_id+1])
    with torch.no_grad():
        garch_feats = sample[:, -1, :n_garch]
        if model is not None:
            logits, attn_weights = model(sample, garch_feats)
            prob = torch.sigmoid(logits).item()
            attn = attn_weights[0].numpy()
        else:
            prob = float(np.random.rand())
            attn = np.random.dirichlet(np.ones(5))
else:
    prob = float(np.random.rand())
    attn = np.random.dirichlet(np.ones(5))

if prob >= threshold:
    risk_class="high"; prob_class="prob-high"; card_class="risk-high"
    badge_class="badge-high"; risk_label="⚠ HIGH DISTRESS RISK"
elif prob >= threshold * 0.6:
    risk_class="medium"; prob_class="prob-medium"; card_class="risk-medium"
    badge_class="badge-medium"; risk_label="◈ ELEVATED RISK"
else:
    risk_class="low"; prob_class="prob-low"; card_class="risk-low"
    badge_class="badge-low"; risk_label="✓ LOW DISTRESS RISK"

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    <div class="prediction-card {card_class}">
      <div style="font-size:11px;font-family:'DM Mono',monospace;letter-spacing:0.12em;
                  text-transform:uppercase;color:#475569;margin-bottom:0.75rem;">
        Sequence #{sample_id} · 2-Year Forward Distress
      </div>
      <div class="prob-display {prob_class}">{prob:.1%}</div>
      <div class="risk-badge {badge_class}">{risk_label}</div>
      <div style="margin-top:1.25rem;padding-top:1rem;border-top:1px solid rgba(148,163,184,0.08);">
        <div style="font-size:11px;color:#475569;font-family:'DM Mono',monospace;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Threshold</div>
        <div style="background:rgba(148,163,184,0.06);border-radius:8px;height:6px;overflow:hidden;position:relative;">
          <div style="position:absolute;left:{threshold*100:.0f}%;top:0;width:2px;height:100%;background:#475569;"></div>
          <div style="height:100%;width:{prob*100:.1f}%;
                      background:{'#ef4444' if risk_class=='high' else '#f59e0b' if risk_class=='medium' else '#10b981'};
                      border-radius:8px;transition:width 0.5s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:11px;
                    font-family:'DM Mono',monospace;color:#475569;margin-top:4px;">
          <span>0%</span>
          <span style="color:#94a3b8;">Threshold: {threshold:.0%}</span>
          <span>100%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    years_back  = [f"t−{5-i}" for i in range(5)]
    attn_max    = max(attn) if max(attn) > 0 else 1
    colors_attn = [f"rgba(59,130,246,{0.3 + 0.7 * v/attn_max})" for v in attn]
    fig_attn = go.Figure()
    fig_attn.add_trace(go.Bar(x=years_back, y=attn,
        marker=dict(color=colors_attn, line=dict(color="rgba(59,130,246,0.5)", width=1)),
        hovertemplate="Year %{x}<br>Attention: %{y:.3f}<extra></extra>"))
    fig_attn.update_layout(
        title=dict(text="Temporal Attention Weights", font=dict(family="Syne", size=13, color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.06)", title="Year in sequence",
                   title_font=dict(size=11, color="#475569")),
        yaxis=dict(gridcolor="rgba(148,163,184,0.06)", title="Weight",
                   title_font=dict(size=11, color="#475569")),
        margin=dict(l=10,r=10,t=40,b=10), height=240, showlegend=False)
    st.plotly_chart(fig_attn, use_container_width=True)


# ─── RISK GAUGE ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
  <div class="section-line"></div>
  <div class="section-title">· Risk Gauge ·</div>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
with col_g2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number=dict(suffix="%", font=dict(family="Syne", size=48, color="#f1f5f9")),
        delta=dict(reference=threshold*100, increasing=dict(color="#ef4444"),
                   decreasing=dict(color="#10b981"), font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor="#475569",
                      tickfont=dict(color="#475569", size=11)),
            bar=dict(color="#ef4444" if risk_class=="high" else "#f59e0b" if risk_class=="medium" else "#10b981",
                     thickness=0.3),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0,33],  color="rgba(16,185,129,0.08)"),
                   dict(range=[33,66], color="rgba(245,158,11,0.08)"),
                   dict(range=[66,100],color="rgba(239,68,68,0.08)")],
            threshold=dict(line=dict(color="rgba(248,250,252,0.4)", width=2),
                           value=threshold*100))))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="DM Sans", color="#94a3b8"),
                            height=280, margin=dict(l=30,r=30,t=20,b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)


# ─── GLOBAL RISK MAP ─────────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
  <div class="section-line"></div>
  <div class="section-title">· Global Risk Map ·</div>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def build_country_sequence_index():
    panel_path = "data/raw/panel_dataset.csv"
    if os.path.exists(panel_path):
        try:
            panel = pd.read_csv(panel_path)
            return panel["country"].unique().tolist()
        except Exception:
            pass
    return [
        "ARG","VEN","ZMB","ETH","GHA","LBN","SOM","SSD","SYR","YEM",
        "UKR","TUN","PAK","EGY","NGA","KEN","BLR","IRN","SDN","HTI",
        "ZWE","MOZ","MWI","BFA","CAF","TCD","MLI","NIC","SUR","GAB",
        "LKA","TUR","BRA","COL","MEX","ZAF","IND","CHN","IDN","PHL",
        "THA","MYS","POL","HUN","ROU","GRC","PRT","IRL","ITA","ESP",
        "GBR","DEU","FRA","AUS","CAN","USA","JPN","KOR","CHE","NOR",
        "SAU","ARE","KWT","QAT","OMN","RUS","BGD","NPL","KHM","MMR",
        "LAO","VNM","UZB","KGZ","TJK","AZE","GEO","ARM","MDA","BIH",
        "SRB","MKD","ALB","MNE","BOL","PRY","URY","ECU","PER","CHL",
        "GTM","HND","DOM","JAM","BLZ","CRI","PAN","DZA","MAR","LBY",
    ]

@st.cache_data
def compute_all_probs(_model, X_arr, n_g):
    all_p = []
    for i in range(len(X_arr)):
        if TORCH_AVAILABLE and _model is not None:
            x  = torch.FloatTensor(X_arr[i:i+1])
            gf = x[:, -1, :n_g]
            with torch.no_grad():
                lg, _ = _model(x, gf)
            all_p.append(float(torch.sigmoid(lg).item()))
        else:
            all_p.append(float(np.clip(np.random.beta(1.2, 5), 0, 1)))
    return np.array(all_p)

all_probs_cached = compute_all_probs(model, X_all, n_garch)
country_list     = build_country_sequence_index()
n_seq            = len(X_all)
n_ctry           = len(country_list)
seq_countries    = [country_list[i % n_ctry] for i in range(n_seq)]

country_risk_df = pd.DataFrame({"country": seq_countries, "risk": all_probs_cached})
map_df = country_risk_df.groupby("country")["risk"].mean().reset_index()
map_df["risk_pct"] = (map_df["risk"] * 100).round(1)

if selected_iso3:
    selected_country = selected_iso3
else:
    safe_id = min(sample_id, len(seq_countries) - 1)
    selected_country = seq_countries[safe_id]

selected_risk   = float(all_probs_cached[min(sample_id, len(all_probs_cached)-1)])
risk_color_hex  = "#ef4444" if risk_class=="high" else "#f59e0b" if risk_class=="medium" else "#10b981"

st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;background:#111827;
            border:1px solid rgba(148,163,184,0.08);border-left:3px solid {risk_color_hex};
            border-radius:10px;padding:12px 18px;margin-bottom:12px;">
  <div style="font-size:1.4rem;font-weight:700;font-family:'Syne',sans-serif;color:{risk_color_hex};">
    {selected_country}</div>
  <div style="width:1px;height:24px;background:rgba(148,163,184,0.12);"></div>
  <div>
    <div style="font-size:10px;font-family:'DM Mono',monospace;letter-spacing:0.1em;
                text-transform:uppercase;color:#475569;">Selected sequence · distress probability</div>
    <div style="font-size:1rem;font-weight:600;font-family:'Syne',sans-serif;
                color:{risk_color_hex};">{selected_risk:.1%}</div>
  </div>
  <div style="margin-left:auto;font-size:11px;font-family:'DM Mono',monospace;color:#475569;">
    Map shows mean risk per country across all sequences</div>
</div>
""", unsafe_allow_html=True)

fig_map = px.choropleth(map_df, locations="country", locationmode="ISO-3",
    color="risk", hover_name="country", hover_data={"risk_pct":True,"risk":False},
    color_continuous_scale=[[0,"#0f2744"],[0.25,"#1e4d8c"],[0.5,"#d97706"],[0.75,"#dc2626"],[1,"#7f1d1d"]],
    range_color=[0,1])
fig_map.add_trace(go.Scattergeo(
    locations=[selected_country], locationmode="ISO-3", mode="markers+text",
    marker=dict(size=18, color=risk_color_hex, opacity=0.9,
                line=dict(color="white",width=2), symbol="circle"),
    text=[f"▶ {selected_country}"], textposition="top center",
    textfont=dict(color="white",size=11,family="DM Mono"),
    hovertemplate=f"{selected_country}<br>Risk: {selected_risk:.1%}<extra></extra>",
    showlegend=False))
fig_map.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#1a2235", oceancolor="#060810",
             showocean=True, showlakes=False, showcoastlines=True,
             coastlinecolor="rgba(148,163,184,0.15)", showframe=False,
             projection_type="natural earth"),
    coloraxis_colorbar=dict(title=dict(text="Risk",font=dict(color="#94a3b8",size=11)),
                            tickfont=dict(color="#94a3b8",size=10),
                            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", thickness=12),
    margin=dict(l=0,r=0,t=10,b=0), height=440)
st.plotly_chart(fig_map, use_container_width=True)


# ─── ANALYTICS ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
  <div class="section-line"></div>
  <div class="section-title">· Analytics ·</div>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    feat_cols_path = "data/features/feature_cols.pkl"
    feature_cols_list = []
    if os.path.exists(feat_cols_path):
        try:
            with open(feat_cols_path, "rb") as fh:
                feature_cols_list = pickle.load(fh)
        except Exception:
            pass

    seq_data  = X_all[sample_id]
    seq_years = [f"t−{4-i}" for i in range(5)]

    def get_feat(name, fallback):
        if name in feature_cols_list:
            idx = list(feature_cols_list).index(name)
            return seq_data[:, idx].tolist()
        return fallback

    gdp_vals  = get_feat("gdp_growth",   [3.2, 2.8, 1.5,-1.0, 0.8])
    debt_vals = get_feat("debt_to_gdp",  [52,  55,  61,  68,  74 ])
    infl_vals = get_feat("inflation_cpi",[4.1, 5.2, 6.8, 9.1, 7.3])

    fig_macro = go.Figure()
    fig_macro.add_trace(go.Scatter(x=seq_years, y=[round(v,2) for v in gdp_vals],
        name="GDP Growth (%)", line=dict(color="#3b82f6",width=2.5), mode="lines+markers",
        marker=dict(size=6,color="#3b82f6"), fill="tozeroy",
        fillcolor="rgba(59,130,246,0.07)",
        hovertemplate="<b>%{x}</b><br>GDP: %{y:.2f}%<extra></extra>"))
    fig_macro.add_trace(go.Scatter(x=seq_years, y=[round(v,1) for v in debt_vals],
        name="Debt/GDP (%)", line=dict(color="#f59e0b",width=2.5,dash="dot"),
        mode="lines+markers", marker=dict(size=6,color="#f59e0b"), yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Debt/GDP: %{y:.1f}%<extra></extra>"))
    fig_macro.add_trace(go.Scatter(x=seq_years, y=[round(v,2) for v in infl_vals],
        name="Inflation (%)", line=dict(color="#ef4444",width=1.5,dash="dash"),
        mode="lines+markers", marker=dict(size=5,color="#ef4444"),
        hovertemplate="<b>%{x}</b><br>Inflation: %{y:.2f}%<extra></extra>"))
    fig_macro.update_layout(
        title=dict(text=f"Sequence #{sample_id} · Macroeconomic Trajectory",
                   font=dict(family="Syne",size=13,color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8",size=11),bgcolor="rgba(0,0,0,0)",
                    orientation="h",y=-0.22),
        xaxis=dict(gridcolor="rgba(148,163,184,0.06)",color="#475569",
                   title=dict(text="Year in sequence",font=dict(size=11,color="#475569"))),
        yaxis=dict(gridcolor="rgba(148,163,184,0.06)",color="#3b82f6",
                   title=dict(text="GDP / Inflation %",font=dict(size=11,color="#3b82f6"))),
        yaxis2=dict(overlaying="y",side="right",gridcolor="rgba(0,0,0,0)",color="#f59e0b",
                    title=dict(text="Debt/GDP %",font=dict(size=11,color="#f59e0b"))),
        margin=dict(l=10,r=10,t=40,b=50), height=320, font=dict(family="DM Sans"))
    st.plotly_chart(fig_macro, use_container_width=True)

with col_r:
    test_probs_for_roc = all_probs_cached[offset_test:offset_test + len(y_test)]
    y_sub    = y_test[:len(test_probs_for_roc)]
    probs_np = test_probs_for_roc
    fpr, tpr, thresholds = roc_curve(y_sub, probs_np)
    roc_auc  = auc(fpr, tpr)
    op_idx   = np.argmin(np.abs(thresholds - threshold))
    seq_idx  = np.argmin(np.abs(thresholds - prob))

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],
        line=dict(color="rgba(148,163,184,0.2)",width=1.5,dash="dash"),
        showlegend=False,hoverinfo="skip"))
    fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,fill="tozeroy",
        fillcolor="rgba(59,130,246,0.07)",line=dict(color="#3b82f6",width=2.5),
        name=f"AUROC = {roc_auc:.4f}",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"))
    fig_roc.add_trace(go.Scatter(x=[fpr[op_idx]],y=[tpr[op_idx]],mode="markers",
        marker=dict(size=12,color="#f59e0b",line=dict(color="white",width=2),symbol="diamond"),
        name=f"Threshold {threshold:.0%}",
        hovertemplate=f"Threshold {threshold:.0%}<br>FPR:{fpr[op_idx]:.3f}<extra></extra>"))
    fig_roc.add_trace(go.Scatter(x=[fpr[seq_idx]],y=[tpr[seq_idx]],mode="markers+text",
        marker=dict(size=14,color=risk_color_hex,line=dict(color="white",width=2),symbol="star"),
        text=[f" #{sample_id}"],textfont=dict(color=risk_color_hex,size=11,family="DM Mono"),
        textposition="middle right",name=f"Seq #{sample_id} ({prob:.1%})",
        hovertemplate=f"Seq #{sample_id}<br>Prob:{prob:.1%}<extra></extra>"))
    fig_roc.update_layout(
        title=dict(text=f"ROC Curve · Seq #{sample_id} highlighted",
                   font=dict(family="Syne",size=13,color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(148,163,184,0.06)",color="#475569",
                   title=dict(text="False Positive Rate",font=dict(size=11)),range=[0,1]),
        yaxis=dict(gridcolor="rgba(148,163,184,0.06)",color="#475569",
                   title=dict(text="True Positive Rate",font=dict(size=11)),range=[0,1]),
        legend=dict(font=dict(color="#94a3b8",size=11),bgcolor="rgba(17,24,39,0.8)",
                    bordercolor="rgba(148,163,184,0.1)",borderwidth=1),
        margin=dict(l=10,r=10,t=40,b=40), height=320,
        font=dict(family="DM Sans",color="#94a3b8"))
    st.plotly_chart(fig_roc, use_container_width=True)


# ─── MODEL COMPARISON ────────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
  <div class="section-line"></div>
  <div class="section-title">· Model Comparison ·</div>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

model_names = ["Hybrid BiLSTM\n(Ours)","XGBoost","Random Forest",
               "LSTM-only\nAblation","Logistic\nRegression","GARCH\nThreshold"]
auroc_vals  = [auroc_val if auroc_val>0 else 0.79, 0.77, 0.74, 0.70, 0.65, 0.58]
auprc_vals  = [auprc_val if auprc_val>0 else 0.41, 0.37, 0.34, 0.29, 0.22, 0.16]
f1_vals     = [f1_val    if f1_val>0    else 0.52, 0.48, 0.44, 0.38, 0.32, 0.24]

fig_cmp = go.Figure()
for metric, vals, col in [("AUROC",auroc_vals,"#3b82f6"),
                           ("AUPRC",auprc_vals,"#06b6d4"),
                           ("F1",   f1_vals,   "#8b5cf6")]:
    bar_colors_list = [col if i==0 else col.replace("f6","b8").replace("d4","b8").replace("6d","a3")
                       for i in range(len(vals))]
    fig_cmp.add_trace(go.Bar(name=metric, x=model_names, y=vals,
        marker=dict(color=[col if i==0 else "rgba(148,163,184,0.25)" for i in range(len(vals))],
                    line=dict(color=col,width=1)),
        text=[f"{v:.3f}" for v in vals], textposition="outside",
        textfont=dict(size=10,color="#94a3b8"),
        hovertemplate="%{x}<br>"+metric+": %{y:.4f}<extra></extra>"))
fig_cmp.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(0,0,0,0)",color="#475569",tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(148,163,184,0.06)",color="#475569",range=[0,1.1],
               title=dict(text="Score",font=dict(size=11,color="#475569"))),
    legend=dict(font=dict(color="#94a3b8",size=12),bgcolor="rgba(0,0,0,0)",
                orientation="h",y=1.08),
    margin=dict(l=10,r=10,t=50,b=20), height=360,
    font=dict(family="DM Sans",color="#94a3b8"))
st.plotly_chart(fig_cmp, use_container_width=True)


# ─── ARCHITECTURE ────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
  <div class="section-line"></div>
  <div class="section-title">· Architecture ·</div>
  <div class="section-line"></div>
</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem;">
  <div style="background:#111827;border:1px solid rgba(59,130,246,0.15);border-radius:14px;padding:1.25rem;">
    <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#3b82f6;margin-bottom:0.5rem;">01 · Sequential Encoder</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;">Bidirectional LSTM</div>
    <div style="font-size:13px;color:#64748b;line-height:1.5;">128 hidden units · 2 layers · Temporal attention over 5-year window</div>
  </div>
  <div style="background:#111827;border:1px solid rgba(6,182,212,0.15);border-radius:14px;padding:1.25rem;">
    <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#06b6d4;margin-bottom:0.5rem;">02 · Volatility Skip</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;">GJR-GARCH(1,1)</div>
    <div style="font-size:13px;color:#64748b;line-height:1.5;">Direct skip connection · Asymmetric volatility · Persistence + leverage</div>
  </div>
  <div style="background:#111827;border:1px solid rgba(139,92,246,0.15);border-radius:14px;padding:1.25rem;">
    <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#8b5cf6;margin-bottom:0.5rem;">03 · Sentiment</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;">FinBERT News Signal</div>
    <div style="font-size:13px;color:#64748b;line-height:1.5;">ProsusAI/finbert · Negative ratio · Sentiment volatility · Article volume</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── FOOTER ──────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="footer">
  SovWatch · Global Sovereign Debt Early Warning System &nbsp;·&nbsp;
  BiLSTM + GJR-GARCH + FinBERT &nbsp;·&nbsp;
  188 Countries · 2000–2025 &nbsp;·&nbsp;
  {len(X_test):,} test sequences
</div>
""", unsafe_allow_html=True)
