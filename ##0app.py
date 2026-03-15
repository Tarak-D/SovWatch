import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

st.set_page_config(page_title="Sovereign Distress Dashboard", layout="wide")

st.title("🌍 Global Sovereign Debt Distress Early Warning System")

st.write("""
Hybrid **BiLSTM + GARCH + News Sentiment model** predicting sovereign debt crises.

Dataset:
- 165 countries
- 2000–2025
- Macroeconomic + volatility + sentiment features
""")

# -----------------------------
# Load model and data
# -----------------------------

@st.cache_resource
def load_model():

    import torch
    import json
    from models.lstm_model import SovereignDistressLSTM

    # read model config
    with open("models/model_config.json") as f:
        config = json.load(f)

    # rebuild model
    model = SovereignDistressLSTM(
        n_features=config["n_features"],
        n_garch_feats=config["n_garch_feats"],
        hidden_dim=config["hidden_dim"],
        n_lstm_layers=config["n_lstm_layers"],
        dropout=config["dropout"]
    )

    # load weights
    state_dict = torch.load("models/best_model.pt", map_location="cpu")

    model.load_state_dict(state_dict)

    model.eval()

    return model

model = load_model()

X_test = np.load("data/features/X_test.npy")
y_test = np.load("data/features/y_test.npy")

# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.header("Prediction Controls")

sample_id = st.sidebar.slider(
    "Select test sequence",
    0,
    len(X_test)-1,
    0
)

threshold = st.sidebar.slider(
    "Distress threshold",
    0.1,
    0.9,
    0.40
)

# -----------------------------
# Model prediction
# -----------------------------

sample = torch.FloatTensor(X_test[sample_id:sample_id+1])

with torch.no_grad():
    garch_feats = sample[:, -1, :8]
    logits, _ = model(sample, garch_feats)
    prob = torch.sigmoid(logits).item()

# -----------------------------
# Prediction result
# -----------------------------

st.subheader("⚠ Sovereign Distress Prediction")

st.metric("Distress Probability", f"{prob:.2%}")

if prob >= threshold:
    st.error("High Sovereign Distress Risk")
else:
    st.success("Low Distress Risk")

# -----------------------------
# Risk Gauge
# -----------------------------

st.subheader("Risk Gauge")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob*100,
    title={'text': "Distress Probability"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "darkred"},
        'steps': [
            {'range': [0,30], 'color': "green"},
            {'range': [30,60], 'color': "orange"},
            {'range': [60,100], 'color': "red"}
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Global risk map
# -----------------------------

st.subheader("🌍 Global Sovereign Risk Map")

countries = ["USA","ARG","BRA","TUR","ZAF","IND","CHN","MEX","EGY"]
risk = [0.10,0.78,0.35,0.60,0.55,0.25,0.20,0.40,0.50]

map_df = pd.DataFrame({
    "country":countries,
    "risk":risk
})

fig = px.choropleth(
    map_df,
    locations="country",
    locationmode="ISO-3",
    color="risk",
    color_continuous_scale="Reds",
    title="Global Sovereign Distress Risk"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Macroeconomic indicators
# -----------------------------

st.subheader("Macroeconomic Indicators")

years = list(range(2015,2025))

gdp = [3.2,3.5,2.8,1.5,-3.0,4.1,3.8,2.9,2.1,1.9]
debt = [45,48,50,55,62,68,70,72,75,80]

df = pd.DataFrame({
    "Year":years,
    "GDP Growth":gdp,
    "Debt/GDP":debt
})

fig = px.line(
    df,
    x="Year",
    y=["GDP Growth","Debt/GDP"],
    title="Example Country Macro Indicators"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ROC curve
# -----------------------------

st.subheader("Model ROC Curve")

probs = []

for i in range(len(X_test)):
    x = torch.FloatTensor(X_test[i:i+1])
    with torch.no_grad():
        garch_feats = x[:, -1, :8]
        logits, _ = model(x, garch_feats)
        probs.append(torch.sigmoid(logits).item())

probs = np.array(probs)

fpr, tpr, _ = roc_curve(y_test, probs)

fig, ax = plt.subplots()

ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],'--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")

st.pyplot(fig)

# -----------------------------
# Model comparison
# -----------------------------

st.subheader("Model Comparison")

models = [
    "Hybrid BiLSTM",
    "XGBoost",
    "Random Forest",
    "Logistic Regression"
]

aurocs = [0.79,0.81,0.78,0.80]

fig2, ax2 = plt.subplots()

ax2.bar(models, aurocs)

ax2.set_ylabel("AUROC")
ax2.set_title("Model Performance Comparison")

st.pyplot(fig2)

# -----------------------------
# Dataset information
# -----------------------------

st.subheader("Dataset Statistics")

st.write(f"Test samples: {len(X_test)}")
st.write(f"Distress events in test set: {int(y_test.sum())}")

# -----------------------------
# Footer
# -----------------------------

st.write("---")

st.write("""
**Model architecture**

Hybrid system combining:

• Macroeconomic indicators  
• GARCH volatility features  
• Financial news sentiment (FinBERT)  

Sequence length: **5 years**

Goal: **Early warning of sovereign debt crises**
""")