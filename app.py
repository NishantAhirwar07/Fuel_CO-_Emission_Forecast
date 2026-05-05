import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px

# FIX: Proper import for components
import streamlit.components.v1 as components


# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Emission Forecast",
    page_icon="🌫️",
    layout="wide"
)


# ── MODEL ───────────────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)

    n = 1000
    cylinders = np.random.choice([4, 6, 8, 10, 12], size=n)
    engine = cylinders * 0.35 + np.random.normal(0, 0.4, n)
    engine = np.clip(engine, 1.0, 8.5)

    fuel = engine * 2.1 + cylinders * 0.5 + np.random.normal(0, 0.8, n)
    fuel = np.clip(fuel, 4.0, 22.0)

    co2 = 7*cylinders + 11.6*engine + 9.3*fuel + 65 + np.random.normal(0, 12, n)

    X = np.column_stack([cylinders, engine, fuel])
    y = co2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


model = train_model()


# ── SIDEBAR INPUT ───────────────────────────────────────────
st.sidebar.title("⚙ Vehicle Inputs")

cylinders = st.sidebar.slider("Cylinders", 2, 16, 6)
engine_size = st.sidebar.slider("Engine Size (L)", 1.0, 8.0, 2.5)
fuel = st.sidebar.slider("Fuel Consumption", 4.0, 22.0, 10.0)


# ── SAFE PREDICTION ─────────────────────────────────────────
try:
    X_input = np.array([[cylinders, engine_size, fuel]])
    prediction = float(model.predict(X_input)[0])
    prediction = max(80, prediction)
except:
    prediction = 0


# ── LEVEL LOGIC ─────────────────────────────────────────────
if prediction < 150:
    level, color = "LOW", "green"
elif prediction < 200:
    level, color = "MODERATE", "orange"
elif prediction < 270:
    level, color = "HIGH", "red"
else:
    level, color = "EXTREME", "darkred"


# ── UI ──────────────────────────────────────────────────────
st.title("🌫️ CO₂ Emission Predictor")

st.metric("Predicted CO₂ (g/km)", f"{prediction:.0f}", level)


# ── GAUGE CHART ─────────────────────────────────────────────
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    gauge={
        "axis": {"range": [0, 450]},
        "bar": {"color": color}
    }
))

st.plotly_chart(fig, use_container_width=True)


# ── FEATURE IMPACT ──────────────────────────────────────────
coefs = model.coef_

features = ["Cylinders", "Engine", "Fuel"]
values = [
    coefs[0]*cylinders,
    coefs[1]*engine_size,
    coefs[2]*fuel
]

fig_bar = px.bar(
    x=values,
    y=features,
    orientation='h',
    title="Feature Contribution"
)

st.plotly_chart(fig_bar, use_container_width=True)


# ── FOOTER ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("🚀 Built with Streamlit & Scikit-learn")
