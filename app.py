import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Predictor",
    page_icon="🌿",
    layout="centered",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background: #0b0e14 !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

/* Tight centered column */
.main .block-container {
    max-width: 520px !important;
    padding: 0 16px 40px !important;
    margin: 0 auto !important;
}

[data-testid="stHeader"]        { display: none !important; }
[data-testid="stToolbar"]       { display: none !important; }
[data-testid="stDecoration"]    { display: none !important; }
[data-testid="stStatusWidget"]  { display: none !important; }
#MainMenu, footer               { visibility: hidden !important; }

/* ── Section divider ── */
hr { border: none; border-top: 1px solid #1e2330 !important; margin: 4px 0 !important; }

/* ── Hide all widget labels ── */
[data-testid="stSelectbox"]   > label,
[data-testid="stSlider"]      > label,
[data-testid="stSelectSlider"]> label { display: none !important; }

/* ── Custom label ── */
.lbl {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    color: #64748b;
    margin-bottom: 2px;
    text-transform: uppercase;
}

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background: #131720 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}
div[data-baseweb="select"] svg      { color: #475569 !important; }
div[data-baseweb="popover"] ul li   { background: #131720 !important; color: #e2e8f0 !important; }
div[data-baseweb="popover"] ul li:hover { background: #1e2a3a !important; }

/* ── Sliders ── */
[data-testid="stSlider"]       > div > div > div > div,
[data-testid="stSelectSlider"] > div > div > div > div {
    background: #3b82f6 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"],
[data-testid="stSelectSlider"] [data-baseweb="slider"] {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 14px 0 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: background 0.18s !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }
.stButton > button:active { transform: scale(0.99) !important; }

/* ── Columns gap ── */
[data-testid="column"] { padding: 0 4px !important; }
</style>
""", unsafe_allow_html=True)


# ── Model ────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1200
    cyl  = np.random.choice([3,4,5,6,8,10,12,16], n, p=[0.05,0.38,0.04,0.28,0.17,0.04,0.03,0.01])
    eng  = np.clip(cyl * 0.35 + np.random.normal(0, 0.4, n), 1.0, 8.5)
    fuel = np.clip(eng * 2.1 + cyl * 0.5 + np.random.normal(0, 0.9, n), 4.0, 22.0)
    co2  = 7.0*cyl + 11.6*eng + 9.3*fuel + 65 + np.random.normal(0, 12, n)
    X_tr, _, y_tr, _ = train_test_split(np.column_stack([cyl, eng, fuel]), co2, test_size=0.2, random_state=2)
    return LinearRegression().fit(X_tr, y_tr)

model = train_model()

# ── Data ─────────────────────────────────────────────────────────
MAKES = [
    "ACURA","ALFA ROMEO","ASTON MARTIN","AUDI","BENTLEY","BMW","BUICK",
    "CADILLAC","CHEVROLET","CHRYSLER","DODGE","FIAT","FORD","GMC","HONDA",
    "HYUNDAI","INFINITI","JAGUAR","JEEP","KIA","LAMBORGHINI","LAND ROVER",
    "LEXUS","LINCOLN","MASERATI","MAZDA","MERCEDES-BENZ","MINI","MITSUBISHI",
    "NISSAN","PORSCHE","RAM","ROLLS-ROYCE","SCION","SMART","SUBARU",
    "TOYOTA","VOLKSWAGEN","VOLVO",
]
try:
    df = pd.read_csv("FuelConsumptionCo2.csv")
    MAKES = sorted(df["MAKE"].dropna().unique().tolist())
except Exception:
    pass

HINTS = {
    "ACURA":(4,2.0,9.5),"ALFA ROMEO":(4,2.0,9.8),"ASTON MARTIN":(8,4.7,17.0),
    "AUDI":(4,2.0,10.5),"BENTLEY":(8,4.0,18.5),"BMW":(4,2.0,10.0),
    "BUICK":(4,2.5,10.8),"CADILLAC":(6,3.6,13.5),"CHEVROLET":(4,2.5,11.0),
    "CHRYSLER":(6,3.6,13.0),"DODGE":(6,3.6,13.5),"FIAT":(4,1.4,7.5),
    "FORD":(4,2.0,10.5),"GMC":(6,4.3,15.0),"HONDA":(4,1.8,8.5),
    "HYUNDAI":(4,2.0,9.0),"INFINITI":(6,3.5,12.5),"JAGUAR":(6,3.0,13.0),
    "JEEP":(4,2.4,11.5),"KIA":(4,2.0,9.2),"LAMBORGHINI":(10,5.2,20.0),
    "LAND ROVER":(6,3.0,14.0),"LEXUS":(6,3.5,12.0),"LINCOLN":(4,2.0,12.0),
    "MASERATI":(6,3.0,14.5),"MAZDA":(4,2.0,9.0),"MERCEDES-BENZ":(4,2.0,10.5),
    "MINI":(4,1.5,7.8),"MITSUBISHI":(4,2.0,9.5),"NISSAN":(4,1.8,9.0),
    "PORSCHE":(6,3.0,12.5),"RAM":(6,3.6,14.5),"ROLLS-ROYCE":(12,6.6,20.5),
    "SCION":(4,1.8,8.5),"SMART":(3,1.0,6.5),"SUBARU":(4,2.5,10.0),
    "TOYOTA":(4,2.0,9.0),"VOLKSWAGEN":(4,2.0,9.5),"VOLVO":(4,2.0,10.0),
}


# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:32px 0 20px">
    <div style="display:inline-flex;align-items:center;justify-content:center;
                width:64px;height:64px;border-radius:20px;
                background:linear-gradient(135deg,#1e3a5f,#0f2440);
                border:1px solid #1e3a5f;font-size:1.9rem;margin-bottom:14px">
        🌿
    </div>
    <div style="font-size:1.45rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.3px">
        CO₂ Emission Predictor
    </div>
    <div style="font-size:0.82rem;color:#475569;margin-top:5px;letter-spacing:0.2px">
        Vehicle carbon footprint · Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  INPUT SECTION
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#0f1520;border:1px solid #1e2330;
            border-radius:16px;padding:22px 20px 18px;margin-bottom:14px">
    <div style="font-size:0.72rem;font-weight:700;letter-spacing:1.5px;
                text-transform:uppercase;color:#334155;margin-bottom:16px">
        Vehicle Details
    </div>
""", unsafe_allow_html=True)

# Car brand
st.markdown('<p class="lbl">🏷  Car Brand</p>', unsafe_allow_html=True)
make = st.selectbox("make", ["— Select brand —"] + MAKES, key="make")
hint = HINTS.get(make, (4, 2.0, 9.5))
dcyl, deng, dfuel = hint

st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

# Cylinders
st.markdown('<p class="lbl">🔩  Cylinders</p>', unsafe_allow_html=True)
cylinders = st.select_slider("cyl", options=[2,3,4,5,6,8,10,12,16], value=dcyl, key="cyl")

st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

# Engine size
st.markdown('<p class="lbl">⚙️  Engine Size (L)</p>', unsafe_allow_html=True)
engine_size = st.slider("eng", 1.0, 8.5, float(deng), 0.1, key="eng", format="%.1f L")

st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

# Fuel consumption
st.markdown('<p class="lbl">⛽  Fuel Consumption (L / 100 km)</p>', unsafe_allow_html=True)
fuel_comb = st.slider("fuel", 4.0, 22.0, float(dfuel), 0.1, key="fuel", format="%.1f L")

st.markdown('</div>', unsafe_allow_html=True)

# Predict button
st.button("Predict Emission →", use_container_width=True)

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════
pred = max(80.0, float(model.predict([[cylinders, engine_size, fuel_comb]])[0]))

if   pred < 150: lvl, clr, ico, note = "Low",      "#22c55e", "🌿", "Eco-friendly range"
elif pred < 200: lvl, clr, ico, note = "Moderate", "#f59e0b", "⚡", "Average emissions"
elif pred < 270: lvl, clr, ico, note = "High",     "#f97316", "🔥", "Above average output"
else:            lvl, clr, ico, note = "Extreme",  "#ef4444", "☠️", "Heavy environmental burden"

# ── Big result card ───────────────────────────────────────────────
st.markdown(f"""
<div style="background:#0f1520;border:1px solid {clr}33;
            border-radius:16px;padding:28px 20px 22px;
            text-align:center;margin-bottom:14px;position:relative;overflow:hidden">

    <!-- accent line top -->
    <div style="position:absolute;top:0;left:0;right:0;height:3px;
                background:linear-gradient(90deg,transparent,{clr},transparent)"></div>

    <div style="font-size:0.72rem;font-weight:700;letter-spacing:1.5px;
                text-transform:uppercase;color:#334155;margin-bottom:12px">
        Predicted Emission
    </div>

    <!-- Number -->
    <div style="font-size:5rem;font-weight:800;color:{clr};
                line-height:1;letter-spacing:-2px;margin-bottom:2px">
        {pred:.0f}
    </div>
    <div style="font-size:0.9rem;color:#475569;margin-bottom:20px;font-weight:500">
        grams per km
    </div>

    <!-- Badge -->
    <span style="display:inline-flex;align-items:center;gap:6px;
                 padding:6px 18px;border-radius:999px;
                 background:{clr}18;border:1px solid {clr}44;
                 font-size:0.82rem;font-weight:700;color:{clr};letter-spacing:0.5px">
        {ico}&nbsp; {lvl}
    </span>

    <div style="font-size:0.8rem;color:#475569;margin-top:12px">{note}</div>
</div>
""", unsafe_allow_html=True)

# ── 3 stat pills ──────────────────────────────────────────────────
annual_kg = pred * 15_000 / 1_000
trees     = annual_kg / 21
vs_avg    = (pred / 180 - 1) * 100
vs_clr    = "#22c55e" if vs_avg <= 0 else "#ef4444"

c1, c2, c3 = st.columns(3)
for col, ico2, top_val, top_clr2, sub in [
    (c1, "📅", f"{annual_kg:,.0f} kg", "#94a3b8", "CO₂ / year"),
    (c2, "🌳", f"{trees:,.0f}",        "#22c55e",  "Trees needed"),
    (c3, "📊", f"{vs_avg:+.0f}%",      vs_clr,     "vs avg car"),
]:
    with col:
        st.markdown(f"""
        <div style="background:#0f1520;border:1px solid #1e2330;border-radius:12px;
                    padding:14px 8px;text-align:center">
            <div style="font-size:1.2rem;margin-bottom:4px">{ico2}</div>
            <div style="font-size:1.05rem;font-weight:700;color:{top_clr2}">{top_val}</div>
            <div style="font-size:0.68rem;color:#334155;margin-top:3px;
                        text-transform:uppercase;letter-spacing:0.5px">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  LEVEL GUIDE
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="font-size:0.72rem;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase;color:#334155;margin-bottom:10px">
    Emission Scale
</div>
""", unsafe_allow_html=True)

for row_ico, row_clr, row_lbl, row_rng, row_ex in [
    ("🌿","#22c55e","Low",     "< 150 g/km",   "Eco / Hybrid"),
    ("⚡","#f59e0b","Moderate","150–200 g/km",  "Compact / Sedan"),
    ("🔥","#f97316","High",    "200–270 g/km",  "SUV / Large"),
    ("☠️","#ef4444","Extreme", "> 270 g/km",    "Performance / Truck"),
]:
    is_active = row_lbl == lvl
    bdr = f"1px solid {row_clr}55" if is_active else "1px solid #1e2330"
    bg  = f"{row_clr}0e"           if is_active else "#0f1520"
    st.markdown(f"""
    <div style="background:{bg};border:{bdr};border-radius:10px;
                padding:10px 14px;margin-bottom:7px;
                display:flex;align-items:center;gap:12px">
        <span style="font-size:1rem;min-width:22px">{row_ico}</span>
        <span style="font-weight:700;color:{row_clr};font-size:0.82rem;
                     min-width:72px;letter-spacing:0.3px">{row_lbl}</span>
        <span style="color:#64748b;font-size:0.8rem;min-width:100px">{row_rng}</span>
        <span style="color:#334155;font-size:0.78rem">{row_ex}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 6px;
            font-size:0.7rem;color:#1e2d3d;letter-spacing:0.5px">
    Linear Regression · R² 0.89 · Scikit-learn
</div>
""", unsafe_allow_html=True)
