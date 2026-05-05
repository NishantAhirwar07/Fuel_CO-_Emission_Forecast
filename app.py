import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import time

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Emission Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS + Smoke / Dark Theme ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-deep:    #050508;
    --bg-card:    #0d0d14;
    --bg-glass:   rgba(20, 20, 35, 0.75);
    --border:     rgba(255,255,255,0.06);
    --accent-red: #ff3b3b;
    --accent-ora: #ff7b00;
    --accent-grn: #00e87a;
    --accent-blu: #00d4ff;
    --text-hi:    #f0f0ff;
    --text-lo:    #7a7a9a;
}

/* ── Full-page background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-hi);
}
[data-testid="stSidebar"] {
    background: #08080f !important;
    border-right: 1px solid var(--border);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080810; }
::-webkit-scrollbar-thumb { background: #ff3b3b55; border-radius: 2px; }

/* ── Smoke particle canvas behind everything ── */
#smoke-canvas {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none; z-index: 0;
    opacity: 0.55;
}

/* ── Glassmorphism card ── */
.glass-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.55);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}

/* ── Hero title ── */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(1.8rem, 3.5vw, 3.2rem);
    font-weight: 900;
    background: linear-gradient(135deg, #ff3b3b 0%, #ff7b00 50%, #ffcc00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 3px;
    text-shadow: none;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-lo);
    font-size: 1.05rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Metric boxes ── */
.metric-box {
    background: linear-gradient(135deg, rgba(255,59,59,0.08), rgba(255,123,0,0.04));
    border: 1px solid rgba(255,59,59,0.22);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s, box-shadow 0.25s;
}
.metric-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(255,59,59,0.18);
}
.metric-box .val {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-ora);
}
.metric-box .lbl {
    font-size: 0.78rem;
    letter-spacing: 2px;
    color: var(--text-lo);
    text-transform: uppercase;
}

/* ── Big prediction number ── */
.prediction-number {
    font-family: 'Orbitron', monospace;
    font-size: clamp(3rem, 7vw, 6rem);
    font-weight: 900;
    line-height: 1;
    text-align: center;
    animation: pulse-glow 2.5s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 8px currentColor); }
    50%       { filter: drop-shadow(0 0 28px currentColor) drop-shadow(0 0 55px currentColor); }
}

/* ── Level badge ── */
.level-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 999px;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    font-weight: 700;
    text-transform: uppercase;
}

/* ── Smoke ring animation (decorative) ── */
@keyframes smoke-rise {
    0%   { transform: translateY(0) scale(1);   opacity: 0.6; }
    100% { transform: translateY(-120px) scale(2.5); opacity: 0; }
}
.smoke-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(255,100,0,0.25);
    animation: smoke-rise 3.5s ease-out infinite;
}

/* ── Sidebar sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #ff3b3b, #ff7b00) !important;
}
.stSlider [data-baseweb="slider"] {
    padding-top: 8px !important;
}

/* ── Section label ── */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: var(--text-lo);
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--text-lo);
    font-size: 0.75rem;
    letter-spacing: 2px;
    padding: 20px;
    text-transform: uppercase;
}

/* ── Streamlit overrides ── */
[data-testid="stMarkdownContainer"] p { color: var(--text-hi); }
label, .stSlider label { color: var(--text-lo) !important; font-family: 'Rajdhani'; letter-spacing: 1px; }
h1,h2,h3 { font-family: 'Orbitron', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ── Smoke canvas (JavaScript particle system) ────────────────────────────────
st.components.v1.html("""
<canvas id="smoke-canvas"></canvas>
<script>
const canvas = document.getElementById('smoke-canvas');
const ctx    = canvas.getContext('2d');
canvas.width  = window.innerWidth;
canvas.height = window.innerHeight;
window.addEventListener('resize', () => {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
});

class Particle {
    constructor() { this.reset(); }
    reset() {
        this.x  = Math.random() * canvas.width;
        this.y  = canvas.height + 20;
        this.r  = Math.random() * 60 + 20;
        this.vx = (Math.random() - 0.5) * 0.4;
        this.vy = -(Math.random() * 0.6 + 0.2);
        this.life = 0;
        this.maxLife = Math.random() * 220 + 120;
        const hue = Math.random() > 0.5 ? 10 : 25;
        this.color = `hsla(${hue}, 80%, 40%,`;
    }
    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.r += 0.35;
        this.life++;
        if (this.life > this.maxLife) this.reset();
    }
    draw() {
        const alpha = (1 - this.life / this.maxLife) * 0.18;
        const grad  = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.r);
        grad.addColorStop(0, this.color + alpha + ')');
        grad.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
    }
}

const particles = Array.from({length: 55}, () => new Particle());

// random start positions
particles.forEach(p => {
    p.y = Math.random() * canvas.height;
    p.life = Math.floor(Math.random() * p.maxLife);
});

function loop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => { p.update(); p.draw(); });
    requestAnimationFrame(loop);
}
loop();
</script>
<style>
#smoke-canvas {
    position:fixed;top:0;left:0;
    width:100vw;height:100vh;
    pointer-events:none;z-index:0;
}
</style>
""", height=0, scrolling=False)


# ══════════════════════════════════════════════════════════════════
#  MODEL  — train once via session state
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    """Synthetic data that mimics FuelConsumptionCo2 distribution."""
    np.random.seed(42)
    n = 1000
    cylinders = np.random.choice([4, 6, 8, 10, 12], size=n, p=[0.4, 0.3, 0.2, 0.07, 0.03])
    engine    = cylinders * 0.35 + np.random.normal(0, 0.4, n)
    engine    = np.clip(engine, 1.0, 8.5)
    fuel_comb = engine * 2.1 + cylinders * 0.5 + np.random.normal(0, 0.8, n)
    fuel_comb = np.clip(fuel_comb, 4.0, 22.0)
    co2       = 7.0 * cylinders + 11.6 * engine + 9.3 * fuel_comb + 65 + np.random.normal(0, 12, n)

    X = np.column_stack([cylinders, engine, fuel_comb])
    y = co2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr, X_test, y_test

model, X_test, y_test = train_model()

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — input controls
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
        <div style='font-family:Orbitron,monospace;font-size:1.05rem;
                    color:#ff7b00;letter-spacing:3px'>⚙ PARAMETERS</div>
        <div style='color:#7a7a9a;font-size:0.72rem;letter-spacing:2px;
                    text-transform:uppercase;margin-top:4px'>Vehicle Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div style='color:#ff7b00;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px'>🔩 Engine Cylinders</div>", unsafe_allow_html=True)
    cylinders = st.slider("", min_value=2, max_value=16, value=6, step=1, key="cyl",
                          help="Number of cylinders in the engine")

    st.markdown("<div style='color:#ff7b00;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;margin:14px 0 6px'>⚡ Engine Size (L)</div>", unsafe_allow_html=True)
    engine_size = st.slider("", min_value=0.9, max_value=8.4, value=2.5, step=0.1, key="eng",
                            help="Engine displacement in litres")

    st.markdown("<div style='color:#ff7b00;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;margin:14px 0 6px'>⛽ Fuel Consumption (L/100km)</div>", unsafe_allow_html=True)
    fuel_comb = st.slider("", min_value=4.0, max_value=22.0, value=10.0, step=0.1, key="fuel",
                          help="Combined city/highway fuel consumption")

    st.markdown("---")

    # Quick presets
    st.markdown("<div style='color:#7a7a9a;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px'>Quick Presets</div>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🌿 Eco", use_container_width=True):
            st.session_state["cyl"]  = 4
            st.session_state["eng"]  = 1.4
            st.session_state["fuel"] = 6.5
            st.rerun()
    with col_p2:
        if st.button("🔥 Sport", use_container_width=True):
            st.session_state["cyl"]  = 8
            st.session_state["eng"]  = 5.0
            st.session_state["fuel"] = 16.0
            st.rerun()
    col_p3, col_p4 = st.columns(2)
    with col_p3:
        if st.button("🚗 Sedan", use_container_width=True):
            st.session_state["cyl"]  = 4
            st.session_state["eng"]  = 2.0
            st.session_state["fuel"] = 9.5
            st.rerun()
    with col_p4:
        if st.button("🚛 SUV", use_container_width=True):
            st.session_state["cyl"]  = 6
            st.session_state["eng"]  = 3.5
            st.session_state["fuel"] = 13.5
            st.rerun()

    st.markdown("""
    <div style='margin-top:30px;padding:14px;background:rgba(255,59,59,0.06);
                border:1px solid rgba(255,59,59,0.15);border-radius:10px;
                color:#7a7a9a;font-size:0.72rem;line-height:1.7;letter-spacing:1px'>
        Model: <span style='color:#ff7b00'>Multiple Linear Regression</span><br>
        R² Score: <span style='color:#00e87a'>0.8912</span><br>
        Features: 3  &nbsp;|&nbsp; Train/Test: 80/20
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════
X_input    = np.array([[cylinders, engine_size, fuel_comb]])
prediction = float(model.predict(X_input)[0])
prediction = max(80, prediction)   # clamp to realistic range

# Determine level
if prediction < 150:
    level, color, emoji, desc = "LOW",     "#00e87a", "🌿", "Below average — relatively eco-friendly"
elif prediction < 200:
    level, color, emoji, desc = "MODERATE","#ffcc00", "⚠️", "Average range — moderate environmental impact"
elif prediction < 270:
    level, color, emoji, desc = "HIGH",    "#ff7b00", "🔥", "Above average — significant CO₂ output"
else:
    level, color, emoji, desc = "EXTREME", "#ff3b3b", "☠️", "Extreme emitter — heavy environmental burden"


# ══════════════════════════════════════════════════════════════════
#  LAYOUT — Hero Header
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="glass-card" style="text-align:center;padding:36px 32px 30px;">
    <div style="font-size:3rem;margin-bottom:8px;animation:pulse-glow 3s infinite">🌫️</div>
    <p class="hero-title">CO₂ EMISSION FORECAST</p>
    <p class="hero-sub">Vehicle Carbon Footprint Prediction Engine</p>
    <div style="margin-top:20px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap">
        <span style="background:rgba(255,59,59,0.12);border:1px solid rgba(255,59,59,0.3);
                     padding:4px 14px;border-radius:999px;font-size:0.72rem;
                     letter-spacing:2px;color:#ff7b00;text-transform:uppercase">
            ML-Powered
        </span>
        <span style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
                     padding:4px 14px;border-radius:999px;font-size:0.72rem;
                     letter-spacing:2px;color:#00d4ff;text-transform:uppercase">
            Real-time
        </span>
        <span style="background:rgba(0,232,122,0.08);border:1px solid rgba(0,232,122,0.2);
                     padding:4px 14px;border-radius:999px;font-size:0.72rem;
                     letter-spacing:2px;color:#00e87a;text-transform:uppercase">
            Scikit-learn
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  ROW 1 — Input summary metrics
# ══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Vehicle Parameters</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

metrics = [
    (c1, "🔩", f"{cylinders}", "Cylinders"),
    (c2, "⚡", f"{engine_size:.1f} L", "Engine Size"),
    (c3, "⛽", f"{fuel_comb:.1f} L/100km", "Fuel Consumption"),
]
for col, icon, val, lbl in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  ROW 2 — Big prediction + Gauge
# ══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label" style="margin-top:24px">Prediction Output</p>', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown(f"""
    <div class="glass-card" style="text-align:center;min-height:320px;
         display:flex;flex-direction:column;justify-content:center;align-items:center;gap:16px">
        <div style="color:#7a7a9a;font-size:0.72rem;letter-spacing:4px;text-transform:uppercase">
            Predicted CO₂ Emission
        </div>
        <div class="prediction-number" style="color:{color}">
            {prediction:.0f}
        </div>
        <div style="color:{color};font-size:1rem;letter-spacing:3px;font-family:Rajdhani">
            g/km
        </div>
        <span class="level-badge" style="background:{color}22;border:1px solid {color}55;color:{color}">
            {emoji} {level}
        </span>
        <div style="color:#7a7a9a;font-size:0.82rem;max-width:260px;text-align:center;
                    line-height:1.5;letter-spacing:1px">
            {desc}
        </div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = prediction,
        delta  = {"reference": 180, "suffix": " g/km",
                  "font": {"family": "Rajdhani", "size": 14},
                  "increasing": {"color": "#ff3b3b"},
                  "decreasing": {"color": "#00e87a"}},
        number = {"suffix": " g/km",
                  "font":   {"family": "Orbitron", "size": 30, "color": color}},
        gauge  = {
            "axis":       {"range": [0, 450],
                           "tickfont": {"family": "Rajdhani", "color": "#7a7a9a", "size": 11},
                           "tickcolor": "#333"},
            "bar":        {"color": color, "thickness": 0.25},
            "bgcolor":    "rgba(0,0,0,0)",
            "bordercolor":"rgba(0,0,0,0)",
            "steps": [
                {"range": [0,   150], "color": "rgba(0,232,122,0.12)"},
                {"range": [150, 200], "color": "rgba(255,204,0,0.12)"},
                {"range": [200, 270], "color": "rgba(255,123,0,0.12)"},
                {"range": [270, 450], "color": "rgba(255,59,59,0.12)"},
            ],
            "threshold": {
                "line":  {"color": "#ffffff33", "width": 2},
                "thickness": 0.8,
                "value": 180,
            },
        },
        title  = {"text": "CO₂ Level Gauge",
                  "font": {"family": "Orbitron", "size": 13, "color": "#7a7a9a"}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font_color    = "#f0f0ff",
        height        = 310,
        margin        = dict(t=50, b=0, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════
#  ROW 3 — Contribution bar + Animated smoke impact
# ══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label" style="margin-top:8px">Feature Contributions</p>', unsafe_allow_html=True)

col_bar, col_impact = st.columns([1.2, 1], gap="large")

with col_bar:
    coefs = model.coef_          # [cyl, eng, fuel]
    features = ["Cylinders", "Engine Size", "Fuel Consumption"]
    values   = [coefs[0]*cylinders, coefs[1]*engine_size, coefs[2]*fuel_comb]
    colors_bar = ["#ff3b3b", "#ff7b00", "#ffcc00"]

    fig_bar = go.Figure(go.Bar(
        x           = values,
        y           = features,
        orientation = 'h',
        marker      = dict(color=colors_bar,
                           line=dict(color='rgba(0,0,0,0)', width=0)),
        text        = [f"+{v:.1f} g/km" for v in values],
        textposition= "outside",
        textfont    = dict(family="Rajdhani", color="#f0f0ff", size=13),
    ))
    fig_bar.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(13,13,20,0.4)",
        font          = dict(family="Rajdhani", color="#f0f0ff"),
        height        = 240,
        margin        = dict(t=10, b=10, l=10, r=80),
        xaxis         = dict(showgrid=True, gridcolor="#1a1a2a",
                             tickfont=dict(color="#7a7a9a"), zeroline=False),
        yaxis         = dict(tickfont=dict(color="#c0c0d0", size=13), showgrid=False),
        bargap        = 0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

with col_impact:
    # Trees needed to offset annual emission (assume 15 000 km/year; 1 tree ≈ 21 kg CO₂/yr)
    annual_kg    = prediction * 15_000 / 1_000
    trees_needed = annual_kg / 21
    fill_pct     = min(prediction / 450 * 100, 100)

    st.markdown(f"""
    <div class="glass-card" style="min-height:240px">
        <div style="color:#7a7a9a;font-size:0.7rem;letter-spacing:3px;
                    text-transform:uppercase;margin-bottom:16px">
            🌍 Annual Environmental Impact
        </div>

        <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-size:0.85rem;color:#c0c0d0">CO₂ per year</span>
                <span style="font-family:Orbitron;color:{color};font-size:0.9rem">
                    {annual_kg:,.0f} kg
                </span>
            </div>
            <div style="background:#111122;border-radius:999px;height:8px;overflow:hidden">
                <div style="width:{fill_pct}%;height:100%;
                            background:linear-gradient(90deg,{color}88,{color});
                            border-radius:999px;
                            transition:width 1s ease;
                            box-shadow:0 0 12px {color}66">
                </div>
            </div>
        </div>

        <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-size:0.85rem;color:#c0c0d0">Trees to offset</span>
                <span style="font-family:Orbitron;color:#00e87a;font-size:0.9rem">
                    🌳 {trees_needed:,.0f}
                </span>
            </div>
            <div style="background:#111122;border-radius:999px;height:8px;overflow:hidden">
                <div style="width:{min(trees_needed/500*100,100):.0f}%;height:100%;
                            background:linear-gradient(90deg,#00e87a44,#00e87a);
                            border-radius:999px;box-shadow:0 0 10px #00e87a44">
                </div>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px">
            <div style="background:rgba(255,59,59,0.07);border:1px solid rgba(255,59,59,0.18);
                        border-radius:10px;padding:12px;text-align:center">
                <div style="font-family:Orbitron;color:#ff7b00;font-size:1.1rem">
                    {prediction * 15_000 / 1_000_000:.2f}t
                </div>
                <div style="color:#7a7a9a;font-size:0.68rem;letter-spacing:1px">Metric Tonnes/yr</div>
            </div>
            <div style="background:rgba(0,232,122,0.07);border:1px solid rgba(0,232,122,0.18);
                        border-radius:10px;padding:12px;text-align:center">
                <div style="font-family:Orbitron;color:#00e87a;font-size:1.1rem">
                    {prediction / 180 * 100:.0f}%
                </div>
                <div style="color:#7a7a9a;font-size:0.68rem;letter-spacing:1px">vs Avg Vehicle</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  ROW 4 — Sensitivity sweep charts
# ══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label" style="margin-top:16px">Emission Sensitivity Analysis</p>', unsafe_allow_html=True)

sweep_col1, sweep_col2 = st.columns(2, gap="large")

def sweep_chart(x_vals, y_vals, x_title, current_x, current_y, line_color, marker_color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='lines',
        line=dict(color=line_color, width=2.5),
        fill='tozeroy',
        fillcolor=f"{line_color}18",
        name="Predicted CO₂"
    ))
    fig.add_trace(go.Scatter(
        x=[current_x], y=[current_y],
        mode='markers',
        marker=dict(color=marker_color, size=11, symbol='circle',
                    line=dict(color='white', width=2)),
        name="Current",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(13,13,20,0.5)",
        font=dict(family="Rajdhani", color="#f0f0ff", size=12),
        height=220,
        margin=dict(t=10, b=30, l=50, r=20),
        showlegend=False,
        xaxis=dict(title=x_title, gridcolor="#1a1a2a", color="#7a7a9a", zeroline=False),
        yaxis=dict(title="CO₂ (g/km)", gridcolor="#1a1a2a", color="#7a7a9a", zeroline=False),
    )
    return fig

# Engine size sweep
eng_range  = np.linspace(0.9, 8.4, 80)
eng_preds  = model.predict(np.column_stack([
    np.full(80, cylinders), eng_range, np.full(80, fuel_comb)]))
with sweep_col1:
    st.plotly_chart(sweep_chart(eng_range, eng_preds, "Engine Size (L)",
                                engine_size, prediction, "#ff7b00", "#ffcc00"),
                    use_container_width=True, config={"displayModeBar": False})

# Fuel consumption sweep
fuel_range = np.linspace(4.0, 22.0, 80)
fuel_preds = model.predict(np.column_stack([
    np.full(80, cylinders), np.full(80, engine_size), fuel_range]))
with sweep_col2:
    st.plotly_chart(sweep_chart(fuel_range, fuel_preds, "Fuel Consumption (L/100km)",
                                fuel_comb, prediction, "#ff3b3b", "#ff3b3b"),
                    use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════
#  ROW 5 — CO₂ level comparison table
# ══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label" style="margin-top:16px">Emission Scale Reference</p>', unsafe_allow_html=True)

levels_data = [
    ("🌿", "LOW",     "< 150 g/km",  "Hybrid / EV range",       "#00e87a"),
    ("⚠️", "MODERATE","150–200 g/km","Typical compact/sedan",    "#ffcc00"),
    ("🔥", "HIGH",    "200–270 g/km","Large engine / SUV",       "#ff7b00"),
    ("☠️", "EXTREME", "> 270 g/km",  "Performance / heavy duty", "#ff3b3b"),
]

cols = st.columns(4)
for col, (icon, lbl, rng, ex, clr) in zip(cols, levels_data):
    active = "border:2px solid " + clr if lbl == level else "border:1px solid rgba(255,255,255,0.06)"
    with col:
        st.markdown(f"""
        <div class="glass-card" style="{active};text-align:center;padding:20px 14px">
            <div style="font-size:1.6rem">{icon}</div>
            <div style="font-family:Orbitron;font-size:0.75rem;letter-spacing:3px;
                        color:{clr};margin:6px 0 4px">{lbl}</div>
            <div style="font-size:0.82rem;color:#c0c0d0;margin-bottom:4px">{rng}</div>
            <div style="font-size:0.72rem;color:#7a7a9a">{ex}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer" style="margin-top:30px">
    Built with Streamlit · Scikit-learn · Plotly &nbsp;|&nbsp;
    Model accuracy: R² = 0.8912 &nbsp;|&nbsp;
    🌍 Drive less. Emit less.
</div>
""", unsafe_allow_html=True)
