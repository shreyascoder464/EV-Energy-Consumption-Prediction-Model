# app.py
import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_TITLE = "⚡ EV Energy Consumption Predictor"
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

SCALER = MODELS / "scaler.pkl"
XGB = MODELS / "xgb_model.pkl"
LGB = MODELS / "lgb_model.pkl"
CAT = MODELS / "cat_model.pkl"
FEAT_ORDER = MODELS / "feature_order.json"
WEIGHTS = MODELS / "weights.json"

st.set_page_config(page_title=APP_TITLE, page_icon="⚡", layout="wide")
st.title(APP_TITLE)
st.caption("Predict kWh using XGBoost, LightGBM, CatBoost and a weighted ensemble.")

# ---------- load helpers ----------
@st.cache_resource
def _load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def _try(path: Path, label: str):
    if not path.exists():
        st.sidebar.warning(f"Missing {label}: {path.name}")
        return None
    try:
        obj = _load_pickle(path)
        st.sidebar.success(f"Loaded {label}")
        return obj
    except Exception as e:
        st.sidebar.error(f"Failed to load {label}: {e}")
        return None

def _load_json(p: Path, default=None):
    if not p.exists(): return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

# ---------- models ----------
with st.sidebar:
    st.header("📦 Artifacts")
    scaler = _try(SCALER, "scaler.pkl")
    xgb = _try(XGB, "xgb_model.pkl")
    lgb = _try(LGB, "lgb_model.pkl")
    cat = _try(CAT, "cat_model.pkl")
    feat_order = _load_json(FEAT_ORDER, default=None)
    weights = _load_json(WEIGHTS, default={"w1": 1.0, "w2": 1.0, "w3": 1.0})

    st.markdown("---")
    st.header("🧮 Weights")
    w1 = st.number_input("XGBoost (w1)", 0.0, 5.0, float(weights.get("w1", 1.0)), 0.1)
    w2 = st.number_input("LightGBM (w2)", 0.0, 5.0, float(weights.get("w2", 1.0)), 0.1)
    w3 = st.number_input("CatBoost (w3)", 0.0, 5.0, float(weights.get("w3", 1.0)), 0.1)

st.info("Tips: Higher **slope**/**acceleration** → higher consumption. Temperature far from ~22°C also hurts efficiency.")

# ---------- inputs (numeric encodings to match your training) ----------
with st.form("trip"):
    st.subheader("🧭 Trip & Vehicle Inputs (use numeric codes as in your dataset)")
    c1, c2, c3 = st.columns(3)
    with c1:
        Speed_kmh = st.slider("Speed_kmh", 0, 200, 60)
        Acceleration_ms2 = st.number_input("Acceleration_ms2", 0.0, 5.0, 1.0, 0.1)
        Slope_perc = st.number_input("Slope_%", -20.0, 20.0, 0.0, 0.5)
        Distance_Travelled_km = st.number_input("Distance_Travelled_km", 0.1, 2000.0, 10.0, 0.1)
        Tire_Pressure_psi = st.number_input("Tire_Pressure_psi", 20.0, 60.0, 35.0, 0.5)
    with c2:
        Battery_State_perc = st.slider("Battery_State_%", 1, 100, 80)
        Battery_Voltage_V = st.number_input("Battery_Voltage_V", 150.0, 1000.0, 380.0, 1.0)
        Battery_Temperature_C = st.number_input("Battery_Temperature_C", -40.0, 80.0, 30.0, 0.5)
        Vehicle_Weight_kg = st.number_input("Vehicle_Weight_kg", 600.0, 4000.0, 1500.0, 10.0)
        Wind_Speed_ms = st.number_input("Wind_Speed_ms", 0.0, 30.0, 5.0, 0.1)
    with c3:
        Temperature_C = st.number_input("Temperature_C", -40.0, 60.0, 25.0, 0.5)
        Humidity_perc = st.slider("Humidity_%", 0, 100, 50)
        Traffic_Condition = st.slider("Traffic_Condition (1-5)", 1, 5, 2)
        Driving_Mode = st.number_input("Driving_Mode (code)", 0, 5, 1)     # keep as numeric code
        Road_Type = st.number_input("Road_Type (code)", 0, 5, 1)          # keep as numeric code
        Weather_Condition = st.number_input("Weather_Condition (code)", 0, 10, 2)  # keep as numeric code

    submit = st.form_submit_button("🚀 Predict Energy Consumption", use_container_width=True)

def engineer_features(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])
    # Derived features — EXACTLY matching train.py
    df['Speed_Squared'] = df['Speed_kmh'] ** 2
    df['Speed_Cubed'] = df['Speed_kmh'] ** 3
    df['Speed_Slope'] = df['Speed_kmh'] * df['Slope_%']
    df['Weight_Accel'] = df['Vehicle_Weight_kg'] * abs(df['Acceleration_ms2'])
    df['Kinetic_Energy'] = 0.5 * df['Vehicle_Weight_kg'] * (df['Speed_kmh'] / 3.6) ** 2 / 1000
    df['Battery_Power'] = df['Battery_Voltage_V'] * df['Battery_State_%'] / 100
    df['Battery_Efficiency'] = df['Battery_Voltage_V'] / (df['Battery_Temperature_C'] + 1)
    df['Distance_Per_Battery'] = df['Distance_Travelled_km'] / (100 - df['Battery_State_%'] + 1)
    df['Energy_Efficiency'] = df['Speed_kmh'] / (abs(df['Acceleration_ms2']) + 1)
    df['Total_Load'] = df['Wind_Speed_ms'] + abs(df['Slope_%']) + df['Traffic_Condition']
    df['Climate_Impact'] = abs(df['Temperature_C'] - 22) * (1 + df['Humidity_%'] / 100)
    df['Speed_Traffic'] = df['Speed_kmh'] * df['Traffic_Condition']
    df['Mode_Speed'] = df['Driving_Mode'] * df['Speed_kmh']
    return df

if submit:
    raw = {
        'Speed_kmh': float(Speed_kmh),
        'Acceleration_ms2': float(Acceleration_ms2),
        'Battery_State_%': float(Battery_State_perc),
        'Battery_Voltage_V': float(Battery_Voltage_V),
        'Battery_Temperature_C': float(Battery_Temperature_C),
        'Driving_Mode': float(Driving_Mode),
        'Road_Type': float(Road_Type),
        'Traffic_Condition': float(Traffic_Condition),
        'Slope_%': float(Slope_perc),
        'Weather_Condition': float(Weather_Condition),
        'Temperature_C': float(Temperature_C),
        'Humidity_%': float(Humidity_perc),
        'Wind_Speed_ms': float(Wind_Speed_ms),
        'Tire_Pressure_psi': float(Tire_Pressure_psi),
        'Vehicle_Weight_kg': float(Vehicle_Weight_kg),
        'Distance_Travelled_km': float(Distance_Travelled_km),
    }

    feats = engineer_features(raw)

    # Order columns per training
    if feat_order is None:
        st.error("feature_order.json not found in /models. Run training first.")
        st.stop()
    feats = feats.reindex(columns=feat_order, fill_value=0.0)

    # Scale
    if scaler is None:
        st.error("scaler.pkl missing. Run training first.")
        st.stop()
    X = scaler.transform(feats)

    # Predict
    preds = {}
    errs = {}

    def predict(model, name):
        if model is None:
            errs[name] = f"{name} model not loaded."
            return None
        try:
            y = model.predict(X)
            return float(y[0])
        except Exception as e:
            errs[name] = str(e)
            return None

    px = predict(xgb, "XGBoost")
    pl = predict(lgb, "LightGBM")
    pc = predict(cat, "CatBoost")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("XGBoost (kWh)", "—" if px is None else f"{px:.3f}")
    with c2: st.metric("LightGBM (kWh)", "—" if pl is None else f"{pl:.3f}")
    with c3: st.metric("CatBoost (kWh)", "—" if pc is None else f"{pc:.3f}")
    for k, v in errs.items():
        st.caption(f"⚠️ {k}: {v}")

    # Ensemble
    used = [(px, w1), (pl, w2), (pc, w3)]
    used = [(p, w) for p, w in used if p is not None and w > 0]
    total_w = sum(w for _, w in used) or 1.0
    ensemble = sum(p * w for p, w in used) / total_w

    st.success(f"### ✅ Final Ensemble Prediction: **{ensemble:.3f} kWh**")

    # Bar chart
    labels, values = [], []
    if px is not None: labels.append("XGBoost"); values.append(px)
    if pl is not None: labels.append("LightGBM"); values.append(pl)
    if pc is not None: labels.append("CatBoost"); values.append(pc)
    labels.append("Ensemble"); values.append(ensemble)

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("kWh")
    ax.set_title("Model Outputs")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    st.pyplot(fig)
else:
    st.info("Fill inputs and click **Predict Energy Consumption**.")
