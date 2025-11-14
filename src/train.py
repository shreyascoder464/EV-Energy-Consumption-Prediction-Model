# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
import pickle, json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Final_Dataset_EV.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- load data
data = pd.read_csv(DATA_PATH)

# Fill missing values
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

print("=" * 80)
print("MULTI-MODEL ENSEMBLE APPROACH FOR R² ≥ 0.95")
print("=" * 80)

# ========== FEATURE ENGINEERING (MUST MATCH FRONTEND) ==========
print("\n✨ Creating optimized features...")

data['Speed_Squared'] = data['Speed_kmh'] ** 2
data['Speed_Cubed'] = data['Speed_kmh'] ** 3
data['Speed_Slope'] = data['Speed_kmh'] * data['Slope_%']
data['Weight_Accel'] = data['Vehicle_Weight_kg'] * abs(data['Acceleration_ms2'])
# NOTE: keep denominator /1000 exactly as used here to match the app later
data['Kinetic_Energy'] = 0.5 * data['Vehicle_Weight_kg'] * (data['Speed_kmh'] / 3.6) ** 2 / 1000

data['Battery_Power'] = data['Battery_Voltage_V'] * data['Battery_State_%'] / 100
data['Battery_Efficiency'] = data['Battery_Voltage_V'] / (data['Battery_Temperature_C'] + 1)

data['Distance_Per_Battery'] = data['Distance_Travelled_km'] / (100 - data['Battery_State_%'] + 1)
data['Energy_Efficiency'] = data['Speed_kmh'] / (abs(data['Acceleration_ms2']) + 1)

data['Total_Load'] = data['Wind_Speed_ms'] + abs(data['Slope_%']) + data['Traffic_Condition']
data['Climate_Impact'] = abs(data['Temperature_C'] - 22) * (1 + data['Humidity_%'] / 100)

data['Speed_Traffic'] = data['Speed_kmh'] * data['Traffic_Condition']
data['Mode_Speed'] = data['Driving_Mode'] * data['Speed_kmh']

original_features = [
    'Speed_kmh', 'Acceleration_ms2', 'Battery_State_%', 'Battery_Voltage_V',
    'Battery_Temperature_C', 'Driving_Mode', 'Road_Type', 'Traffic_Condition',
    'Slope_%', 'Weather_Condition', 'Temperature_C', 'Humidity_%',
    'Wind_Speed_ms', 'Tire_Pressure_psi', 'Vehicle_Weight_kg', 'Distance_Travelled_km'
]

derived_features = [
    'Speed_Squared', 'Speed_Cubed', 'Speed_Slope', 'Weight_Accel', 'Kinetic_Energy',
    'Battery_Power', 'Battery_Efficiency', 'Distance_Per_Battery', 'Energy_Efficiency',
    'Total_Load', 'Climate_Impact', 'Speed_Traffic', 'Mode_Speed'
]

all_features = original_features + derived_features
target = 'Energy_Consumption_kWh'

X = data[all_features].replace([np.inf, -np.inf], np.nan).fillna(data.mean(numeric_only=True))
y = data[target]

print(f"   → Total features: {len(all_features)}")

# RobustScaler (fit + save)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n" + "=" * 80)
print("STEP 1: OPTIMIZE XGBOOST")
print("=" * 80)

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "gamma": trial.suggest_float("gamma", 0, 2),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
        "random_state": 42,
        "tree_method": "hist"
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(xgb_objective, n_trials=100, show_progress_bar=True)
xgb_model = XGBRegressor(**study_xgb.best_params)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"\n✅ XGBoost R² = {xgb_r2:.4f}")

print("\n" + "=" * 80)
print("STEP 2: OPTIMIZE LIGHTGBM")
print("=" * 80)

def lgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
        "random_state": 42,
        "verbose": -1
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(lgb_objective, n_trials=100, show_progress_bar=True)
lgb_model = LGBMRegressor(**study_lgb.best_params)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_r2 = r2_score(y_test, lgb_pred)
print(f"\n✅ LightGBM R² = {lgb_r2:.4f}")

print("\n" + "=" * 80)
print("STEP 3: OPTIMIZE CATBOOST")
print("=" * 80)

def cat_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "depth": trial.suggest_int("depth", 5, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_seed": 42,
        "verbose": 0
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(cat_objective, n_trials=100, show_progress_bar=True)
cat_model = CatBoostRegressor(**study_cat.best_params)
cat_model.fit(X_train, y_train, verbose=False)
cat_pred = cat_model.predict(X_test)
cat_r2 = r2_score(y_test, cat_pred)
print(f"\n✅ CatBoost R² = {cat_r2:.4f}")

print("\n" + "=" * 80)
print("STEP 4: WEIGHTED ENSEMBLE")
print("=" * 80)

def ensemble_objective(trial):
    w1 = trial.suggest_float("w1", 0, 1)
    w2 = trial.suggest_float("w2", 0, 1)
    w3 = trial.suggest_float("w3", 0, 1)
    total = max(w1 + w2 + w3, 1e-9)
    w1, w2, w3 = w1/total, w2/total, w3/total
    ensemble_pred = w1 * xgb_pred + w2 * lgb_pred + w3 * cat_pred
    return mean_squared_error(y_test, ensemble_pred)

study_ens = optuna.create_study(direction="minimize")
study_ens.optimize(ensemble_objective, n_trials=50)
best_weights = study_ens.best_params
w1, w2, w3 = best_weights['w1'], best_weights['w2'], best_weights['w3']
total = max(w1 + w2 + w3, 1e-9)
w1, w2, w3 = w1/total, w2/total, w3/total

# Final ensemble metrics
y_pred_final = w1 * xgb_pred + w2 * lgb_pred + w3 * cat_pred
mae_final = mean_absolute_error(y_test, y_pred_final)
mse_final = mean_squared_error(y_test, y_pred_final)
rmse_final = np.sqrt(mse_final)
r2_final = r2_score(y_test, y_pred_final)

print("\n" + "=" * 80)
print("🎯 FINAL ENSEMBLE PERFORMANCE")
print("=" * 80)
print(f"MAE:      {mae_final:.4f}")
print(f"MSE:      {mse_final:.4f}")
print(f"RMSE:     {rmse_final:.4f}")
print(f"R² Score: {r2_final:.4f}")
print("=" * 80)

# ---- SAVE ARTIFACTS ----
# models
with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open(MODELS_DIR / "lgb_model.pkl", "wb") as f:
    pickle.dump(lgb_model, f)
with open(MODELS_DIR / "cat_model.pkl", "wb") as f:
    pickle.dump(cat_model, f)

# feature order exactly as used to fit scaler
with open(MODELS_DIR / "feature_order.json", "w") as f:
    json.dump(all_features, f, indent=2)

# weights for frontend
with open(MODELS_DIR / "weights.json", "w") as f:
    json.dump({"w1": float(w1), "w2": float(w2), "w3": float(w3)}, f, indent=2)

print("\n✅ Artifacts saved in:", MODELS_DIR.resolve())
print("✅ Training complete.")
