
# ⚡ EV Energy Consumption Prediction

A complete **end-to-end machine learning project** that predicts the **electric energy consumption (kWh)** of an Electric Vehicle (EV) trip using real-world driving and environmental parameters.  
Built with a multi-model ensemble of **XGBoost**, **LightGBM**, and **CatBoost**, combined through an optimized weighted average.  
Includes a **Streamlit web app frontend** for interactive trip-wise prediction.

---

## 🚀 Project Overview

This project demonstrates:
- Data preprocessing and robust feature engineering for EV trip data.
- Hyperparameter tuning with **Optuna** for XGBoost, LightGBM, and CatBoost.
- Ensemble modeling for improved prediction accuracy (R² ≥ 0.95).
- Interactive UI built with **Streamlit** for real-time energy consumption estimation.

---

## 🧠 Features

- ⚙️ **Machine Learning Backend:**
  - Auto-tunes XGBoost, LightGBM, and CatBoost regressors.
  - Computes 13 engineered features (speed, kinetic energy, battery power, etc.).
  - Uses weighted ensemble for final energy prediction.
- 🎨 **Frontend (Streamlit App):**
  - Interactive sliders and dropdowns for trip parameters.
  - Real-time prediction and model comparison charts.
  - Visual, user-friendly dashboard with tooltips and metrics.

---

## 🏗️ Folder Structure

```

ev-energy/
├── data/
│   └── Final_Dataset_EV.csv               # Dataset used for model training
├── models/                                # Trained models + scaler (auto-generated)
│   ├── scaler.pkl
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── cat_model.pkl
│   ├── feature_order.json
│   └── weights.json
├── src/
│   └── train.py                           # Training & model saving script
├── app.py                                 # Streamlit frontend application
├── requirements.txt
└── README.md

````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/ev-energy.git
cd ev-energy
````

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Add your dataset

Place your dataset inside the **data/** folder and name it:

```
Final_Dataset_EV.csv
```

### 5️⃣ Train the models

Run the backend training pipeline (this will create all `.pkl` and `.json` files in `/models`):

```bash
python src/train.py
```

### 6️⃣ Launch the Streamlit app

Once training completes successfully:

```bash
streamlit run app.py
```

---

## 💾 Model Artifacts (auto-generated)

| File                 | Description                              |
| -------------------- | ---------------------------------------- |
| `scaler.pkl`         | Feature scaling object (RobustScaler)    |
| `xgb_model.pkl`      | Trained XGBoost model                    |
| `lgb_model.pkl`      | Trained LightGBM model                   |
| `cat_model.pkl`      | Trained CatBoost model                   |
| `feature_order.json` | Column order used for training & scaling |
| `weights.json`       | Ensemble weights (w₁, w₂, w₃)            |

---

## 🧩 Input Features (in the app)

| Feature                  | Description                   |
| ------------------------ | ----------------------------- |
| Speed (km/h)             | Average vehicle speed         |
| Acceleration (m/s²)      | Average acceleration          |
| Slope (%)                | Road slope or incline         |
| Battery State (%)        | Current battery SOC           |
| Battery Voltage (V)      | Pack voltage                  |
| Battery Temperature (°C) | Battery temperature           |
| Driving Mode             | Eco / Normal / Sport          |
| Road Type                | Urban / Highway / Rural       |
| Traffic Condition        | Scale 1–5                     |
| Weather Condition        | Sunny / Rainy / Cloudy / etc. |
| Temperature (°C)         | Ambient temperature           |
| Humidity (%)             | Air humidity                  |
| Wind Speed (m/s)         | Wind resistance               |
| Tire Pressure (psi)      | Tire pressure                 |
| Vehicle Weight (kg)      | Vehicle + load weight         |
| Distance (km)            | Trip distance                 |

---

## 📊 Engineered Features

| Feature                    | Meaning                      |
| -------------------------- | ---------------------------- |
| Speed_Squared, Speed_Cubed | Non-linear speed effects     |
| Speed_Slope                | Interaction of slope × speed |
| Weight_Accel               | Vehicle mass × acceleration  |
| Kinetic_Energy             | Dynamic energy due to motion |
| Battery_Power              | Approx. available power      |
| Battery_Efficiency         | Battery performance ratio    |
| Distance_Per_Battery       | Efficiency per SOC           |
| Energy_Efficiency          | Relative trip efficiency     |
| Total_Load                 | Aggregate external load      |
| Climate_Impact             | Environmental energy loss    |
| Speed_Traffic              | Speed × congestion factor    |
| Mode_Speed                 | Speed × drive mode factor    |

---

## 🧮 Example Output

| Model                | Predicted kWh   |
| -------------------- | --------------- |
| XGBoost              | 12.56           |
| LightGBM             | 12.33           |
| CatBoost             | 12.48           |
| **Ensemble (final)** | **12.45 kWh ⚡** |

The Streamlit app also displays:

* 📊 Bar chart comparing model outputs
* 🎯 Progress bar gauge for consumption level

---

## ☁️ Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Connect your repo and select `app.py` as the entry point.
4. Ensure `requirements.txt` is included.
5. (Optional) Keep `.pkl` files under `models/` for small demos.

---

## 🧠 Tech Stack

* **Python 3.10+**
* **Pandas / NumPy / Scikit-Learn**
* **XGBoost / LightGBM / CatBoost**
* **Optuna** for hyperparameter tuning
* **Streamlit** for the web interface
* **Matplotlib** for visualization

---

## 🧾 License

This project is licensed under the **MIT License** — feel free to use and modify it for learning or research.

---

## 🙌 Acknowledgments

Developed by Shreyas Rayas💡
Inspired by EV analytics, real-world energy modeling, and sustainable mobility research.

---

````

---

## 🔧 Next steps for you
1. Copy the content above into a file named **`README.md`** at the root of your project (`C:\Users\shrey\Downloads\ev energy\README.md`).  
2. Replace `<your-username>` in the GitHub link with your actual username.  
3. Commit & push to GitHub:
   ```bash
   git add README.md
   git commit -m "Add README.md"
   git push
````

4. When you open your repo on GitHub, you’ll see this beautiful formatted README.

---

Would you like me to generate a **README badge section** (e.g., Python version, Streamlit, License, etc.) and GitHub banner for visual appeal?
