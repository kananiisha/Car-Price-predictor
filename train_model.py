"""
train_model.py
--------------
Standalone training script for Car Price Predictor.
Run this to regenerate car_price_model.pkl and encoders/*.pkl from scratch.

Usage:
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

os.makedirs("encoders", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data from 'car_data.csv' ...")
df = pd.read_csv("car_data.csv")
print(f"  Shape: {df.shape}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
CURRENT_YEAR = 2024
df["Car_Age"] = CURRENT_YEAR - df["Year"]
df.drop(columns=["Car_Name", "Year"], inplace=True)

# ── 3. Encode categoricals ────────────────────────────────────────────────────
encoders = {}
for col, name in [("Fuel_Type",    "le_fuel"),
                  ("Selling_type", "le_seller"),
                  ("Transmission", "le_transmission")]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[name] = le
    print(f"  Encoded '{col}': {list(le.classes_)}")

# ── 4. Train / test split ─────────────────────────────────────────────────────
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# ── 5. Train Gradient Boosting ────────────────────────────────────────────────
print("\nTraining GradientBoostingRegressor ...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("\n── Test-set Metrics ──────────────────────────────────")
print(f"  R²   : {r2:.4f}")
print(f"  RMSE : {rmse:.4f} Lakhs")
print(f"  MAE  : {mae:.4f} Lakhs")
print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 7. Bootstrap prediction interval ─────────────────────────────────────────
residuals = y_test.values - y_pred
rng = np.random.default_rng(42)
bootstrap_ranges = []
for _ in range(1000):
    sample = rng.choice(residuals, size=len(residuals), replace=True)
    q5, q95 = np.percentile(sample, [5, 95])
    bootstrap_ranges.append((q5, q95))

avg_q5  = np.mean([r[0] for r in bootstrap_ranges])
avg_q95 = np.mean([r[1] for r in bootstrap_ranges])
mean_pred = y_pred.mean()
interval_pct = max(abs(avg_q5), abs(avg_q95)) / mean_pred

print(f"\n── Bootstrap Prediction Interval ─────────────────────")
print(f"  90% of predictions fall within ±{interval_pct*100:.1f}% of estimate")

# ── 8. Save artifacts ─────────────────────────────────────────────────────────
joblib.dump(model, "car_price_model.pkl")
print(f"\nSaved model → car_price_model.pkl")

for name, le in encoders.items():
    enc_path = os.path.join("encoders", f"{name}.pkl")
    joblib.dump(le, enc_path)
    print(f"Saved encoder → {enc_path}")

print("\n✅ Training complete. You can now run: streamlit run app.py")
