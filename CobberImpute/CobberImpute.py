import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Load data & flag deleted rows
df_original = pd.read_csv("alkane_dataset.csv")
df_original["is_deleted"] = df_original.isnull().any(axis=1)

# Calculate deletion statistics
n_original = len(df_original)
df_clean = df_original[~df_original["is_deleted"]].copy().reset_index(drop=True)
n_clean = len(df_clean)
n_deleted = n_original - n_clean
pct_deleted = (n_deleted / n_original) * 100

# 2. Specify target variable and predictor features
target_col = "viscosity"
feature_cols = [
    "carbons",
    "molecular weight",
    "thermal conductivity",
    "heat capacity",
    "branch number",
]

X = df_clean[feature_cols]
y = df_clean[target_col]

# 3. Train/Test split & model fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 4. Predictions & MAE calculation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("=== DATASET CLEANING & PREDICTION REPORT ===")
print(f"Original Row Count: {n_original}")
print(f"Rows Retained:      {n_clean}")
print(f"Rows Deleted:       {n_deleted} ({pct_deleted:.1f}%)")
print(f"Target Property:    {target_col}")
print(f"MAE Score:          {mae:.3f}")

# 5. Create 2-Panel Plot: Prediction Quality + Deletion Bias Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Panel A: Actual vs Predicted Evaluation
ax1.scatter(y_test, y_pred, alpha=0.8, color="navy", s=60, label="Predictions")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="crimson",
    linestyle="--",
    linewidth=2,
    label="Ideal Match (1:1)",
)
ax1.set_title(
    f"Prediction Quality: Actual vs. Predicted {target_col.title()}\nMAE: {mae:.3f}",
    fontsize=11,
)
ax1.set_xlabel(f"Actual {target_col.title()}", fontsize=10)
ax1.set_ylabel(f"Predicted {target_col.title()}", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()

# Panel B: Listwise Deletion Bias Plot
retained = df_original[~df_original["is_deleted"]]
deleted = df_original[df_original["is_deleted"]]

ax2.scatter(
    retained["carbons"],
    retained["molecular weight"],
    color="navy",
    s=70,
    alpha=0.8,
    label=f"Retained Rows (n={len(retained)})",
)
ax2.scatter(
    deleted["carbons"],
    deleted["molecular weight"],
    color="crimson",
    s=80,
    marker="x",
    linewidth=2,
    label=f"Deleted Rows (n={len(deleted)})",
)

ax2.set_title(
    f"Listwise Deletion Bias Check (Molecular Size)\n{pct_deleted:.1f}% Data Lost ({n_deleted}/{n_original} Rows)",
    fontsize=11,
)
ax2.set_xlabel("Carbons", fontsize=10)
ax2.set_ylabel("Molecular Weight", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig("prediction_and_bias_plot.png", dpi=300)
plt.show()