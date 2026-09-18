import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Load data
df_original = pd.read_csv("alkane_dataset.csv")

# 2. Perform listwise deletion
df_clean = df_original.dropna().reset_index(drop=True)

# 3. Specify target variable and predictors
target_col = "boiling point"  # Example target feature
feature_cols = [
    "carbons",
    "molecular weight",
    "viscosity",
    "thermal conductivity",
    "heat capacity",
    "branch number",
]

X = df_clean[feature_cols]
y = df_clean[target_col]

# 4. Train/Test split and model fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 5. Predictions & MAE calculation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("=== PREDICTION METRICS REPORT ===")
print(f"Target Property: {target_col}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")

# 6. Generate and Save Prediction Quality Graph
plt.figure(figsize=(8, 6))

# Plot Actual vs Predicted scatter
plt.scatter(y_test, y_pred, alpha=0.7, color="navy", label="Predictions")

# Plot 1:1 Perfect Prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="crimson",
    linestyle="--",
    linewidth=2,
    label="Ideal Match (1:1)",
)

plt.title(
    f"Prediction Quality: Actual vs. Predicted {target_col.title()}\nMAE: {mae:.3f}",
    fontsize=12,
)
plt.xlabel(f"Actual {target_col.title()}", fontsize=10)
plt.ylabel(f"Predicted {target_col.title()}", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save image file to repository for Git tracking
plt.savefig("prediction_quality.png", dpi=300)
plt.show()