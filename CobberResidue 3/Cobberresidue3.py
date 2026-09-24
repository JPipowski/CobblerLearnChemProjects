import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Output directory for saving generated plots
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Load dataset_positive_deviation.csv
csv_path = "dataset_positive_deviation.csv"
df = pd.read_csv(csv_path)

# Extract actual, predicted, and calculate residuals
y_actual = df["Actual"]
y_pred = df["Predicted"]
residuals = y_actual - y_pred

# 2. Compute Metrics (MAE, MSE, R²)
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE):  {mse:.2f}")
print(f"Coefficient of Determination (R²): {r2:.3f}")

# 3. Create Actual vs. Predicted Plot
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(
    y_actual,
    y_pred,
    alpha=0.5,
    color="navy",
    edgecolors="w",
    s=40,
    label="Data Points",
)

min_val = min(y_actual.min(), y_pred.min())
max_val = max(y_actual.max(), y_pred.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    lw=2,
    label="Ideal Prediction Line ($y = x$)",
)

plt.title(
    f"Actual vs. Predicted Evaluation (MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.3f})",
    fontsize=11,
)
plt.xlabel("Actual Values", fontsize=10)
plt.ylabel("Predicted Values", fontsize=10)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

actual_vs_pred_path = os.path.join(
    output_dir, "actual_vs_predicted_positive_deviation.png"
)
plt.savefig(actual_vs_pred_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Actual vs. Predicted plot to '{actual_vs_pred_path}'")

# 4. Create Residuals Plot (Predicted vs. Residual Error)
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(
    y_pred,
    residuals,
    alpha=0.5,
    color="darkred",
    edgecolors="w",
    s=40,
    label="Residuals",
)
plt.axhline(y=0, color="black", linestyle="--", lw=2, label="Zero Error Line")

plt.title("Residuals Plot (Predicted vs. Residual Error)", fontsize=11)
plt.xlabel("Predicted Values", fontsize=10)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=10)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

residuals_plot_path = os.path.join(
    output_dir, "residuals_plot_positive_deviation.png"
)
plt.savefig(residuals_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Residuals plot to '{residuals_plot_path}'")

# 5. Automatically stage, commit, and push artifacts to GitHub
try:
    subprocess.run(["git", "add", output_dir], check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            f"Add positive deviation Actual vs Predicted & Residuals plots (MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.3f})",
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Successfully committed and pushed all artifacts to GitHub.")
except subprocess.CalledProcessError as e:
    print(f"Git execution note: {e}")