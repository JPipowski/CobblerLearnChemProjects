import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Output directory for saving generated plots
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Define NumPy Arrays
actual = np.array([2, 4, 5, 4, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8.8])

# Calculate residuals (Actual - Predicted)
residuals = actual - predicted

# 2. Compute Metrics using Pure NumPy
mae = np.mean(np.abs(residuals))
mse = np.mean(residuals**2)

# R² = 1 - (SS_res / SS_tot)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((actual - np.mean(actual)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE):  {mse:.3f}")
print(f"Coefficient of Determination (R²): {r2:.3f}")

# 3. Create Actual vs. Predicted Plot
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(
    actual,
    predicted,
    alpha=0.8,
    color="navy",
    edgecolors="w",
    s=60,
    label="Data Points",
)

min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
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

actual_vs_pred_path = os.path.join(output_dir, "actual_vs_predicted_numpy.png")
plt.savefig(actual_vs_pred_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Actual vs. Predicted plot to '{actual_vs_pred_path}'")

# 4. Create Residuals Plot (Predicted vs. Residual Error)
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(
    predicted,
    residuals,
    alpha=0.8,
    color="darkred",
    edgecolors="w",
    s=60,
    label="Residuals",
)
plt.axhline(y=0, color="black", linestyle="--", lw=2, label="Zero Error Line")

plt.title("Residuals Plot (Predicted vs. Residual Error)", fontsize=11)
plt.xlabel("Predicted Values", fontsize=10)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=10)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

residuals_plot_path = os.path.join(output_dir, "residuals_plot_numpy.png")
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
            f"Add NumPy array evaluation & residual plots (MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.3f})",
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Successfully committed and pushed all artifacts to GitHub.")
except subprocess.CalledProcessError as e:
    print(f"Git execution note: {e}")