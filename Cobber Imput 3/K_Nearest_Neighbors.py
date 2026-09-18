import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error

# 1. Load original data
df_original = pd.read_csv("alkane_dataset.csv")

# 2. Select numerical columns
num_cols = [
    "carbons",
    "molecular weight",
    "boiling point",
    "viscosity",
    "thermal conductivity",
    "heat capacity",
    "branch number",
]

# 3. Perform K-Nearest Neighbors (KNN) Imputation for full dataset
imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = pd.DataFrame(
    imputer.fit_transform(df_original[num_cols]), columns=num_cols
)

# 4. Evaluate Imputation Quality (MAE via Masking Experiment on Complete Cases)
df_complete = df_original[num_cols].dropna().reset_index(drop=True)

mae_scores = {}
actual_vs_imputed = {}

np.random.seed(42)  # For reproducible evaluation

for col in num_cols:
    df_masked = df_complete.copy()
    n_samples = len(df_masked)

    # Artificially mask 20% of complete values to measure true imputation error
    mask_indices = np.random.choice(
        n_samples, size=int(n_samples * 0.2), replace=False
    )
    df_masked.loc[mask_indices, col] = np.nan

    # Fit imputer on masked data
    eval_imputer = KNNImputer(n_neighbors=5)
    imputed_array = eval_imputer.fit_transform(df_masked)
    df_eval_imputed = pd.DataFrame(imputed_array, columns=num_cols)

    # Extract ground truth vs. imputed values
    y_true = df_complete.loc[mask_indices, col]
    y_pred = df_eval_imputed.loc[mask_indices, col]

    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    mae_scores[col] = mae
    actual_vs_imputed[col] = (y_true, y_pred)

print("=== KNN IMPUTATION MAE EVALUATION ===")
for col, score in mae_scores.items():
    print(f"{col:<20}: MAE = {score:.4f}")

# 5. Compute feature means and percentage bias
mean_orig = df_original[num_cols].mean()
mean_knn = df_knn_imputed[num_cols].mean()
pct_bias = ((mean_knn - mean_orig) / mean_orig) * 100

# 6. Build and Save Bias Plot
bias_image_path = "knn_imputation_bias.png"
plt.figure(figsize=(10, 5), dpi=300)
bar_colors = ["navy" if val >= 0 else "crimson" for val in pct_bias]

plt.barh(pct_bias.index, pct_bias, color=bar_colors, edgecolor="black")
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.title("Bias Introduced by KNN Imputation (% Change in Mean)", fontsize=12)
plt.xlabel("Percentage Shift in Mean (%)", fontsize=10)
plt.ylabel("Property", fontsize=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(bias_image_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Bias chart saved to {bias_image_path}")

# 7. Build and Save Prediction Quality Graph
pred_image_path = "knn_prediction_quality.png"
fig, axes = plt.subplots(3, 3, figsize=(12, 10), dpi=300)
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    y_true, y_pred = actual_vs_imputed[col]

    ax.scatter(y_true, y_pred, color="navy", alpha=0.7, edgecolors="k", s=30)

    # 1:1 Identity Line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=1.5,
        label="Ideal Match",
    )

    ax.set_title(
        f"{col.title()}\n(MAE: {mae_scores[col]:.3f})", fontsize=9, weight="bold"
    )
    ax.set_xlabel("Actual Values", fontsize=8)
    ax.set_ylabel("Imputed Values", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

# Remove empty subplots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("KNN Imputation Quality Across Properties", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(pred_image_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Prediction quality chart saved to {pred_image_path}")

# 8. Git Commit and Push Automation
try:
    subprocess.run(
        ["git", "add", bias_image_path, pred_image_path], check=True
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Add KNN bias plot and prediction quality evaluation graphs",
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Successfully committed and pushed all charts to GitHub.")
except subprocess.CalledProcessError as e:
    print(f"Git execution failed: {e}")