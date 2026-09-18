import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Parse command line arguments for custom commit message
parser = argparse.ArgumentParser(
    description="Run Random Forest Imputation analysis and commit figures to Git."
)
parser.add_argument(
    "-m",
    "--message",
    type=str,
    default="Add Random Forest bias plot and prediction quality evaluation graphs",
    help="Custom commit message for Git",
)
args = parser.parse_args()
commit_message = args.message

# 1. Load original data
df_original = pd.read_csv("alkane_dataset.csv")

# 2. Select numerical property columns
num_cols = [
    "carbons",
    "molecular weight",
    "boiling point",
    "viscosity",
    "thermal conductivity",
    "heat capacity",
    "branch number",
]

# 3. Perform Random Forest Imputation for full dataset
df_rf_imputed = df_original[num_cols].copy()

complete_cols = [
    col for col in num_cols if df_rf_imputed[col].isnull().sum() == 0
]
incomplete_cols = [
    col for col in num_cols if df_rf_imputed[col].isnull().sum() > 0
]

for col in incomplete_cols:
    missing_mask = df_rf_imputed[col].isnull()

    train_data = df_rf_imputed[~missing_mask]
    test_data = df_rf_imputed[missing_mask]

    X_train = train_data[complete_cols]
    y_train = train_data[col]
    X_test = test_data[complete_cols]

    # Fit Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Impute missing values
    df_rf_imputed.loc[missing_mask, col] = rf.predict(X_test)

# 4. Evaluate Imputation Quality (MAE via Masking Experiment on Complete Cases)
df_complete = df_original[num_cols].dropna().reset_index(drop=True)

mae_scores = {}
actual_vs_imputed = {}

np.random.seed(42)  # For reproducible evaluation

for col in incomplete_cols:
    df_masked = df_complete.copy()
    n_samples = len(df_masked)

    # Artificially mask 20% of complete target values to measure true imputation error
    mask_indices = np.random.choice(
        n_samples, size=int(n_samples * 0.2), replace=False
    )
    df_masked.loc[mask_indices, col] = np.nan

    # Split train/test for evaluation model
    eval_missing_mask = df_masked[col].isnull()
    eval_train = df_masked[~eval_missing_mask]
    eval_test = df_masked[eval_missing_mask]

    X_train_eval = eval_train[complete_cols]
    y_train_eval = eval_train[col]
    X_test_eval = eval_test[complete_cols]

    # Fit and predict
    rf_eval = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_eval.fit(X_train_eval, y_train_eval)

    y_pred = rf_eval.predict(X_test_eval)
    y_true = df_complete.loc[mask_indices, col]

    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    mae_scores[col] = mae
    actual_vs_imputed[col] = (y_true, y_pred)

print("=== RANDOM FOREST IMPUTATION MAE EVALUATION ===")
for col, score in mae_scores.items():
    print(f"{col:<20}: MAE = {score:.4f}")

# 5. Compute feature means and percentage bias
mean_orig = df_original[num_cols].mean()
mean_rf = df_rf_imputed[num_cols].mean()

pct_bias = ((mean_rf - mean_orig) / mean_orig) * 100

# 6. Build and Save Bias Plot
bias_image_path = "rf_imputation_bias.png"
plt.figure(figsize=(10, 5), dpi=300)
bar_colors = ["navy" if val >= 0 else "crimson" for val in pct_bias]

plt.barh(pct_bias.index, pct_bias, color=bar_colors, edgecolor="black")
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.title(
    "Bias Introduced by Random Forest Imputation (% Change in Mean)",
    fontsize=12,
)
plt.xlabel("Percentage Shift in Mean (%)", fontsize=10)
plt.ylabel("Property", fontsize=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()

plt.savefig(bias_image_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Bias chart saved to {bias_image_path}")

# 7. Build and Save Prediction Quality Graph
pred_image_path = "rf_prediction_quality.png"
n_plots = len(incomplete_cols)

fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
axes = axes.flatten()

for i, col in enumerate(incomplete_cols):
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
        f"{col.title()}\n(MAE: {mae_scores[col]:.3f})", fontsize=10, weight="bold"
    )
    ax.set_xlabel("Actual Values", fontsize=8)
    ax.set_ylabel("Imputed Values", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

# Remove unused subplot axes
for j in range(n_plots, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(
    "Random Forest Imputation Quality Across Properties", fontsize=14, y=1.02
)
plt.tight_layout()

# Display plot in interactive window before closing
plt.savefig(pred_image_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Prediction quality chart saved to {pred_image_path}")

# 8. Git Commit and Push Automation with Custom Message
try:
    subprocess.run(
        ["git", "add", bias_image_path, pred_image_path], check=True
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            commit_message,
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print(
        f"Successfully committed with message '{commit_message}' and pushed both images to GitHub."
    )
except subprocess.CalledProcessError as e:
    print(f"Git execution note: {e}")