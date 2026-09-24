import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Load data
df_original = pd.read_csv("../CobberImpute/alkane_dataset.csv")
df_imputed = df_original.copy()

# 2. Complete predictor variables (0 missing values)
predictors = ["carbons", "branch number"]

# Target property columns requiring imputation
target_cols = [
    "molecular weight",
    "boiling point",
    "viscosity",
    "thermal conductivity",
    "heat capacity",
]

# Track MAE scores and predictions for visualization
mae_results = {}
predictions_dict = {}

# 3. Perform Log-Linear Regression Imputation & Measure MAE
for col in target_cols:
    missing_mask = df_imputed[col].isnull()

    if missing_mask.sum() > 0:
        # Fit model on observed complete cases
        train_data = df_imputed[~missing_mask]
        test_data = df_imputed[missing_mask]

        # Log-transform target variable
        y_train_log = np.log(train_data[col])
        X_train = train_data[predictors]
        X_test = test_data[predictors]

        # Fit linear model in log-space
        model = LinearRegression()
        model.fit(X_train, y_train_log)

        # In-sample actual vs. fitted back-transformed predictions for MAE evaluation
        y_fitted_log = model.predict(X_train)
        y_fitted = np.exp(y_fitted_log)
        y_actual = train_data[col]

        # Calculate Mean Absolute Error
        col_mae = mean_absolute_error(y_actual, y_fitted)
        mae_results[col] = col_mae
        predictions_dict[col] = (y_actual, y_fitted)

        # Predict missing values and exponentiate back to original scale
        y_pred_log = model.predict(X_test)
        df_imputed.loc[missing_mask, col] = np.exp(y_pred_log)

# Print MAE metrics report
print("=== LOG-LINEAR IMPUTATION MAE REPORT ===")
for col, score in mae_results.items():
    print(f"{col:<20}: MAE = {score:.4f}")

# 4. Compute feature means and percentage bias
num_cols = ["carbons"] + target_cols + ["branch number"]
mean_orig = df_original[num_cols].mean()
mean_imputed = df_imputed[num_cols].mean()
pct_bias = ((mean_imputed - mean_orig) / mean_orig) * 100

# 5. Build and Save Percentage Bias Plot
plt.figure(figsize=(10, 5), dpi=300)
bar_colors = ["navy" if val >= 0 else "crimson" for val in pct_bias]

plt.barh(pct_bias.index, pct_bias, color=bar_colors, edgecolor="black")
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.title(
    "Bias Introduced by Log-Linear Regression Imputation (% Change in Mean)",
    fontsize=12,
)
plt.xlabel("Percentage Shift in Mean (%)", fontsize=10)
plt.ylabel("Property", fontsize=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()

bias_image_path = "log_linear_imputation_bias.png"
plt.savefig(bias_image_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Bias chart saved to {bias_image_path}")

# 6. Build and Save Prediction Quality Graph
fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
axes = axes.flatten()

for i, col in enumerate(target_cols):
    ax = axes[i]
    y_act, y_fit = predictions_dict[col]

    ax.scatter(y_act, y_fit, alpha=0.7, color="navy", edgecolors="k", s=30)

    # 1:1 perfect fit reference line
    min_val = min(y_act.min(), y_fit.min())
    max_val = max(y_act.max(), y_fit.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=1.5,
        label="Ideal Match",
    )

    ax.set_title(
        f"{col.title()}\n(MAE: {mae_results[col]:.3f})", fontsize=10, weight="bold"
    )
    ax.set_xlabel("Actual Values", fontsize=8)
    ax.set_ylabel("Fitted Values", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

# Remove unused subplots
if len(target_cols) < len(axes):
    for j in range(len(target_cols), len(axes)):
        fig.delaxes(axes[j])

plt.suptitle(
    "Log-Linear Model Prediction Quality Across Properties", fontsize=14, y=1.02
)
plt.tight_layout()

pred_image_path = "log_linear_prediction_quality.png"
plt.savefig(pred_image_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Prediction quality chart saved to {pred_image_path}")


# 7. Git Commit & Push Automation
def push_to_github(files, commit_message="Update bias and prediction quality graphs"):
    try:
        # Stage files
        subprocess.run(["git", "add"] + files, check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push to origin
        subprocess.run(["git", "push"], check=True)
        print("Successfully committed and pushed figures to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")


# Execute Git commands for generated figures
push_to_github([bias_image_path, pred_image_path])