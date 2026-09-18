import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# 3. Perform Log-Linear Regression Imputation
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

        # Predict missing values and exponentiate back to original scale
        y_pred_log = model.predict(X_test)
        df_imputed.loc[missing_mask, col] = np.exp(y_pred_log)

# 4. Compute feature means and percentage bias
num_cols = ["carbons"] + target_cols + ["branch number"]
mean_orig = df_original[num_cols].mean()
mean_imputed = df_imputed[num_cols].mean()
pct_bias = ((mean_imputed - mean_orig) / mean_orig) * 100

# 5. Build plot
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

# 6. Save image for GitHub commit
output_image_path = "log_linear_imputation_bias.png"
plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Chart successfully saved to {output_image_path}")