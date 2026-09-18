import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

# 3. Perform Random Forest Imputation
df_rf_imputed = df_original[num_cols].copy()

complete_cols = [col for col in num_cols if df_rf_imputed[col].isnull().sum() == 0]
incomplete_cols = [col for col in num_cols if df_rf_imputed[col].isnull().sum() > 0]

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

# 4. Compute feature means and percentage bias
mean_orig = df_original[num_cols].mean()
mean_rf = df_rf_imputed[num_cols].mean()

pct_bias = ((mean_rf - mean_orig) / mean_orig) * 100

# 5. Build plot
plt.figure(figsize=(10, 5), dpi=300)
bar_colors = ["navy" if val >= 0 else "crimson" for val in pct_bias]

plt.barh(pct_bias.index, pct_bias, color=bar_colors, edgecolor="black")
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.title(
    "Bias Introduced by Random Forest Imputation (% Change in Mean)", fontsize=12
)
plt.xlabel("Percentage Shift in Mean (%)", fontsize=10)
plt.ylabel("Property", fontsize=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()

# 6. Save image file for GitHub commit
output_image_path = "rf_imputation_bias.png"
plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Chart successfully saved to {output_image_path}")

# 7. Automatically stage, commit, and push to GitHub
try:
    subprocess.run(["git", "add", output_image_path], check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Add Random Forest imputation bias plot",
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Successfully committed and pushed image to GitHub.")
except subprocess.CalledProcessError as e:
    print(f"Git execution note: {e}")