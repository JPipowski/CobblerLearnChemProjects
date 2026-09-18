import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

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

# 3. Perform Ensemble Model Imputation (Iterative Gradient Boosting Ensemble)
estimator = HistGradientBoostingRegressor(random_state=42)
imputer = IterativeImputer(
    estimator=estimator, max_iter=10, random_state=42
)

df_ensemble_imputed = pd.DataFrame(
    imputer.fit_transform(df_original[num_cols]), columns=num_cols
)

# 4. Compute feature means and percentage bias
mean_orig = df_original[num_cols].mean()
mean_ensemble = df_ensemble_imputed[num_cols].mean()

pct_bias = ((mean_ensemble - mean_orig) / mean_orig) * 100

# 5. Build plot
plt.figure(figsize=(10, 5), dpi=300)
bar_colors = ["navy" if val >= 0 else "crimson" for val in pct_bias]

plt.barh(pct_bias.index, pct_bias, color=bar_colors, edgecolor="black")
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.title(
    "Bias Introduced by Ensemble (Gradient Boosting) Imputation (% Change in Mean)",
    fontsize=12,
)
plt.xlabel("Percentage Shift in Mean (%)", fontsize=10)
plt.ylabel("Property", fontsize=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()

# 6. Save image file for GitHub commit
output_image_path = "ensemble_imputation_bias.png"
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
            "Add Ensemble Model imputation bias plot",
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Successfully committed and pushed image to GitHub.")
except subprocess.CalledProcessError as e:
    print(f"Git execution note: {e}")