import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer

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

# 3. Perform KNN Imputation
imputer = KNNImputer(n_neighbors=5)

df_knn_imputed = pd.DataFrame(
    imputer.fit_transform(df_original[num_cols]), columns=num_cols
)

# 4. Compute feature means and percentage bias
mean_orig = df_original[num_cols].mean()
mean_knn = df_knn_imputed[num_cols].mean()

pct_bias = ((mean_knn - mean_orig) / mean_orig) * 100

# 5. Build and Save Bias Plot
plt.figure(figsize=(10, 5), dpi=300)
bar_colors = ["navy" if val >= 0 else "crimson" for val in pct_bias]

plt.barh(pct_bias.index, pct_bias, color=bar_colors, edgecolor="black")
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.title(
    "Bias Introduced by KNN Imputation (% Change in Mean)",
    fontsize=12,
)
plt.xlabel("Percentage Shift in Mean (%)", fontsize=10)
plt.ylabel("Property", fontsize=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()

bias_image_path = "knn_imputation_bias.png"
plt.savefig(bias_image_path, dpi=300, bbox_inches="tight")
plt.close()

# 6. Compute, Print, and Build Correlation Matrix Heatmap
corr_matrix = df_knn_imputed.corr()

print("=== CORRELATION MATRIX (KNN IMPUTED DATA) ===")
print(corr_matrix.round(3).to_string())

plt.figure(figsize=(9, 7), dpi=300)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    cbar_kws={"shrink": 0.8},
    linewidths=0.5,
)
plt.title("Property Correlation Matrix (KNN Imputed)", fontsize=12)
plt.tight_layout()

corr_image_path = "correlation_matrix.png"
plt.savefig(corr_image_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved bias chart to '{bias_image_path}'")
print(f"Saved correlation matrix plot to '{corr_image_path}'")

# 7. Automatically stage, commit, and push both artifacts to GitHub
files_to_commit = [bias_image_path, corr_image_path]

try:
    subprocess.run(["git", "add"] + files_to_commit, check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Add KNN imputation bias plot and correlation matrix heatmap",
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Successfully committed and pushed both artifacts to GitHub.")
except subprocess.CalledProcessError as e:
    print(f"Git execution note: {e}")