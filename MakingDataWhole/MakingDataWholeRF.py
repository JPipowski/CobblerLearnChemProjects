import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Create output directory for saving plots
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Titanic dataset
titanic = sns.load_dataset("titanic")

# 2. Select numerical features and compute initial correlation matrix
features = ["pclass", "age", "sibsp", "parch", "fare"]
df_numeric = titanic[features].copy()

print("--- Correlation Matrix (Original Data with NaNs) ---")
corr_original = df_numeric.corr()
print(corr_original.round(3))
print("\n")

# Scale features to maintain consistent variance across columns
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_numeric), columns=features, index=df_numeric.index
)

# 3. Perform Random Forest Imputation (via IterativeImputer) on full dataset
rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
imputer = IterativeImputer(
    estimator=rf_estimator, max_iter=10, random_state=42
)

df_rf_imputed_scaled = imputer.fit_transform(df_scaled)

# Convert scaled back to original feature scale for analysis & plotting
df_imputed = pd.DataFrame(
    scaler.inverse_transform(df_rf_imputed_scaled),
    columns=features,
    index=df_numeric.index,
)

# Print post-imputation correlation matrix
print("--- Correlation Matrix (After Random Forest Imputation) ---")
corr_imputed = df_imputed.corr()
print(corr_imputed.round(3))
print("\n")

# Save and visualize Post-Imputation Correlation Heatmap
plt.figure(figsize=(7, 5), dpi=300)
sns.heatmap(corr_imputed, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Post-Random Forest Imputation)", fontsize=12)
plt.tight_layout()

corr_plot_path = os.path.join(output_dir, "correlation_matrix.png")
plt.savefig(corr_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved correlation matrix plot to '{corr_plot_path}'")

# 4. Evaluation via Random Masking on Known Ages
known_mask = ~titanic["age"].isna()
known_indices = df_numeric[known_mask].index

np.random.seed(42)
test_mask = np.random.rand(len(known_indices)) < 0.2
test_indices = known_indices[test_mask]

# Mask 20% of known values for evaluation
df_eval_scaled = df_scaled.copy()
df_eval_scaled.loc[test_indices, "age"] = np.nan

# Predict masked ages using Random Forest Imputer
eval_imputed_scaled = imputer.fit_transform(df_eval_scaled)

# Unscale predicted age back to original units (years)
age_idx = features.index("age")
age_mean = scaler.mean_[age_idx]
age_std = scaler.scale_[age_idx]

predicted_ages_scaled = eval_imputed_scaled[test_indices, age_idx]
predicted_ages = (predicted_ages_scaled * age_std) + age_mean
known_ages = df_numeric.loc[test_indices, "age"]

mae = mean_absolute_error(known_ages, predicted_ages)
print(
    f"Mean Absolute Error (MAE) for Random Forest Age Imputation: {mae:.2f} years"
)

# 5. Build and Save Actual vs. Predicted Plot
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(
    known_ages, predicted_ages, alpha=0.6, color="navy", edgecolors="w", s=40
)
plt.plot(
    [known_ages.min(), known_ages.max()],
    [known_ages.min(), known_ages.max()],
    "r--",
    lw=2,
    label="Ideal Prediction Line",
)

plt.title(
    f"RF Imputation Evaluation: Actual vs. Predicted Age (MAE: {mae:.2f})",
    fontsize=12,
)
plt.xlabel("Actual Age", fontsize=10)
plt.ylabel("Random Forest Predicted Age", fontsize=10)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

eval_plot_path = os.path.join(
    output_dir, "rf_age_prediction_actual_vs_predicted.png"
)
plt.savefig(eval_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved scatter plot to '{eval_plot_path}'")

# 6. Automatically stage entire output directory and commit/push to GitHub
try:
    subprocess.run(["git", "add", output_dir], check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            f"Add correlation heatmap and RF age evaluation plot (MAE: {mae:.2f})",
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Successfully committed and pushed all artifacts to GitHub.")
except subprocess.CalledProcessError as e:
    print(f"Git execution note: {e}")