import matplotlib.pyplot as plt
import pandas as pd

# 1. Load data
df_original = pd.read_csv("alkane_dataset.csv")

# 2. Perform listwise deletion
df_clean = df_original.dropna().reset_index(drop=True)

# 3. Select numerical property columns
num_cols = [
    "carbons",
    "molecular weight",
    "boiling point",
    "viscosity",
    "thermal conductivity",
    "heat capacity",
    "branch number",
]

# 4. Compute original vs. cleaned feature means
mean_orig = df_original[num_cols].mean()
mean_clean = df_clean[num_cols].mean()

# 5. Calculate percentage bias (shift in mean relative to original)
pct_bias = ((mean_clean - mean_orig) / mean_orig) * 100

# Compile into a metrics table
bias_report = pd.DataFrame(
    {
        "Original Mean": mean_orig.round(3),
        "Cleaned Mean": mean_clean.round(3),
        "Percentage Bias (%)": pct_bias.round(2),
    }
)

print("=== BIAS METRICS REPORT ===")
print(bias_report)

# 6. Plot the Percentage Bias caused by Listwise Deletion
plt.figure(figsize=(10, 5))
bar_colors = ["navy" if val >= 0 else "crimson" for val in pct_bias]

plt.barh(pct_bias.index, pct_bias, color=bar_colors, edgecolor="black")
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.title(
    "Bias Introduced by Listwise Deletion (% Change in Mean)", fontsize=12
)
plt.xlabel("Percentage Shift in Mean (%)", fontsize=10)
plt.ylabel("Property", fontsize=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()