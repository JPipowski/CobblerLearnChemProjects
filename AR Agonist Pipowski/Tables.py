import matplotlib.pyplot as plt
import pandas as pd

# Load datasets
df_benchmark = pd.read_csv('QSAR_Master_Benchmark_Results.csv')
df_aux = pd.read_csv('Agonist_Viability_Auxiliary_Feature_Results.csv')

# Clean configuration column to match baseline fingerprint types
df_aux['Base_FP'] = df_aux['Configuration'].str.replace(' + Viability Feature', '', regex=False)
df_agonist_base = df_benchmark[df_benchmark['Assay'] == 'Agonist Assay'].copy()

# Merge baseline and auxiliary feature results
merged = pd.merge(
    df_agonist_base,
    df_aux,
    left_on=['Fingerprint Type', 'Model Combination'],
    right_on=['Base_FP', 'Model'],
    suffixes=('_baseline', '_with_viability')
)

# Extract numerical Mean ROC-AUC values
merged['Baseline_ROC'] = merged['Mean ROC-AUC (±std)'].apply(lambda x: float(x.split()[0]))
merged['Viability_ROC'] = merged['Mean ROC-AUC'].apply(lambda x: float(x.split()[0]))

# Prepare plotting data
plot_data = merged[['Fingerprint Type', 'Model Combination', 'Baseline_ROC', 'Viability_ROC']].copy()
plot_data['Model_FP'] = plot_data['Model Combination'] + " (" + plot_data['Fingerprint Type'] + ")"

# Generate the grouped bar chart
plt.figure(figsize=(10, 6))
x = range(len(plot_data))
width = 0.35

plt.bar([i - width/2 for i in x], plot_data['Baseline_ROC'], width, label='Baseline ROC-AUC', color='#2ecc71')
plt.bar([i + width/2 for i in x], plot_data['Viability_ROC'], width, label='With Viability Feature ROC-AUC', color='#e74c3c')

plt.xlabel('Model & Fingerprint Configuration', fontweight='bold')
plt.ylabel('Mean ROC-AUC', fontweight='bold')
plt.title('Performance Comparison: Baseline vs. Viability Feature Added (Agonist Assay)', fontweight='bold', fontsize=12)
plt.xticks(x, plot_data['Model_FP'], rotation=45, ha='right')
plt.ylim(0.6, 0.85)
plt.legend()
plt.tight_layout()

# Save and display figure
plt.savefig('roc_auc_comparison.png', dpi=300)
plt.show()