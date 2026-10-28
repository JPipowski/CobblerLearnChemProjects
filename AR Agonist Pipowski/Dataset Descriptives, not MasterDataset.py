import pandas as pd

# Define filenames
file1 = 'Assay List TOX21_AR_LUC_MDAKB2_Agonist_3uM_Nilutamide_viability-2026-10-06.csv'
file2 = 'Assay List TOX21_AR_LUC_MDAKB2_Agonist-2026-10-09.csv'

# Load datasets
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("=" * 70)
print("1. TOTAL CHEMICAL COUNTS BY DTXSID")
print("=" * 70)

print(f"\n--- File 1: {file1} ---")
print(f"Total Rows / Total Entries: {len(df1)}")
print(f"Unique DTXSID Count: {df1['DTXSID'].nunique()}")
print("Chemical counts by DTXSID (Value counts preview):")
print(df1['DTXSID'].value_counts().head(5))

print(f"\n--- File 2: {file2} ---")
print(f"Total Rows / Total Entries: {len(df2)}")
print(f"Unique DTXSID Count: {df2['DTXSID'].nunique()}")
print("Chemical counts by DTXSID (Value counts preview):")
print(df2['DTXSID'].value_counts().head(5))


print("\n" + "=" * 70)
print("2. MISSING DATA REPORT BY COLUMN")
print("=" * 70)

def generate_missing_report(df, filename):
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    report = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_pct
    })
    print(f"\nMissing Data Report for: {filename}")
    print(report.to_string())
    print("-" * 70)

generate_missing_report(df1, file1)
generate_missing_report(df2, file2)