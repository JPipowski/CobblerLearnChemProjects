import pandas as pd

# Load dataset
file_path = "LUC_Master_with_Fingerprints.csv"
df = pd.read_csv(file_path)

# Total rows and unique chemicals count using DTXSID
total_rows = len(df)
unique_chemicals = df["DTXSID"].nunique()

# Map requested columns to actual CSV column names
target_columns = {
    "Hitcall (Agonist Assay)": "HIT CALL_Agonist",
    "Hitcall (Viability Assay)": "HIT CALL_Viability",
    "SMILES": "SMILES",
    "QSAR SMILES": "QSAR_READY_SMILES",
    "ECFP6 Fingerprint": "ECFP6_Str",
    "FCFP6 Fingerprint": "FCFP6_Str",
}

report = []

for label, col_name in target_columns.items():
    if col_name in df.columns:
        missing_cnt = df[col_name].isnull().sum()
        present_cnt = total_rows - missing_cnt
        missing_pct = (missing_cnt / total_rows) * 100

        report.append(
            {
                "Target Feature": label,
                "CSV Column Name": col_name,
                "Present Values": present_cnt,
                "Missing Values": missing_cnt,
                "Missing (%)": f"{missing_pct:.2f}%",
            }
        )

report_df = pd.DataFrame(report)

print("=" * 70)
print(f"DATASET REPORT: {file_path}")
print(f"Total Rows: {total_rows}")
print(f"Unique Chemicals (DTXSID): {unique_chemicals}")
print("=" * 70)
print(report_df.to_string(index=False))

# Detailed breakdown for both Hitcall assays
print("\n" + "=" * 70)
print("DETAILED BREAKDOWN: HITCALL ASSAYS")
print("=" * 70)

for col_name, label in [
    ("HIT CALL_Agonist", "Hitcall (Agonist Assay)"),
    ("HIT CALL_Viability", "Hitcall (Viability Assay)"),
]:
    print(f"\n--- {label} ({col_name}) ---")
    value_counts = df[col_name].value_counts(dropna=False)
    for val, count in value_counts.items():
        status_label = "Missing (NaN)" if pd.isna(val) else val
        pct = (count / total_rows) * 100
        print(f"  * {status_label}: {count} rows ({pct:.2f}%)")