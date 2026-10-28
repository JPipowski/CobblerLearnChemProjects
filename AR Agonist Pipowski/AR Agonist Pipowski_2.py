import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# File paths
file_viability = 'Assay List TOX21_AR_LUC_MDAKB2_Agonist_3uM_Nilutamide_viability-2026-10-06.csv'
file_agonist = 'Assay List TOX21_AR_LUC_MDAKB2_Agonist-2026-10-09.csv'
file_master = 'LUC_Master_dataset.csv'
file_new = 'CCD-Batch-Search_2026-10-16_05_15_34.csv'

# Load datasets
df_viab = pd.read_csv(file_viability)
df_agon = pd.read_csv(file_agonist)
df_mast = pd.read_csv(file_master)
df_new = pd.read_csv(file_new)

print("=" * 75)
print("1. CHEMICAL COUNTS BY DTXSID")
print("=" * 75)

# Clean DTXSID in master dataset for proper matching (removes URL prefix if present)
df_mast['DTXSID_clean'] = df_mast['DTXSID'].str.replace('https://comptox.epa.gov/dashboard/chemical/details/', '')

datasets = {
    "File 1 (Viability)": df_viab,
    "File 2 (Agonist)": df_agon,
    "File 3 (Master Dataset)": df_mast,
    "File 4 (CCD Batch Search New)": df_new
}

for name, df in datasets.items():
    id_col = 'DTXSID_clean' if 'DTXSID_clean' in df.columns else 'DTXSID'
    print(f"\n--- {name} ---")
    print(f"Total Rows / Entries: {len(df):,}")
    print(f"Unique DTXSIDs: {df[id_col].nunique():,}")
    print("Top 3 DTXSID Count Sample:")
    print(df[id_col].value_counts().head(3))

print("\n" + "=" * 75)
print("2. MISSING DATA REPORT FOR EACH INDIVIDUAL FILE")
print("=" * 75)


def get_missing_report(df, name):
    missing_cnt = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    report = pd.DataFrame({
        'Missing Count': missing_cnt,
        'Missing %': missing_pct
    })
    print(f"\nMissing Data Breakdown: {name}")
    print(report)


get_missing_report(df_viab, "File 1 (Viability)")
get_missing_report(df_agon, "File 2 (Agonist)")
get_missing_report(df_mast, "File 3 (Master Dataset)")
get_missing_report(df_new, "File 4 (CCD Batch Search New)")

print("\n" + "=" * 75)
print("3. MODERN ECFP6 & FCFP6 FINGERPRINT GENERATION (Radius = 3, 2048 bits)")
print("=" * 75)

# Initialize modern RDKit generators (radius=3 for ECFP6/FCFP6, 2048 bits)
ecfp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
fcfp_inv_generator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
fcfp_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, fpSize=2048, atomInvariantsGenerator=fcfp_inv_generator
)


def compute_and_serialize_fingerprints(row):
    smiles = row['QSAR_READY_SMILES'] if pd.notna(row['QSAR_READY_SMILES']) else row['SMILES']
    if pd.isna(smiles):
        return '', ''
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return '', ''

    # Generate ECFP6 & FCFP6 bit vectors and join into semicolon-separated strings for CSV
    ecfp_str = ';'.join(map(str, ecfp_generator.GetFingerprint(mol)))
    fcfp_str = ';'.join(map(str, fcfp_generator.GetFingerprint(mol)))

    return ecfp_str, fcfp_str


print("Generating modern ECFP6 and FCFP6 bit vectors and serializing to strings...")
fp_results = df_new.apply(compute_and_serialize_fingerprints, axis=1)

df_new['ECFP6_Str'] = [res[0] for res in fp_results]
df_new['FCFP6_Str'] = [res[1] for res in fp_results]

successful_ecfp = (df_new['ECFP6_Str'] != '').sum()
successful_fcfp = (df_new['FCFP6_Str'] != '').sum()

print(f"Successfully generated ECFP6 for {successful_ecfp:,} out of {len(df_new):,} chemicals.")
print(f"Successfully generated FCFP6 for {successful_fcfp:,} out of {len(df_new):,} chemicals.")

print("\n" + "=" * 75)
print("4. BEFORE & AFTER MERGE MISSING DATA REPORT (Master + New File)")
print("=" * 75)

# Before merge: combined missing counts and percentages across separate unmerged tables
before_missing_counts = pd.concat([df_mast.isnull().sum(), df_new.isnull().sum()]).groupby(level=0).sum()
total_rows_before = len(df_mast) + len(df_new)
before_missing_pct = (before_missing_counts / total_rows_before * 100).round(2)

# Perform outer merge on DTXSID
df_merged_final = pd.merge(
    df_mast,
    df_new,
    left_on='DTXSID_clean',
    right_on='DTXSID',
    how='outer',
    suffixes=('', '_new')
)

after_missing_counts = df_merged_final.isnull().sum()
after_missing_pct = (df_merged_final.isnull().mean() * 100).round(2)

comparison_df = pd.DataFrame({
    'Before Merge Missing Count': before_missing_counts,
    'Before Merge Missing %': before_missing_pct,
    'After Merge Missing Count': after_missing_counts,
    'After Merge Missing %': after_missing_pct
}).fillna(0)

print(comparison_df.to_string())

print("\n" + "=" * 75)
print("5. EXPORTING DATA (OPTION 1: CSV FORMAT WITH STRINGIFIED FINGERPRINTS)")
print("=" * 75)

output_csv = 'LUC_Master_with_Fingerprints.csv'
df_merged_final.to_csv(output_csv, index=False)
print(f"Successfully exported full merged dataset with semicolon-separated fingerprints to CSV: {output_csv}")