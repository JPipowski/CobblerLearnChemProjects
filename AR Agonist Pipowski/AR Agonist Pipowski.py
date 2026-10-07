import glob
import re
import pandas as pd

# 1. Define chemical metadata columns (shared across files)
META_COLS = [
    'DTXSID',
    'PREFERRED NAME',
    'CASRN',
    'MOLECULAR FORMULA',
    'MONOISOTOPIC MASS',
]
NUMERICAL_VARS = [
    'AC50',
    'LOGAC50',
    'TOP',
    'SCALED TOP',
    'CONTINUOUS HIT CALL',
    'MONOISOTOPIC MASS',
]

file_paths = glob.glob('Assay List *.csv')

dfs = {}
merged_dataframes = []

# 2. Load and process individual datasets
for path in file_paths:
    df = pd.read_csv(path)
    assay_name = re.sub(
        r'Assay List (.*?)(-\d{4}-\d{2}-\d{2})?\.csv', r'\1', path
    )

    # Store individual df for analysis
    dfs[assay_name] = df.copy()

    # Rename assay-specific columns with assay prefix
    rename_dict = {
        col: f'{assay_name}_{col}' for col in df.columns if col not in META_COLS
    }
    df_renamed = df.rename(columns=rename_dict)
    merged_dataframes.append(df_renamed)

# 3. Create Master Dataset via Outer Merge
master_df = merged_dataframes[0]
for next_df in merged_dataframes[1:]:
    cols_to_drop = [
        col for col in META_COLS if col != 'DTXSID' and col in next_df.columns
    ]
    master_df = pd.merge(
        master_df, next_df.drop(columns=cols_to_drop), on='DTXSID', how='outer'
    )

# 4. Export combined master CSV
output_filename = 'Master_Agonist_Dataset.csv'
master_df.to_csv(output_filename, index=False)

# 5. Reporting Section
print('==================================================')
print('1. CHEMICAL COUNTS')
print('==================================================')
for assay_name, df in dfs.items():
    print(f'{assay_name}: {df["DTXSID"].nunique():,} unique chemicals')
print(
    f'\nCombined Master Dataset ({output_filename}): {master_df["DTXSID"].nunique():,} total unique chemicals'
)

print('\n==================================================')
print('2. PRIMARY ENDPOINT ACTIVITY (HIT CALL)')
print('==================================================')
for assay_name, df in dfs.items():
    print(f'\n--- Assay / Endpoint: {assay_name} ---')
    hit_counts = df['HIT CALL'].value_counts()
    hit_pcts = df['HIT CALL'].value_counts(normalize=True) * 100

    summary_df = pd.DataFrame(
        {'Count': hit_counts, 'Percentage (%)': hit_pcts.round(2)}
    )
    print(summary_df)

print('\n==================================================')
print('3. CROSS-ASSAY ACTIVE CHEMICAL SUMMARY')
print('==================================================')
hit_cols = [col for col in master_df.columns if col.endswith('_HIT CALL')]

# Count how many assays each chemical was active in
active_matrix = master_df[hit_cols] == 'Active'
active_counts = active_matrix.sum(axis=1)

print(
    f'Chemicals Active across ALL 4 Assays: {(active_counts == 4).sum():,} ({((active_counts == 4).sum() / len(master_df)) * 100:.2f}%)'
)
print(
    f'Chemicals Active in AT LEAST 1 Assay: {(active_counts >= 1).sum():,} ({((active_counts >= 1).sum() / len(master_df)) * 100:.2f}%)'
)

print('\nDetailed Breakdown (Active Assay Count per Chemical):')
breakdown_df = pd.DataFrame({
    'Active Assays': active_counts.value_counts().sort_index().index,
    'Chemical Count': active_counts.value_counts().sort_index().values,
    'Percentage (%)': (
        active_counts.value_counts(normalize=True).sort_index().values * 100
    ).round(2),
})
print(breakdown_df.to_string(index=False))

print('\n==================================================')
print('4. NUMERICAL VARIABLE SUMMARIES (BY ASSAY)')
print('==================================================')
for assay_name, df in dfs.items():
    print(f'\n--- Numerical Summary: {assay_name} ---')
    avail_num = [col for col in NUMERICAL_VARS if col in df.columns]
    desc = df[avail_num].describe().T[
        ['count', 'mean', 'std', 'min', '50%', 'max']
    ]
    desc.rename(columns={'50%': 'median'}, inplace=True)
    print(desc.round(4))