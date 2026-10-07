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

# Export combined master CSV
master_df.to_csv('Master_Assay_Dataset_Combined.csv', index=False)

# 4. Reporting Section
print('==================================================')
print('1. CHEMICAL COUNTS')
print('==================================================')
for assay_name, df in dfs.items():
    print(f'{assay_name}: {df["DTXSID"].nunique():,} unique chemicals')
print(
    f'\nCombined Master Dataset: {master_df["DTXSID"].nunique():,} total unique chemicals'
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
print('3. NUMERICAL VARIABLE SUMMARIES (BY ASSAY)')
print('==================================================')
for assay_name, df in dfs.items():
    print(f'\n--- Numerical Summary: {assay_name} ---')
    avail_num = [col for col in NUMERICAL_VARS if col in df.columns]
    desc = df[avail_num].describe().T[
        ['count', 'mean', 'std', 'min', '50%', 'max']
    ]
    desc.rename(columns={'50%': 'median'}, inplace=True)
    print(desc.round(4))