import glob
import re
from functools import reduce
import pandas as pd

# 1. Define chemical metadata columns (shared across files)
META_COLS = [
    'DTXSID',
    'PREFERRED NAME',
    'CASRN',
    'MOLECULAR FORMULA',
    'MONOISOTOPIC MASS',
]

# 2. Locate all matching CSV files
file_paths = glob.glob('Assay List *.csv')

dfs = []
for path in file_paths:
    # Read the CSV file
    df = pd.read_csv(path)

    # Extract clean assay name from filename (e.g., 'ATG_AR_TRANS')
    assay_name = re.sub(r'Assay List (.*?)(-\d{4}-\d{2}-\d{2})?\.csv', r'\1', path)

    # Prefix non-metadata columns with the assay name to distinguish assay results
    rename_dict = {
        col: f'{assay_name}_{col}' for col in df.columns if col not in META_COLS
    }
    df = df.rename(columns=rename_dict)

    dfs.append(df)

# 3. Merge datasets iteratively on DTXSID using an outer join
master_df = dfs[0]
for next_df in dfs[1:]:
    # Drop shared metadata columns from subsequent dataframes to avoid duplicate '_x' / '_y' columns
    cols_to_drop = [
        col for col in META_COLS if col != 'DTXSID' and col in next_df.columns
    ]
    next_df_trimmed = next_df.drop(columns=cols_to_drop)

    # Outer join ensures chemicals present in any file are preserved
    master_df = pd.merge(master_df, next_df_trimmed, on='DTXSID', how='outer')

# 4. Export combined master dataset
output_filename = 'Master_Assay_Dataset_Combined.csv'
master_df.to_csv(output_filename, index=False)

print(
    f'Successfully combined {len(file_paths)} files into {output_filename}'
)
print(
    f'Master dataset shape: {master_df.shape[0]} rows, {master_df.shape[1]} columns'
)