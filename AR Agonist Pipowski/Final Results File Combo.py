import pandas as pd
import os

# 1. Load the original environmental candidates list and QSAR prediction outputs
excel_path = "Environmental candidates file.xlsx"
cas_path = "cas_predictions_final.csv"
smiles_path = "smiles_predictions_final.csv"

xls = pd.ExcelFile(excel_path)
excel_df = pd.read_excel(excel_path, sheet_name=xls.sheet_names[0])
cas_df = pd.read_csv(cas_path)
smiles_df = pd.read_csv(smiles_path)

# 2. Clean and align keys for precise merging
excel_df['Cas RN_clean'] = excel_df['Cas RN'].astype(str).str.strip()
cas_df['CAS RN_clean'] = cas_df['CAS RN'].astype(str).str.strip()

# Merge initial candidates with CAS predictions
merged_df = pd.merge(excel_df, cas_df, left_on='Cas RN_clean', right_on='CAS RN_clean', how='left')

# 3. Construct the master DataFrame
df = pd.DataFrame({
    'No.': merged_df['No.'],
    'Candidate': merged_df['Candidate'],
    'Cas RN': merged_df['Cas RN'].astype(str).str.strip(),
    'SMILES String': merged_df['SMILES String'],
    'Resolved SMILES': merged_df.get('Resolved SMILES', ''),
    'Agonist Prediction': merged_df.get('Agonist (Act / Inact)', ''),
    'Viability Prediction': merged_df.get('Viability (Cyt / Safe)', ''),
    'AR Agonist Call': merged_df.get('AR Agonist Call', ''),
    'Screening Status': merged_df.get('Status', '')
})

# 4. Fill in any rows where CAS resolution failed using the secondary SMILES predictions
smiles_map = smiles_df.set_index('Input SMILES').to_dict('index')
for idx, row in df.iterrows():
    if row['Screening Status'] == 'FAILED' and row['SMILES String'] in smiles_map:
        s_row = smiles_map[row['SMILES String']]
        df.at[idx, 'Resolved SMILES'] = row['SMILES String'] + " (via SMILES input)"
        df.at[idx, 'Agonist Prediction'] = s_row['Agonist (Act / Inact)']
        df.at[idx, 'Viability Prediction'] = s_row['Viability (Cyt / Safe)']
        df.at[idx, 'AR Agonist Call'] = s_row['AR Agonist Call']
        df.at[idx, 'Screening Status'] = s_row['Status']

# 5. Clean internal line breaks so cells export cleanly to CSV format
for col in ['Agonist Prediction', 'Viability Prediction']:
    df[col] = df[col].astype(str).str.replace('\n', ' | ', regex=False)

# 6. Save out to the final CSV file
csv_filename = "Environmental Candidates Prediction Results Final.csv"
df.to_csv(csv_filename, index=False)

print(f"Successfully generated master CSV: {csv_filename}")