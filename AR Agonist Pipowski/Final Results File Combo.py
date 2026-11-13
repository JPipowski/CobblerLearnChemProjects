import pandas as pd

# 1. Load your original environmental candidates list and QSAR prediction outputs
excel_path = "Environmental candidates file.xlsx"
xls = pd.ExcelFile(excel_path)
excel_df = pd.read_excel(excel_path, sheet_name=xls.sheet_names[0])
cas_df = pd.read_csv("cas_predictions_final.csv")

# 2. Clean and align CAS RN keys for precise merging
excel_df['Cas RN_clean'] = excel_df['Cas RN'].astype(str).str.strip()
cas_df['CAS RN_clean'] = cas_df['CAS RN'].astype(str).str.strip()

merged_df = pd.merge(excel_df, cas_df, left_on='Cas RN_clean', right_on='CAS RN_clean', how='left')

# 3. Format the final output columns
output_df = pd.DataFrame({
    'No.': merged_df['No.'],
    'Candidate': merged_df['Candidate'],
    'Cas RN': merged_df['Cas RN'],
    'SMILES String': merged_df['SMILES String'],
    'Resolved SMILES': merged_df.get('Resolved SMILES', ''),
    'Agonist Prediction': merged_df.get('Agonist (Act / Inact)', ''),
    'Viability Prediction': merged_df.get('Viability (Cyt / Safe)', ''),
    'AR Agonist Call': merged_df.get('AR Agonist Call', ''),
    'Screening Status': merged_df.get('Status', '')
})

# 4. Save to a final master Excel workbook
output_filename = "Environmental Candidates Prediction Results Final.xlsx"
output_df.to_excel(output_filename, index=False)

print(f"Successfully generated master workbook: {output_filename}")