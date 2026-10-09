import pandas as pd

# 1. Load the two Tox21 dataset CSV files
file_agonist = "Assay List TOX21_AR_LUC_MDAKB2_Agonist-2026-10-09.csv"
file_viability = "Assay List TOX21_AR_LUC_MDAKB2_Agonist_3uM_Nilutamide_viability-2026-10-06.csv"

df_agonist = pd.read_csv(file_agonist)
df_viability = pd.read_csv(file_viability)

# 2. Define chemical metadata and assay-specific readout columns
metadata_cols = [
    "DTXSID",
    "PREFERRED NAME",
    "CASRN",
    "MOLECULAR FORMULA",
    "MONOISOTOPIC MASS",
]
readout_cols = [
    "TOXCAST ACTIVE",
    "TOXCAST TOTAL",
    "% TOXCAST ACTIVE",
    "HIT CALL",
    "CONTINUOUS HIT CALL",
    "TOP",
    "SCALED TOP",
    "AC50",
    "LOGAC50",
]

# 3. Rename assay readout columns with specific suffixes to avoid column collision
df_ag_prep = df_agonist.rename(
    columns={col: f"{col}_Agonist" for col in readout_cols}
)
df_vi_prep = df_viability.rename(
    columns={col: f"{col}_Viability" for col in readout_cols}
)

# 4. Combine into master dataset using DTXSID (and shared chemical metadata)
LUC_Master_dataset = pd.merge(
    df_ag_prep, df_vi_prep, on=metadata_cols, how="outer"
)

# 5. Export combined dataset
LUC_Master_dataset.to_csv("LUC_Master_dataset.csv", index=False)

print(
    f"Successfully created LUC_Master_dataset with shape: {LUC_Master_dataset.shape}"
)