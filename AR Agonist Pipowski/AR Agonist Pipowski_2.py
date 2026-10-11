import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# ---------------------------------------------------------
# Part 1: Create the Master Dataset from the 2 Tox21 Files
# ---------------------------------------------------------
file_agonist = "Assay List TOX21_AR_LUC_MDAKB2_Agonist-2026-10-09.csv"
file_viability = "Assay List TOX21_AR_LUC_MDAKB2_Agonist_3uM_Nilutamide_viability-2026-10-06.csv"

df_agonist = pd.read_csv(file_agonist)
df_viability = pd.read_csv(file_viability)

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

# Rename assay readout columns with specific suffixes to avoid column collision
df_ag_prep = df_agonist.rename(
    columns={col: f"{col}_Agonist" for col in readout_cols}
)
df_vi_prep = df_viability.rename(
    columns={col: f"{col}_Viability" for col in readout_cols}
)

# Combine into master dataset using DTXSID and shared chemical metadata
LUC_Master_dataset = pd.merge(
    df_ag_prep, df_vi_prep, on=metadata_cols, how="outer"
)

# Export intermediate master dataset
LUC_Master_dataset.to_csv("LUC_Master_dataset.csv", index=False)
print(f"Created LUC_Master_dataset with shape: {LUC_Master_dataset.shape}")

# ---------------------------------------------------------
# Part 2: Merge SMILES Strings to create LUC_Master_Strings
# ---------------------------------------------------------
batch_file = "Batch-Search-chemical-list-2026-10-11.csv"
df_batch = pd.read_csv(batch_file)

# Extract relevant columns from the batch file
df_smiles = df_batch[["DTXSID", "SMILES"]]

# Merge the SMILES strings into the master dataset
luc_master_strings = pd.merge(
    LUC_Master_dataset, df_smiles, on="DTXSID", how="left"
)

# Export the combined master dataset with SMILES
luc_master_strings.to_csv("LUC_Master_Strings.csv", index=False)
print(f"Created 'LUC_Master_Strings.csv' with shape: {luc_master_strings.shape}")

# ---------------------------------------------------------
# Part 3: Generate Molecular Fingerprints via MorganGenerator
# ---------------------------------------------------------
print("Generating Morgan fingerprints using MorganGenerator (Radius=2, 2048 bits)...")

# Initialize the modern RDKit Morgan fingerprint generator
# useCountSimulation=False ensures standard binary bit-vector representation equivalent to legacy methods
mfpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, useCountSimulation=False
)


def compute_morgan_fingerprint(smiles):
    """Computes Morgan fingerprint bit vector from a SMILES string using MorganGenerator."""
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES string

    fp = mfpgen.GetFingerprint(mol)
    # Convert explicit bit vector to a list of integers (0s and 1s)
    return list(fp)


# Apply fingerprint generation to each row based on the SMILES column
luc_master_strings["Morgan_FP"] = luc_master_strings["SMILES"].apply(compute_morgan_fingerprint)

# Export the final enhanced dataset
final_output = "LUC_Master_With_Fingerprints.csv"
luc_master_strings.to_csv(final_output, index=False)

print(f"Successfully created '{final_output}' with shape: {luc_master_strings.shape}")