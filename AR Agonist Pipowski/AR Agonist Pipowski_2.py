import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# ---------------------------------------------------------
# Part 1: Load Datasets
# ---------------------------------------------------------
file_master = "LUC_Master_dataset.csv"
file_ccd = "CCD-Batch-Search_2026-10-16_05_15_34.csv"

luc_df = pd.read_csv(file_master)
ccd_df = pd.read_csv(file_ccd)

# ---------------------------------------------------------
# Part 2: Normalize DTXSID & Merge SMILES Data
# ---------------------------------------------------------
luc_df["normalized_DTXSID"] = luc_df["DTXSID"].apply(
    lambda x: x.split("/")[-1] if isinstance(x, str) else x
)
ccd_df["normalized_DTXSID"] = ccd_df["DTXSID"].apply(
    lambda x: x.split("/")[-1] if isinstance(x, str) else x
)

merged_df = pd.merge(
    luc_df,
    ccd_df[["normalized_DTXSID", "SMILES", "QSAR_READY_SMILES"]],
    on="normalized_DTXSID",
    how="left",
).drop(columns=["normalized_DTXSID"])


# ---------------------------------------------------------
# Part 3: Generate Radius 3 Morgan Fingerprints (Single Column)
# ---------------------------------------------------------
# Initialize modern RDKit Morgan fingerprint generator (Radius 3, 2048 bits)
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, fpSize=2048
)


def get_morgan_fingerprint(smiles):
  """Generates a Morgan fingerprint bit vector (Radius 3, ECFP6-equivalent)

  as a list of integers (0/1) from a QSAR-ready SMILES string using RDKit.
  """
  if pd.isna(smiles):
    return None
  try:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is not None:
      fp = morgan_generator.GetFingerprint(mol)
      return list(fp)
  except Exception:
    pass
  return None


print("Generating Radius 3 Morgan fingerprints...")
merged_df["MORGAN_FP_R3"] = merged_df["QSAR_READY_SMILES"].apply(
    get_morgan_fingerprint
)

# ---------------------------------------------------------
# Part 4: Export Compact Dataset
# ---------------------------------------------------------
output_filename = "LUC_Master_dataset_with_morgan_fps_r3.csv"
merged_df.to_csv(output_filename, index=False)

print(f"Successfully created '{output_filename}' with shape: {merged_df.shape}")
print(
    f"Total valid fingerprints generated:"
    f" {merged_df['MORGAN_FP_R3'].notna().sum()}"
)