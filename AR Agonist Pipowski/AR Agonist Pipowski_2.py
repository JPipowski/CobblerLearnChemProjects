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
# Part 3: Generate Radius 3 Morgan Fingerprints
# ---------------------------------------------------------
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, fpSize=2048
)


def get_fp_list(smiles):
  if pd.isna(smiles):
    return [0] * 2048  # Default zero vector for missing chemicals
  try:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is not None:
      return list(morgan_generator.GetFingerprint(mol))
  except Exception:
    pass
  return [0] * 2048


print("Generating Radius 3 Morgan fingerprints...")
fp_lists = merged_df["QSAR_READY_SMILES"].apply(get_fp_list)


# ---------------------------------------------------------
# Part 4A: Create the GitHub-Friendly (Compact) File
# ---------------------------------------------------------
# Stores fingerprints as a single list column to keep file size small
compact_df = merged_df.copy()
compact_df["MORGAN_FP_R3"] = fp_lists.apply(
    lambda x: x if sum(x) > 0 else None
)  # Keep None for invalid/missing

compact_filename = "LUC_Master_dataset_compact.csv"
compact_df.to_csv(compact_filename, index=False)
print(f"✅ Created GitHub-friendly file: '{compact_filename}'")


# ---------------------------------------------------------
# Part 4B: Create the Model-Ready (Expanded) File
# ---------------------------------------------------------
# Expands fingerprints into 2048 separate binary columns (FP_R3_0 to FP_R3_2047)
fp_df = pd.DataFrame(fp_lists.tolist(), index=merged_df.index, dtype=np.int8)
fp_df.columns = [f"FP_R3_{i}" for i in range(2048)]

expanded_df = pd.concat([merged_df, fp_df], axis=1)

expanded_filename = "LUC_Master_dataset_expanded.csv"
expanded_df.to_csv(expanded_filename, index=False)
print(f"✅ Created model-ready file: '{expanded_filename}'")

print(f"\nProcessing complete! Dataset shape: {merged_df.shape}")