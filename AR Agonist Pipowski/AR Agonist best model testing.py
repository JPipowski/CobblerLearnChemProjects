import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

# 1. Load dataset and train the final model on ALL valid data
file_path = "LUC_Master_with_Fingerprints.csv"
df = pd.read_csv(file_path)

target_cols = ["HIT CALL_Agonist", "HIT CALL_Viability"]
fp_cols = ["ECFP6_Str", "FCFP6_Str"]


def parse_fingerprints(fp_series):
  return np.array([list(map(int, fp.split(";"))) for fp in fp_series])


required_cols = target_cols + fp_cols
model_df = df.dropna(subset=required_cols).copy()

X_ecfp = parse_fingerprints(model_df["ECFP6_Str"])
X_fcfp = parse_fingerprints(model_df["FCFP6_Str"])
X_train_full = np.hstack([X_ecfp, X_fcfp])

Y_train_full = np.column_stack([
    model_df[target_cols[0]].apply(lambda x: 1 if x == "Active" else 0).values,
    model_df[target_cols[1]].apply(lambda x: 1 if x == "Active" else 0).values,
])

print("Training final Multi-Output Ensemble on full dataset...")
print(f"Expected Feature Dimension: {X_train_full.shape[1]} (4096 total)")

base_rf = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
multi_rf = MultiOutputClassifier(base_rf, n_jobs=-1)

base_xgb = xgb.XGBClassifier(
    n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=-1
)
multi_xgb = MultiOutputClassifier(base_xgb, n_jobs=-1)

# Fit on all data
multi_rf.fit(X_train_full, Y_train_full)
multi_xgb.fit(X_train_full, Y_train_full)
print("Model training complete!\n")


# 2. Modern RDKit Fingerprint Generators (2048 bits each -> 4096 total)
ecfp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
fcfp_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)


def get_fingerprint_from_smiles(smiles):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    raise ValueError(f"Invalid SMILES string: {smiles}")

  # Generate numpy arrays for ECFP6 and FCFP6
  ecfp6_arr = ecfp_generator.GetFingerprintAsNumPy(mol)
  fcfp6_arr = fcfp_generator.GetFingerprintAsNumPy(mol)

  # Concatenate both to match the 4096 feature shape
  return np.hstack([ecfp6_arr, fcfp6_arr]).reshape(1, -1)


# 3. Prediction Function for a New Chemical
def predict_chemical(smiles):
  X_new = get_fingerprint_from_smiles(smiles)

  # Get soft probabilities from both models
  probs_rf = multi_rf.predict_proba(X_new)
  probs_xgb = multi_xgb.predict_proba(X_new)

  # Soft-voting average (50/50 split between RF and XGBoost)
  prob_agonist = 0.5 * probs_rf[0][:, 1][0] + 0.5 * probs_xgb[0][:, 1][0]
  prob_viability = 0.5 * probs_rf[1][:, 1][0] + 0.5 * probs_xgb[1][:, 1][0]

  print(f"--- Prediction Results for SMILES: {smiles} ---")
  print(
      f"  -> Agonist Probability:   {prob_agonist:.4f} ("
      f"{'ACTIVE (Agonist)' if prob_agonist >= 0.5 else 'INACTIVE'})"
  )
  print(
      f"  -> Viability Probability: {prob_viability:.4f} ("
      f"{'ACTIVE (Viable/Toxic)' if prob_viability >= 0.5 else 'INACTIVE'})"
  )
  print("-" * 55)


# Example test molecule (Aspirin)
test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
predict_chemical(test_smiles)