import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

# 1. Load Dataset
file_path = "LUC_Master_with_Fingerprints.csv"
df = pd.read_csv(file_path)

target_cols = ["HIT CALL_Agonist", "HIT CALL_Viability"]
fp_cols = ["ECFP6_Str", "FCFP6_Str"]


def parse_fingerprints(fp_series):
  return np.array([list(map(int, fp.split(";"))) for fp in fp_series])


required_cols = target_cols + fp_cols
model_df = df.dropna(subset=required_cols).copy()

# Concatenate 2048-bit ECFP6 + 2048-bit FCFP6 features (4096 total)
X_ecfp = parse_fingerprints(model_df["ECFP6_Str"])
X_fcfp = parse_fingerprints(model_df["FCFP6_Str"])
X_train_full = np.hstack([X_ecfp, X_fcfp])

Y_train_full = np.column_stack([
    model_df[target_cols[0]].apply(lambda x: 1 if x == "Active" else 0).values,
    model_df[target_cols[1]].apply(lambda x: 1 if x == "Active" else 0).values,
])

# 2. Cross-Validation Evaluation to Document Standard Scores
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_rf_cv = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
multi_rf_cv = MultiOutputClassifier(base_rf_cv, n_jobs=-1)

base_xgb_cv = xgb.XGBClassifier(
    n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=-1
)
multi_xgb_cv = MultiOutputClassifier(base_xgb_cv, n_jobs=-1)

y_true_ag, y_prob_ag = [], []
y_true_vi, y_prob_vi = [], []

print("Running 5-Fold Cross-Validation for Documentation Metrics...")
for train_idx, val_idx in cv.split(X_train_full, Y_train_full[:, 0]):
  X_tr, X_va = X_train_full[train_idx], X_train_full[val_idx]
  multi_rf_cv.fit(X_tr, Y_train_full[train_idx])
  multi_xgb_cv.fit(X_tr, Y_train_full[train_idx])

  p_rf = multi_rf_cv.predict_proba(X_va)
  p_xgb = multi_xgb_cv.predict_proba(X_va)

  y_true_ag.extend(Y_train_full[val_idx, 0])
  y_prob_ag.extend(0.5 * p_rf[0][:, 1] + 0.5 * p_xgb[0][:, 1])
  y_true_vi.extend(Y_train_full[val_idx, 1])
  y_prob_vi.extend(0.5 * p_rf[1][:, 1] + 0.5 * p_xgb[1][:, 1])

# Calculate evaluation metrics
roc_ag = roc_auc_score(y_true_ag, y_prob_ag)
pr_ag = average_precision_score(y_true_ag, y_prob_ag)
roc_vi = roc_auc_score(y_true_vi, y_prob_vi)
pr_vi = average_precision_score(y_true_vi, y_prob_vi)

macro_roc = (roc_ag + roc_vi) / 2
macro_pr = (pr_ag + pr_vi) / 2

# 3. Train Final Model on All Data
print("Training Final Model on Complete Dataset...")
base_rf = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
multi_rf = MultiOutputClassifier(base_rf, n_jobs=-1)

base_xgb = xgb.XGBClassifier(
    n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=-1
)
multi_xgb = MultiOutputClassifier(base_xgb, n_jobs=-1)

multi_rf.fit(X_train_full, Y_train_full)
multi_xgb.fit(X_train_full, Y_train_full)

# 4. RDKit Generators & Batch Inference Function
ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)


def get_fingerprints(smiles):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  ecfp = ecfp_gen.GetFingerprintAsNumPy(mol)
  fcfp = fcfp_gen.GetFingerprintAsNumPy(mol)
  return np.hstack([ecfp, fcfp]).reshape(1, -1)


def generate_documentation_csv(
    smiles_list, output_csv="QSAR_Model_Predictions_Documentation.csv"
):
  records = []

  for smiles in smiles_list:
    X_new = get_fingerprints(smiles)

    if X_new is None:
      records.append({
          "SMILES": smiles,
          "Agonist_Probability": np.nan,
          "Agonist_Call": "INVALID_SMILES",
          "Viability_Probability": np.nan,
          "Viability_Call": "INVALID_SMILES",
          "Agonist_ROC_AUC": round(roc_ag, 4),
          "Agonist_PR_AUC": round(pr_ag, 4),
          "Viability_ROC_AUC": round(roc_vi, 4),
          "Viability_PR_AUC": round(pr_vi, 4),
          "Global_Macro_ROC_AUC": round(macro_roc, 4),
          "Global_Macro_PR_AUC": round(macro_pr, 4),
      })
      continue

    probs_rf = multi_rf.predict_proba(X_new)
    probs_xgb = multi_xgb.predict_proba(X_new)

    prob_ag = 0.5 * probs_rf[0][:, 1][0] + 0.5 * probs_xgb[0][:, 1][0]
    prob_vi = 0.5 * probs_rf[1][:, 1][0] + 0.5 * probs_xgb[1][:, 1][0]

    records.append({
        "SMILES": smiles,
        "Agonist_Probability": round(prob_ag, 4),
        "Agonist_Call": "ACTIVE" if prob_ag >= 0.5 else "INACTIVE",
        "Viability_Probability": round(prob_vi, 4),
        "Viability_Call": (
            "CYTOTOXIC (Active)" if prob_vi >= 0.5 else "SAFE (Inactive)"
        ),
        "Agonist_ROC_AUC": round(roc_ag, 4),
        "Agonist_PR_AUC": round(pr_ag, 4),
        "Viability_ROC_AUC": round(roc_vi, 4),
        "Viability_PR_AUC": round(pr_vi, 4),
        "Global_Macro_ROC_AUC": round(macro_roc, 4),
        "Global_Macro_PR_AUC": round(macro_pr, 4),
    })

  doc_df = pd.DataFrame(records)
  doc_df.to_csv(output_csv, index=False)
  print(f"\nDocumentation CSV exported successfully: {output_csv}")
  return doc_df


# 5. Example Batch Run
input_smiles = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
]

df_doc = generate_documentation_csv(input_smiles)
print(df_doc)