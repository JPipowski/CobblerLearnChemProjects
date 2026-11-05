import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# 1. Load dataset
file_path = "LUC_Master_with_Fingerprints.csv"
df = pd.read_csv(file_path)

# Target and auxiliary feature definitions
target_col = "HIT CALL_Agonist"
cross_col = "HIT CALL_Viability"

# Define fingerprint configurations to test for this single target
fingerprint_configs = {
    "ECFP6 Only + Viability Feature": ["ECFP6_Str"],
    "FCFP6 Only + Viability Feature": ["FCFP6_Str"],
    "Concatenated (ECFP6 + FCFP6) + Viability Feature": [
        "ECFP6_Str",
        "FCFP6_Str",
    ],
}


# Helper function to parse semicolon-separated fingerprint strings
def parse_fingerprints(fp_series):
  return np.array([list(map(int, fp.split(";"))) for fp in fp_series])


# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

print("=" * 85)
print(
    f"EVALUATING TARGET: {target_col} WITH AUXILIARY FEATURE: {cross_col}"
)
print("=" * 85)

# Iterate through each fingerprint configuration
for config_name, fp_cols in fingerprint_configs.items():
  print(f"\n--- Testing Feature Set: {config_name} ---")

  # Clean missing data: require target, cross-assay column, and required fingerprints
  required_cols = [target_col, cross_col] + fp_cols
  model_df = df.dropna(subset=required_cols).copy()

  # Build fingerprint feature matrix X
  if len(fp_cols) == 1:
    X_fp = parse_fingerprints(model_df[fp_cols[0]])
  else:
    X_ecfp = parse_fingerprints(model_df["ECFP6_Str"])
    X_fcfp = parse_fingerprints(model_df["FCFP6_Str"])
    X_fp = np.hstack([X_ecfp, X_fcfp])

  # Convert cross-assay column (Viability) into a binary numeric feature (Active = 1, Inactive = 0)
  cross_feature = (
      (model_df[cross_col] == "Active").astype(int).values.reshape(-1, 1)
  )

  # COMBINE: Append viability feature alongside fingerprints
  X_combined = np.hstack([X_fp, cross_feature])

  # Extract binary target vector (Agonist Active = 1, Inactive = 0)
  y = model_df[target_col].apply(lambda x: 1 if x == "Active" else 0).values

  # Calculate class imbalance parameters
  num_neg = sum(y == 0)
  num_pos = sum(y == 1)
  scale_pos_weight_val = num_neg / num_pos

  print(
      f"  Valid Samples: {len(y)} | Feature Shape: {X_combined.shape} | Actives:"
      f" {num_pos} (Inactives: {num_neg})"
  )

  # Define Ensemble Models
  xgb_clf = xgb.XGBClassifier(
      n_estimators=100,
      scale_pos_weight=scale_pos_weight_val,
      eval_metric="aucpr",
      random_state=42,
      n_jobs=-1,
  )

  rf_clf = RandomForestClassifier(
      n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
  )

  ensemble = VotingClassifier(
      estimators=[("rf", rf_clf), ("xgb", xgb_clf)], voting="soft", n_jobs=-1
  )

  # Run 5-Fold Stratified Cross-Validation
  roc_auc_list = []
  pr_auc_list = []

  for train_idx, val_idx in cv.split(X_combined, y):
    X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    ensemble.fit(X_tr, y_tr)
    y_prob = ensemble.predict_proba(X_val)[:, 1]

    roc_auc_list.append(roc_auc_score(y_val, y_prob))
    pr_auc_list.append(average_precision_score(y_val, y_prob))

  mean_roc = np.mean(roc_auc_list)
  std_roc = np.std(roc_auc_list)
  mean_pr = np.mean(pr_auc_list)
  std_pr = np.std(pr_auc_list)

  print(f"  -> Mean ROC-AUC: {mean_roc:.4f} (±{std_roc:.4f})")
  print(f"  -> Mean PR-AUC:  {mean_pr:.4f} (±{std_pr:.4f})")

  # Save to summary record
  all_results.append({
      "Configuration": config_name,
      "Mean ROC-AUC": f"{mean_roc:.4f} (±{std_roc:.4f})",
      "Mean PR-AUC": f"{mean_pr:.4f} (±{std_pr:.4f})",
  })

# Final Summary Table
print("\n" + "=" * 85)
print("SUMMARY: PREDICTING AGONIST USING VIABILITY FEATURE + FINGERPRINTS")
print("=" * 85)
summary_df = pd.DataFrame(all_results)
print(summary_df.to_string(index=False))
print("=" * 85)