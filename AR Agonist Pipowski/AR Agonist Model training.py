import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

# 1. Load dataset
file_path = "LUC_Master_with_Fingerprints.csv"
df = pd.read_csv(file_path)

# Define both target hitcall columns to evaluate
target_assays = {
    "Agonist Assay": "HIT CALL_Agonist",
    "Viability Assay": "HIT CALL_Viability",
}

# Define fingerprint configurations to test
fingerprint_configs = {
    "ECFP6 Only": ["ECFP6_Str"],
    "FCFP6 Only": ["FCFP6_Str"],
    "Concatenated (ECFP6 + FCFP6)": ["ECFP6_Str", "FCFP6_Str"],
}


# Helper function to parse semicolon-separated fingerprint strings
def parse_fingerprints(fp_series):
  return np.array([list(map(int, fp.split(";"))) for fp in fp_series])


# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Master list to store all results
master_results = []

# 2. Outer Loop: Iterate through each assay
for assay_label, target_col in target_assays.items():
  print("=" * 85)
  print(f"EVALUATING ASSAY: {assay_label} ({target_col})")
  print("=" * 85)

  # Inner Loop: Iterate through each fingerprint configuration
  for config_name, fp_cols in fingerprint_configs.items():
    print(f"\n--- Testing Fingerprint Set: {config_name} ---")

    # Filter valid rows for this target and required fingerprint columns
    model_df = df.dropna(subset=[target_col] + fp_cols).copy()

    # Build feature matrix X based on configuration
    if len(fp_cols) == 1:
      X = parse_fingerprints(model_df[fp_cols[0]])
    else:
      X_ecfp = parse_fingerprints(model_df["ECFP6_Str"])
      X_fcfp = parse_fingerprints(model_df["FCFP6_Str"])
      X = np.hstack([X_ecfp, X_fcfp])

    # Extract target vector
    y = model_df[target_col].apply(lambda x: 1 if x == "Active" else 0).values

    num_neg = sum(y == 0)
    num_pos = sum(y == 1)
    scale_pos_weight_val = num_neg / num_pos

    print(f"  Valid Samples: {len(y)} | Feature Shape: {X.shape}")
    print(
        f"  Class Breakdown -> Inactive (0): {num_neg}, Active (1): {num_pos}"
    )

    # Define Base Models
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

    svm_base = SVC(class_weight="balanced", kernel="rbf", random_state=42)
    svm_clf = CalibratedClassifierCV(svm_base, ensemble=False)

    knn_clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    # Define Combined Ensembles
    ensembles = {
        "Random Forest + XGBoost": VotingClassifier(
            estimators=[("rf", rf_clf), ("xgb", xgb_clf)], voting="soft", n_jobs=-1
        ),
        "SVM + XGBoost": VotingClassifier(
            estimators=[("svm", svm_clf), ("xgb", xgb_clf)],
            voting="soft",
            n_jobs=-1,
        ),
        "KNN + XGBoost": VotingClassifier(
            estimators=[("knn", knn_clf), ("xgb", xgb_clf)],
            voting="soft",
            n_jobs=-1,
        ),
    }

    # Evaluate each ensemble using 5-Fold CV
    for model_name, ensemble in ensembles.items():
      roc_auc_list = []
      pr_auc_list = []

      for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        ensemble.fit(X_train, y_train)
        y_prob = ensemble.predict_proba(X_val)[:, 1]

        roc_auc_list.append(roc_auc_score(y_val, y_prob))
        pr_auc_list.append(average_precision_score(y_val, y_prob))

      mean_roc = np.mean(roc_auc_list)
      std_roc = np.std(roc_auc_list)
      mean_pr = np.mean(pr_auc_list)
      std_pr = np.std(pr_auc_list)

      print(
          f"    -> {model_name} | ROC-AUC: {mean_roc:.4f} (±{std_roc:.4f}) |"
          f" PR-AUC: {mean_pr:.4f} (±{std_pr:.4f})"
      )

      master_results.append({
          "Assay": assay_label,
          "Fingerprint Type": config_name,
          "Model Combination": model_name,
          "Mean ROC-AUC": f"{mean_roc:.4f} (±{std_roc:.4f})",
          "Mean PR-AUC": f"{mean_pr:.4f} (±{std_pr:.4f})",
      })

  print("\n" + "=" * 85 + "\n")

# 3. Final Master Summary Table
print("=" * 95)
print("MASTER COMPARISON SUMMARY ACROSS ALL ASSAYS, FINGERPRINTS, AND ENSEMBLES")
print("=" * 95)
summary_df = pd.DataFrame(master_results)
print(summary_df.to_string(index=False))
print("=" * 95)