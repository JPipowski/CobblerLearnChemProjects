import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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


# Helper function to find the threshold that maximizes F1 score
def find_optimal_threshold(y_true, y_prob):
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (np.array(y_prob) >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1


print("=" * 110)
print(
    f"EVALUATING TARGET: {target_col} WITH AUXILIARY FEATURE: {cross_col}"
)
print("=" * 110)

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

    # Define All Ensemble Combinations
    ensembles = {
        "Random Forest + XGBoost": VotingClassifier(
            estimators=[("rf", rf_clf), ("xgb", xgb_clf)],
            voting="soft",
            n_jobs=-1,
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

    # Evaluate each ensemble using 5-Fold Stratified Cross-Validation
    for model_name, ensemble in ensembles.items():
        y_true_oof = []
        y_prob_oof = []
        roc_auc_list = []
        pr_auc_list = []

        for train_idx, val_idx in cv.split(X_combined, y):
            X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            ensemble.fit(X_tr, y_tr)
            y_prob = ensemble.predict_proba(X_val)[:, 1]

            roc_auc_list.append(roc_auc_score(y_val, y_prob))
            pr_auc_list.append(average_precision_score(y_val, y_prob))

            y_true_oof.extend(y_val)
            y_prob_oof.extend(y_prob)

        mean_roc = np.mean(roc_auc_list)
        std_roc = np.std(roc_auc_list)
        mean_pr = np.mean(pr_auc_list)
        std_pr = np.std(pr_auc_list)

        # Find optimal threshold on Out-Of-Fold probabilities and compute precision/recall/f1
        opt_thresh, _ = find_optimal_threshold(y_true_oof, y_prob_oof)
        oof_preds = (np.array(y_prob_oof) >= opt_thresh).astype(int)

        precision = precision_score(y_true_oof, oof_preds, zero_division=0)
        recall = recall_score(y_true_oof, oof_preds, zero_division=0)
        f1 = f1_score(y_true_oof, oof_preds, zero_division=0)

        print(
            f"    -> {model_name:<25} | ROC-AUC: {mean_roc:.4f} | PR-AUC:"
            f" {mean_pr:.4f} | Thresh: {opt_thresh:.2f} | Prec:"
            f" {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}"
        )

        # Save to summary record
        all_results.append({
            "Configuration": config_name,
            "Model": model_name,
            "Optimal Threshold": round(opt_thresh, 2),
            "Mean ROC-AUC": f"{mean_roc:.4f} (±{std_roc:.4f})",
            "Mean PR-AUC": f"{mean_pr:.4f} (±{std_pr:.4f})",
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Raw_ROC_AUC": round(mean_roc, 4),
            "Raw_PR_AUC": round(mean_pr, 4),
        })

print("\n" + "=" * 110 + "\n")

# Final Summary Table & Export
print("=" * 125)
print(
    "SUMMARY: PREDICTING AGONIST USING VIABILITY FEATURE + FINGERPRINTS ACROSS"
    " MODELS"
)
print("=" * 125)
summary_df = pd.DataFrame(all_results)
print(summary_df.to_string(index=False))
print("=" * 125)

# Export results to CSV
output_csv_filename = "Agonist_Viability_Auxiliary_Feature_Results.csv"
summary_df.to_csv(output_csv_filename, index=False)
print(f"\nResults successfully exported to: {output_csv_filename}")