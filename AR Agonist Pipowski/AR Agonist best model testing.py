import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb


def parse_fps(series):
    return np.array([list(map(int, str(s).split(";"))) for s in series], dtype=np.float32)


def best_thresh_f1(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t


def summarize_metrics(name, y_true, y_prob, thresh):
    preds = (y_prob >= thresh).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    print(f"\n=== {name} (Threshold = {thresh:.2f}) ===")
    print(f"  ROC AUC  : {auc(fpr, tpr):.4f} (sklearn roc_auc_score: {roc_auc_score(y_true, y_prob):.4f})")
    print(f"  Recall   : {recall_score(y_true, preds, zero_division=0):.4f}")
    print(f"  Precision: {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"  F1-Score : {f1_score(y_true, preds, zero_division=0):.4f}")


df = pd.read_csv("LUC_Master_With_Fingerprints_Enriched.csv")
req = ["HIT CALL_Agonist", "HIT CALL_Viability", "ECFP6_Str", "FCFP6_Str"]
sub = df.dropna(subset=req).copy()
sub = sub[(sub["ECFP6_Str"].str.strip() != "") & (sub["FCFP6_Str"].str.strip() != "")].copy()

X_fp = np.hstack([parse_fps(sub["ECFP6_Str"]), parse_fps(sub["FCFP6_Str"])])
y_ag_raw = (sub["HIT CALL_Agonist"] == "Active").astype(int).values
y_vi_raw = (sub["HIT CALL_Viability"] == "Active").astype(int).values
y_new_target = ((sub["HIT CALL_Agonist"] == "Active") & (sub["HIT CALL_Viability"] != "Active")).astype(int).values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 1. Original Multi-Output Model (Agonist target evaluated OOF) ---
oof_orig_ag = np.zeros(len(sub))
Y_multi = np.column_stack([y_ag_raw, y_vi_raw])
for tr, va in cv.split(X_fp, y_ag_raw):
    m_rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1))
    m_xgb = MultiOutputClassifier(xgb.XGBClassifier(n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=1))
    m_rf.fit(X_fp[tr], Y_multi[tr])
    m_xgb.fit(X_fp[tr], Y_multi[tr])
    oof_orig_ag[va] = 0.5 * m_rf.predict_proba(X_fp[va])[0][:, 1] + 0.5 * m_xgb.predict_proba(X_fp[va])[0][:, 1]

t_orig = best_thresh_f1(y_ag_raw, oof_orig_ag)
summarize_metrics("Original Model (Agonist Hit Call, 5-fold OOF)", y_ag_raw, oof_orig_ag, t_orig)

# --- 2. New Single Agonist Decision Model (with OOF Cytotoxicity Feature) ---
oof_cyto = np.zeros(len(sub))
for tr, va in cv.split(X_fp, y_vi_raw):
    rf_v = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1).fit(X_fp[tr], y_vi_raw[tr])
    xgb_v = xgb.XGBClassifier(n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=1).fit(X_fp[tr], y_vi_raw[tr])
    oof_cyto[va] = 0.5 * rf_v.predict_proba(X_fp[va])[:, 1] + 0.5 * xgb_v.predict_proba(X_fp[va])[:, 1]

X_new = np.column_stack([X_fp, oof_cyto.astype(np.float32)])
oof_new_ag = np.zeros(len(sub))
for tr, va in cv.split(X_new, y_new_target):
    rf_a = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1).fit(X_new[tr], y_new_target[tr])
    xgb_a = xgb.XGBClassifier(n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=1).fit(X_new[tr], y_new_target[tr])
    oof_new_ag[va] = 0.5 * rf_a.predict_proba(X_new[va])[:, 1] + 0.5 * xgb_a.predict_proba(X_new[va])[:, 1]

t_new = best_thresh_f1(y_new_target, oof_new_ag)
summarize_metrics("New Model (Confirmed Agonist w/ Cytotoxicity Feature, 5-fold OOF)", y_new_target, oof_new_ag, t_new)