import sys
import io
import base64
import numpy as np
import pandas as pd
import urllib.request
import urllib.parse
from flask import Flask, render_template_string, request, jsonify
from rdkit import Chem
from rdkit.Chem import Draw, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

app = Flask(__name__)

# --- MODEL TRAINING ON STARTUP ---
print("Training QSAR model and loading dataset...")
try:
    file_path = "LUC_Master_with_Fingerprints.csv"
    df = pd.read_csv(file_path)

    target_cols = ["HIT CALL_Agonist", "HIT CALL_Viability"]
    fp_cols = ["ECFP6_Str", "FCFP6_Str"]


    def parse_fingerprints(fp_series):
        return np.array([list(map(int, fp.split(";"))) for fp in fp_series])


    model_df = df.dropna(subset=target_cols + fp_cols).copy()
    X_ecfp = parse_fingerprints(model_df["ECFP6_Str"])
    X_fcfp = parse_fingerprints(model_df["FCFP6_Str"])
    X_train_full = np.hstack([X_ecfp, X_fcfp])

    Y_train_full = np.column_stack([
        model_df[target_cols[0]].apply(lambda x: 1 if x == "Active" else 0).values,
        model_df[target_cols[1]].apply(lambda x: 1 if x == "Active" else 0).values,
    ])

    # 5-Fold CV for Threshold Optimization
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_rf_cv = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    multi_rf_cv = MultiOutputClassifier(base_rf_cv, n_jobs=-1)
    base_xgb_cv = xgb.XGBClassifier(n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=-1)
    multi_xgb_cv = MultiOutputClassifier(base_xgb_cv, n_jobs=-1)

    y_true_ag, y_prob_ag = [], []
    y_true_vi, y_prob_vi = [], []

    for train_idx, val_idx in cv.split(X_train_full, Y_train_full[:, 0]):
        X_tr, X_va = X_train_full[train_idx], X_train_full[val_idx]
        y_tr_fold, y_va_fold = Y_train_full[train_idx], Y_train_full[val_idx]
        multi_rf_cv.fit(X_tr, y_tr_fold)
        multi_xgb_cv.fit(X_tr, y_tr_fold)
        p_rf = multi_rf_cv.predict_proba(X_va)
        p_xgb = multi_xgb_cv.predict_proba(X_va)
        y_true_ag.extend(y_va_fold[:, 0])
        y_prob_ag.extend(0.5 * p_rf[0][:, 1] + 0.5 * p_xgb[0][:, 1])
        y_true_vi.extend(y_va_fold[:, 1])
        y_prob_vi.extend(0.5 * p_rf[1][:, 1] + 0.5 * p_xgb[1][:, 1])


    def find_optimal_threshold(y_true, y_prob):
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.05, 0.95, 0.01):
            preds = (np.array(y_prob) >= thresh).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh
        return best_thresh


    opt_thresh_ag = find_optimal_threshold(y_true_ag, y_prob_ag)
    opt_thresh_vi = find_optimal_threshold(y_true_vi, y_prob_vi)

    # Train Final Model
    base_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    multi_rf = MultiOutputClassifier(base_rf, n_jobs=-1)
    base_xgb = xgb.XGBClassifier(n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=-1)
    multi_xgb = MultiOutputClassifier(base_xgb, n_jobs=-1)

    multi_rf.fit(X_train_full, Y_train_full)
    multi_xgb.fit(X_train_full, Y_train_full)

    ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048,
                                                         atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    salt_remover = SaltRemover.SaltRemover()
    uncharger = rdMolStandardize.Uncharger()
    fragment_chooser = rdMolStandardize.LargestFragmentChooser()
    print("Model ready!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


def cas_to_smiles(cas_number):
    try:
        cas_number = cas_number.strip()
        url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(cas_number)}/smiles"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read().decode('utf-8').strip()
    except Exception:
        return None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>QSAR Bioassay Prediction Dashboard (Web)</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f7f6; margin: 0; padding: 20px; color: #333; }
        .container { max-width: 1050px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .input-row { display: flex; gap: 10px; margin-bottom: 10px; }
        input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; }
        button { padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .btn-add { background: #3498db; color: white; }
        .btn-remove { background: #e74c3c; color: white; }
        .btn-run { background: #2ecc71; color: white; width: 100%; margin-top: 15px; font-size: 18px; }
        table { width: 100%; border-collapse: collapse; margin-top: 25px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: center; vertical-align: middle; }
        th { background: #2c3e50; color: white; }
        img { width: 100px; height: 100px; }
        .badge-green { background-color: #d4efdf; color: #145a32; padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; }
        .badge-red { background-color: #fadbd8; color: #78281f; padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; }
    </style>
</head>
<body>
<div class="container">
    <h1>QSAR Bioassay Prediction Dashboard</h1>
    <p>Select input type and add compounds below:</p>

    <label><input type="radio" name="input_type" value="cas" checked onclick="switchMode('cas')"> CAS RN</label>
    <label style="margin-left: 20px;"><input type="radio" name="input_type" value="smiles" onclick="switchMode('smiles')"> SMILES</label>

    <div id="inputs-container" style="margin-top: 15px;">
        <div class="input-row">
            <input type="text" class="compound-input" value="50-00-0" placeholder="Enter CAS or SMILES...">
            <button class="btn-remove" onclick="removeRow(this)">✕</button>
        </div>
    </div>

    <button class="btn-add" onclick="addRow()">+ Add Another Compound</button>
    <button class="btn-run" onclick="runPredictions()">Run Predictions</button>

    <div id="results-section">
        <table id="results-table" style="display:none;">
            <thead>
                <tr>
                    <th>Structure</th>
                    <th id="th-input">Input (CAS)</th>
                    <th>Resolved SMILES</th>
                    <th>Agonist Prob</th>
                    <th>Viability Prob</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="results-body"></tbody>
        </table>
    </div>
</div>

<script>
let mode = 'cas';
function switchMode(newMode) {
    mode = newMode;
    document.getElementById('th-input').innerText = mode === 'cas' ? 'Input (CAS)' : 'Input (SMILES)';
}

function addRow() {
    const container = document.getElementById('inputs-container');
    const div = document.createElement('div');
    div.className = 'input-row';
    div.innerHTML = `<input type="text" class="compound-input" placeholder="Enter CAS or SMILES..."><button class="btn-remove" onclick="removeRow(this)">✕</button>`;
    container.appendChild(div);
}

function removeRow(btn) {
    const rows = document.getElementsByClassName('input-row');
    if (rows.length > 1) {
        btn.parentElement.remove();
    }
}

async function runPredictions() {
    const inputs = Array.from(document.getElementsByClassName('compound-input')).map(i => i.value.trim()).filter(v => v);
    if (inputs.length === 0) { alert('Please enter at least one compound.'); return; }

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode, inputs: inputs })
    });
    const data = await response.json();

    const tbody = document.getElementById('results-body');
    tbody.innerHTML = '';
    document.getElementById('results-table').style.display = 'table';

    data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.img ? `<img src="data:image/png;base64,${row.img}">` : 'No Image'}</td>
            <td>${row.input}</td>
            <td>${row.smiles}</td>
            <td><span class="${row.ag_class}">${row.ag}</span></td>
            <td><span class="${row.vi_class}">${row.vi}</span></td>
            <td>${row.status}</td>
        `;
        tbody.appendChild(tr);
    });
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    mode = data.get("mode")
    inputs = data.get("inputs")
    results = []

    for item in inputs:
        smiles = item if mode == "smiles" else cas_to_smiles(item)
        if not smiles:
            results.append(
                {"input": item, "smiles": "Resolution Failed", "ag": "-", "ag_class": "", "vi": "-", "vi_class": "",
                 "status": "FAILED", "img": None})
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({"input": item, "smiles": smiles, "ag": "-", "ag_class": "", "vi": "-", "vi_class": "",
                            "status": "INVALID SMILES", "img": None})
            continue

        try:
            mol = salt_remover.StripMol(mol)
            mol = uncharger.uncharge(mol)
            mol = fragment_chooser.choose(mol)
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        ecfp = ecfp_gen.GetFingerprintAsNumPy(mol)
        fcfp = fcfp_gen.GetFingerprintAsNumPy(mol)
        X_new = np.hstack([ecfp, fcfp]).reshape(1, -1)

        probs_rf = multi_rf.predict_proba(X_new)
        probs_xgb = multi_xgb.predict_proba(X_new)

        prob_ag = 0.5 * probs_rf[0][:, 1][0] + 0.5 * probs_xgb[0][:, 1][0]
        prob_vi = 0.5 * probs_rf[1][:, 1][0] + 0.5 * probs_xgb[1][:, 1][0]

        ag_status = "ACTIVE" if prob_ag >= opt_thresh_ag else "INACTIVE"
        vi_status = "CYTOTOXIC" if prob_vi >= opt_thresh_vi else "SAFE"

        # Agonist: Active = Green, Inactive = Red
        ag_class = "badge-green" if ag_status == "ACTIVE" else "badge-red"
        # Viability: Safe = Green, Cytotoxic = Red
        vi_class = "badge-green" if vi_status == "SAFE" else "badge-red"

        img = Draw.MolToImage(mol, size=(120, 120))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        results.append({
            "input": item,
            "smiles": smiles,
            "ag": f"{prob_ag:.4f} ({ag_status})",
            "ag_class": ag_class,
            "vi": f"{prob_vi:.4f} ({vi_status})",
            "vi_class": vi_class,
            "status": "SUCCESS",
            "img": img_str
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)