import sys
import io
import os
import numpy as np
import pandas as pd
import urllib.request
import urllib.parse
from rdkit import Chem
from rdkit.Chem import Draw, SaltRemover, Descriptors, FilterCatalog
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import seaborn as sns

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTabWidget, QMessageBox,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QStatusBar, QProgressBar, QSlider
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def cas_to_smiles(cas_number):
    try:
        cas_number = cas_number.strip()
        url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(cas_number)}/smiles"
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            smiles = response.read().decode('utf-8').strip()
            return smiles
    except Exception:
        return None


def validate_molecule_structure(mol, min_mw=10.0, max_mw=1000.0, min_heavy_atoms=2, max_heavy_atoms=100):
    """
    Validates an RDKit molecule object for chemical sanity, size boundaries,
    and structural alerts before running QSAR inference.
    """
    if mol is None:
        return False, "FAILED", "Could not parse SMILES string into a molecular graph."

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return False, "FAILED", f"Sanitization/Valence Error: {str(e)}"

    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy < min_heavy_atoms:
        return False, "FAILED", f"Molecule too small ({num_heavy} heavy atoms; min: {min_heavy_atoms})"
    if num_heavy > max_heavy_atoms:
        return False, "FAILED", f"Molecule too large ({num_heavy} heavy atoms; max: {max_heavy_atoms})"

    mw = Descriptors.ExactMolWt(mol)
    if not (min_mw <= mw <= max_mw):
        return False, "FAILED", f"Molecular weight ({mw:.1f} Da) is outside training domain ({min_mw}-{max_mw} Da)."

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    if catalog.HasMatch(mol):
        match_entry = catalog.GetFirstMatch(mol)
        alert_desc = match_entry.GetDescription() if match_entry else "General PAINS alert"
        return True, "WARNING", f"Structural Alert (PAINS): {alert_desc}"

    return True, "SUCCESS", "Passed all sanity and domain checks."


class ModelTrainingWorker(QThread):
    finished = pyqtSignal(object, object, float, float, object)
    error = pyqtSignal(str)

    def run(self):
        try:
            file_path = resource_path("LUC_Master_with_Fingerprints.csv")
            if not os.path.exists(file_path):
                self.error.emit(f"Dataset not found at: {file_path}")
                return

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

            base_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
            multi_rf = MultiOutputClassifier(base_rf, n_jobs=-1)

            base_xgb = xgb.XGBClassifier(n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=-1)
            multi_xgb = MultiOutputClassifier(base_xgb, n_jobs=-1)

            multi_rf.fit(X_train_full, Y_train_full)
            multi_xgb.fit(X_train_full, Y_train_full)

            models = {"rf": multi_rf, "xgb": multi_xgb}
            self.finished.emit(models, Y_train_full, opt_thresh_ag, opt_thresh_vi, X_train_full)
        except Exception as e:
            self.error.emit(str(e))


class BatchPredictionWorker(QThread):
    row_finished = pyqtSignal(int, object, list, str, str, str, str)
    batch_finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, items_list, is_cas_mode, models, opt_thresh_ag, opt_thresh_vi, ecfp_gen, fcfp_gen, salt_remover,
                 uncharger, fragment_chooser):
        super().__init__()
        self.items_list = items_list
        self.is_cas_mode = is_cas_mode
        self.models = models
        self.opt_thresh_ag = opt_thresh_ag
        self.opt_thresh_vi = opt_thresh_vi
        self.ecfp_gen = ecfp_gen
        self.fcfp_gen = fcfp_gen
        self.salt_remover = salt_remover
        self.uncharger = uncharger
        self.fragment_chooser = fragment_chooser

    def run(self):
        try:
            for idx, item in enumerate(self.items_list):
                if self.is_cas_mode:
                    cas = item
                    smiles = cas_to_smiles(cas)
                    if not smiles:
                        self.row_finished.emit(idx, None, [cas, "Resolution Failed", "-", "-", "-", "FAILED"], "-", "-",
                                               "-", "FAILED")
                        continue
                else:
                    cas = None
                    smiles = item

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    col_vals = [cas, smiles] if self.is_cas_mode else [smiles]
                    col_vals.extend(["INVALID SMILES", "-", "-", "-", "FAILED"])
                    self.row_finished.emit(idx, None, col_vals, "-", "-", "-", "FAILED")
                    continue

                try:
                    mol = self.salt_remover.StripMol(mol)
                    mol = self.uncharger.uncharge(mol)
                    mol = self.fragment_chooser.choose(mol)
                except Exception:
                    pass

                is_valid, status_code, status_msg = validate_molecule_structure(mol)
                if not is_valid:
                    col_vals = [cas, smiles] if self.is_cas_mode else [smiles]
                    col_vals.extend(["-", "-", status_msg, status_code])
                    self.row_finished.emit(idx, mol, col_vals, "-", "-", "-", status_code)
                    continue

                ecfp = self.ecfp_gen.GetFingerprintAsNumPy(mol)
                fcfp = self.fcfp_gen.GetFingerprintAsNumPy(mol)
                X_new = np.hstack([ecfp, fcfp]).reshape(1, -1)

                probs_rf = self.models["rf"].predict_proba(X_new)
                probs_xgb = self.models["xgb"].predict_proba(X_new)

                prob_ag_active = 0.5 * probs_rf[0][:, 1][0] + 0.5 * probs_xgb[0][:, 1][0]
                prob_ag_inactive = 1.0 - prob_ag_active
                prob_vi_cytotoxic = 0.5 * probs_rf[1][:, 1][0] + 0.5 * probs_xgb[1][:, 1][0]
                prob_vi_safe = 1.0 - prob_vi_cytotoxic

                ag_status = "ACTIVE" if prob_ag_active >= self.opt_thresh_ag else "INACTIVE"
                vi_status = "CYTOTOXIC" if prob_vi_cytotoxic >= self.opt_thresh_vi else "SAFE"

                if ag_status == "ACTIVE" and vi_status == "SAFE":
                    ar_call = "Confirmed AR Agonist"
                elif ag_status == "ACTIVE" and vi_status == "CYTOTOXIC":
                    ar_call = "Inconclusive (Cytotoxic)"
                else:
                    ar_call = "Not an AR Agonist"

                p_ag_str = f"Act: {prob_ag_active:.4f}\nInact: {prob_ag_inactive:.4f} [{ag_status}]"
                p_vi_str = f"Cyt: {prob_vi_cytotoxic:.4f}\nSafe: {prob_vi_safe:.4f} [{vi_status}]"

                col_vals = [cas, smiles, p_ag_str, p_vi_str, ar_call, status_code] if self.is_cas_mode else [smiles,
                                                                                                             p_ag_str,
                                                                                                             p_vi_str,
                                                                                                             ar_call,
                                                                                                             status_code]
                self.row_finished.emit(idx, mol, col_vals, ag_status, vi_status, ar_call, status_code)

            self.batch_finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ROCTab(QWidget):
    def __init__(self, X_train_full, y_true_ag, models, opt_thresh_ag, main_app_ref):
        super().__init__()
        self.X_train_full = X_train_full
        self.y_true = y_true_ag
        self.models = models
        self.default_thresh = opt_thresh_ag
        self.current_thresh = opt_thresh_ag
        self.main_app_ref = main_app_ref

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Interactive ROC Curve & Confusion Matrix Explorer</b><br>Move slider to test thresholds, update live screening, or export plots and metrics."))

        plots_layout = QHBoxLayout()

        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        plots_layout.addWidget(self.canvas)

        self.fig_cm = Figure(figsize=(4, 4))
        self.canvas_cm = FigureCanvasQTAgg(self.fig_cm)
        plots_layout.addWidget(self.canvas_cm)

        layout.addLayout(plots_layout)

        slider_layout = QHBoxLayout()
        self.slider_label = QLabel(f"Selected Threshold: {self.current_thresh:.2f} (Default F1-Optimal)")
        slider_layout.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(5, 95)
        self.slider.setValue(int(self.default_thresh * 100))
        self.slider.setTickInterval(5)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self.update_threshold_preview)
        slider_layout.addWidget(self.slider)

        layout.addLayout(slider_layout)

        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply as Active Screening Threshold")
        self.apply_btn.setStyleSheet("background-color: #d4efdf; color: #145a32; font-weight: bold; padding: 6px;")
        self.apply_btn.clicked.connect(self.apply_threshold)
        btn_layout.addWidget(self.apply_btn)

        self.export_plot_btn = QPushButton("Export ROC Plot")
        self.export_plot_btn.clicked.connect(self.export_roc_curve)
        btn_layout.addWidget(self.export_plot_btn)

        self.export_cm_btn = QPushButton("Export Metrics Summary")
        self.export_cm_btn.clicked.connect(self.export_metrics_report)
        btn_layout.addWidget(self.export_cm_btn)

        self.reset_btn = QPushButton("Reset to Default")
        self.reset_btn.clicked.connect(lambda: self.slider.setValue(int(self.default_thresh * 100)))
        btn_layout.addWidget(self.reset_btn)

        layout.addLayout(btn_layout)

        self.metrics_label = QLabel()
        self.metrics_label.setStyleSheet(
            "padding: 8px; background-color: #f8f9f9; border: 1px solid #d5d8dc; border-radius: 4px;")
        layout.addWidget(self.metrics_label)

        self.compute_roc_curve()
        self.plot_roc()
        self.update_threshold_preview(int(self.default_thresh * 100))

    def compute_roc_curve(self):
        multi_rf = self.models["rf"]
        multi_xgb = self.models["xgb"]

        p_rf = multi_rf.predict_proba(self.X_train_full)
        p_xgb = multi_xgb.predict_proba(self.X_train_full)

        self.y_scores = 0.5 * p_rf[0][:, 1] + 0.5 * p_xgb[0][:, 1]
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_scores)
        self.roc_auc = auc(self.fpr, self.tpr)

    def plot_roc(self):
        self.ax = self.fig.add_subplot(111)
        self.ax.clear()

        self.ax.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC (AUC = {self.roc_auc:.3f})')
        self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")

        self.update_marker(self.default_thresh)

        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.05])
        self.ax.set_xlabel('False Positive Rate')
        self.ax.set_ylabel('True Positive Rate')
        self.ax.set_title('Ensemble ROC Curve')
        self.ax.legend(loc="lower right", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_threshold_preview(self, value):
        self.current_thresh = value / 100.0
        is_default = abs(self.current_thresh - self.default_thresh) < 0.005
        suffix = " (Default F1-Optimal)" if is_default else ""
        self.slider_label.setText(f"Selected Threshold: {self.current_thresh:.2f}{suffix}")

        self.update_metrics_display(self.current_thresh)
        self.update_marker(self.current_thresh)
        self.plot_confusion_matrix(self.current_thresh)

    def plot_confusion_matrix(self, thresh):
        preds = (self.y_scores >= thresh).astype(int)
        cm = confusion_matrix(self.y_true, preds)

        self.ax_cm = self.fig_cm.add_subplot(111)
        self.ax_cm.clear()

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=self.ax_cm, annot_kws={"size": 12})
        self.ax_cm.set_title(f'Confusion Matrix (@ {thresh:.2f})', fontsize=10)
        self.ax_cm.set_xlabel('Predicted Label', fontsize=9)
        self.ax_cm.set_ylabel('True Label', fontsize=9)
        self.ax_cm.xaxis.set_ticklabels(['Inactive', 'Active'], fontsize=8)
        self.ax_cm.yaxis.set_ticklabels(['Inactive', 'Active'], fontsize=8)

        self.fig_cm.tight_layout()
        self.canvas_cm.draw()

    def update_metrics_display(self, thresh):
        preds = (self.y_scores >= thresh).astype(int)
        cm = confusion_matrix(self.y_true, preds)
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        self.metrics_label.setText(
            f"<b>Threshold {thresh:.2f} Performance:</b>&nbsp;&nbsp;"
            f"Sensitivity (Recall): <b>{tpr:.3f}</b>&nbsp;|&nbsp;"
            f"Precision: <b>{precision:.3f}</b>&nbsp;|&nbsp;"
            f"F1-Score: <b>{f1:.3f}</b>&nbsp;|&nbsp;"
            f"FPR: <b>{fpr:.3f}</b><br>"
            f"<i>Counts:</i> True Pos: {tp}, False Pos: {fp}, True Neg: {tn}, False Neg: {fn}"
        )

    def update_marker(self, thresh):
        idx = np.argmin(np.abs(self.thresholds - thresh))
        marker_fpr = self.fpr[idx]
        marker_tpr = self.tpr[idx]

        if hasattr(self, 'marker_point') and self.marker_point:
            self.marker_point.remove()

        self.marker_point = self.ax.scatter([marker_fpr], [marker_tpr], color='red', s=70, zorder=5)
        self.canvas.draw()

    def apply_threshold(self):
        self.main_app_ref.opt_thresh_ag = self.current_thresh
        self.main_app_ref.statusBar.showMessage(
            f"Active AR Agonist Threshold successfully updated to {self.current_thresh:.2f}", 6000)
        QMessageBox.information(self, "Threshold Updated",
                                f"Live screening threshold updated to {self.current_thresh:.2f}.\n\nAll subsequent batch predictions will use this new cutoff.")

    def export_roc_curve(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export ROC Plot", "roc_curve_plot.png",
                                              "PNG Files (*.png);;PDF Files (*.pdf)")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Success", f"ROC curve plot exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export ROC curve:\n{str(e)}")

    def export_metrics_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Metrics Report", "model_metrics_report.txt",
                                              "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            preds = (self.y_scores >= self.current_thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_true, preds).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            report_text = f"""QSAR Model Performance & Confusion Matrix Report
==================================================
Threshold Evaluated: {self.current_thresh:.2f}
ROC AUC: {self.roc_auc:.4f}

Confusion Matrix Counts:
- True Positives (TP): {tp}
- False Positives (FP): {fp}
- True Negatives (TN): {tn}
- False Negatives (FN): {fn}

Key Metrics:
- Sensitivity (Recall / TPR): {tpr:.4f}
- Precision: {precision:.4f}
- F1-Score: {f1:.4f}
"""
            with open(path, 'w') as f:
                f.write(report_text)

            QMessageBox.information(self, "Success", f"Metrics report successfully saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save report:\n{str(e)}")


class QSARApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QSAR AR Agonist Prediction Dashboard (Fully Asynchronous + ROC & CM Explorer)")
        self.resize(1350, 850)

        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.statusBar.addPermanentWidget(self.progress_bar)

        self.models = None
        self.opt_thresh_ag = 0.5
        self.opt_thresh_vi = 0.5

        self.ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        self.fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=2048,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        )
        self.salt_remover = SaltRemover.SaltRemover()
        self.uncharger = rdMolStandardize.Uncharger()
        self.fragment_chooser = rdMolStandardize.LargestFragmentChooser()

        self.init_ui()
        self.start_model_training()

    def start_model_training(self):
        self.statusBar.showMessage("Training models in background (Please wait)...")
        self.progress_bar.show()
        self.worker = ModelTrainingWorker()
        self.worker.finished.connect(self.on_model_trained)
        self.worker.error.connect(self.on_training_error)
        self.worker.start()

    def on_model_trained(self, models, y_train_full, thresh_ag, thresh_vi, X_train_full):
        self.models = models
        self.opt_thresh_ag = thresh_ag
        self.opt_thresh_vi = thresh_vi
        self.progress_bar.hide()

        self.statusBar.showMessage(
            f"Model Ready | Optimized AR Agonist Threshold (Active >= {thresh_ag:.2f}) | "
            f"Optimized Viability Threshold (Cytotoxic >= {thresh_vi:.2f})"
        )

        self.roc_tab = ROCTab(X_train_full, y_train_full[:, 0], self.models, self.opt_thresh_ag, self)
        self.tabs.addTab(self.roc_tab, "Model Performance & ROC")

    def on_training_error(self, err_msg):
        self.progress_bar.hide()
        QMessageBox.critical(None, "Model Error", f"Failed to load dataset or train model:\n{err_msg}")
        sys.exit(1)

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- TAB 1: Multi-CAS Input Prediction ---
        tab1 = QWidget()
        layout1 = QVBoxLayout()
        layout1.addWidget(QLabel("Enter CAS RNs (click '+' to add more compounds):"))

        self.scroll_area_cas = QScrollArea()
        self.scroll_area_cas.setWidgetResizable(True)
        self.container_cas = QWidget()
        self.layout_cas = QVBoxLayout(self.container_cas)
        self.layout_cas.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area_cas.setWidget(self.container_cas)
        layout1.addWidget(self.scroll_area_cas)

        self.cas_rows = []
        self.add_cas_row("50-00-0")

        btn_layout_cas = QHBoxLayout()
        add_btn_cas = QPushButton("+ Add CAS RN")
        add_btn_cas.clicked.connect(lambda: self.add_cas_row(""))
        btn_layout_cas.addWidget(add_btn_cas)

        self.predict_btn_cas = QPushButton("Run CAS Predictions")
        self.predict_btn_cas.clicked.connect(self.run_batch_cas_predictions)
        btn_layout_cas.addWidget(self.predict_btn_cas)

        export_btn_cas = QPushButton("Export to CSV & Save Images")
        export_btn_cas.clicked.connect(lambda: self.export_table_to_csv(self.table_cas, "cas_predictions.csv"))
        btn_layout_cas.addWidget(export_btn_cas)

        layout1.addLayout(btn_layout_cas)

        self.table_cas = QTableWidget()
        self.table_cas.setColumnCount(7)
        self.table_cas.setHorizontalHeaderLabels(
            ["Structure", "CAS RN", "Resolved SMILES", "Agonist (Act / Inact)", "Viability (Cyt / Safe)",
             "AR Agonist Call", "Status"]
        )
        self.table_cas.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout1.addWidget(self.table_cas)
        tab1.setLayout(layout1)
        self.tabs.addTab(tab1, "Multi-CAS Screening")

        # --- TAB 2: Multi-SMILES Input Prediction ---
        tab2 = QWidget()
        layout2 = QVBoxLayout()
        layout2.addWidget(QLabel("Enter SMILES Strings (click '+' to add more compounds):"))

        self.scroll_area_smi = QScrollArea()
        self.scroll_area_smi.setWidgetResizable(True)
        self.container_smi = QWidget()
        self.layout_smi = QVBoxLayout(self.container_smi)
        self.layout_smi.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area_smi.setWidget(self.container_smi)
        layout2.addWidget(self.scroll_area_smi)

        self.smi_rows = []
        self.add_smi_row("CC(=O)OC1=CC=CC=C1C(=O)O")

        btn_layout_smi = QHBoxLayout()
        add_btn_smi = QPushButton("+ Add SMILES")
        add_btn_smi.clicked.connect(lambda: self.add_smi_row(""))
        btn_layout_smi.addWidget(add_btn_smi)

        self.predict_btn_smi = QPushButton("Run SMILES Predictions")
        self.predict_btn_smi.clicked.connect(self.run_batch_smi_predictions)
        btn_layout_smi.addWidget(self.predict_btn_smi)

        export_btn_smi = QPushButton("Export to CSV & Save Images")
        export_btn_smi.clicked.connect(lambda: self.export_table_to_csv(self.table_smi, "smiles_predictions.csv"))
        btn_layout_smi.addWidget(export_btn_smi)

        layout2.addLayout(btn_layout_smi)

        self.table_smi = QTableWidget()
        self.table_smi.setColumnCount(6)
        self.table_smi.setHorizontalHeaderLabels(
            ["Structure", "Input SMILES", "Agonist (Act / Inact)", "Viability (Cyt / Safe)", "AR Agonist Call",
             "Status"]
        )
        self.table_smi.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout2.addWidget(self.table_smi)
        tab2.setLayout(layout2)
        self.tabs.addTab(tab2, "Multi-SMILES Screening")

    def add_cas_row(self, default_value=""):
        row = QWidget()
        lyt = QHBoxLayout(row)
        lyt.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(default_value)
        lyt.addWidget(edit)
        btn = QPushButton("✕")
        btn.setFixedWidth(40)
        btn.clicked.connect(lambda: self.remove_row(row, edit, self.cas_rows, self.layout_cas))
        lyt.addWidget(btn)
        self.layout_cas.addWidget(row)
        self.cas_rows.append((row, edit))

    def add_smi_row(self, default_value=""):
        row = QWidget()
        lyt = QHBoxLayout(row)
        lyt.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(default_value)
        lyt.addWidget(edit)
        btn = QPushButton("✕")
        btn.setFixedWidth(40)
        btn.clicked.connect(lambda: self.remove_row(row, edit, self.smi_rows, self.layout_smi))
        lyt.addWidget(btn)
        self.layout_smi.addWidget(row)
        self.smi_rows.append((row, edit))

    def remove_row(self, row_widget, line_edit, row_list, layout):
        if len(row_list) <= 1:
            return
        layout.removeWidget(row_widget)
        row_widget.deleteLater()
        row_list[:] = [item for item in row_list if item[1] != line_edit]

    def run_batch_cas_predictions(self):
        if self.models is None:
            QMessageBox.warning(self, "Please Wait", "Models are still training in the background.")
            return

        cas_list = [edit.text().strip() for _, edit in self.cas_rows if edit.text().strip()]
        if not cas_list:
            QMessageBox.warning(self, "Input Error", "Please enter at least one valid CAS RN.")
            return

        self.table_cas.setRowCount(len(cas_list))
        self.predict_btn_cas.setEnabled(False)
        self.statusBar.showMessage("Running CAS batch predictions asynchronously...")
        self.progress_bar.show()

        self.batch_worker = BatchPredictionWorker(
            cas_list, True, self.models, self.opt_thresh_ag, self.opt_thresh_vi,
            self.ecfp_gen, self.fcfp_gen, self.salt_remover, self.uncharger, self.fragment_chooser
        )
        self.batch_worker.row_finished.connect(
            lambda r, m, vals, ag, vi, ar, sc: self.populate_table_row(self.table_cas, r, m, vals, ag, vi, ar, sc))
        self.batch_worker.batch_finished.connect(lambda: self.on_batch_complete(self.predict_btn_cas))
        self.batch_worker.error.connect(self.on_batch_error)
        self.batch_worker.start()

    def run_batch_smi_predictions(self):
        if self.models is None:
            QMessageBox.warning(self, "Please Wait", "Models are still training in the background.")
            return

        smi_list = [edit.text().strip() for _, edit in self.smi_rows if edit.text().strip()]
        if not smi_list:
            QMessageBox.warning(self, "Input Error", "Please enter at least one valid SMILES string.")
            return

        self.table_smi.setRowCount(len(smi_list))
        self.predict_btn_smi.setEnabled(False)
        self.statusBar.showMessage("Running SMILES batch predictions asynchronously...")
        self.progress_bar.show()

        self.batch_worker = BatchPredictionWorker(
            smi_list, False, self.models, self.opt_thresh_ag, self.opt_thresh_vi,
            self.ecfp_gen, self.fcfp_gen, self.salt_remover, self.uncharger, self.fragment_chooser
        )
        self.batch_worker.row_finished.connect(
            lambda r, m, vals, ag, vi, ar, sc: self.populate_table_row(self.table_smi, r, m, vals, ag, vi, ar, sc))
        self.batch_worker.batch_finished.connect(lambda: self.on_batch_complete(self.predict_btn_smi))
        self.batch_worker.error.connect(self.on_batch_error)
        self.batch_worker.start()

    def on_batch_complete(self, button_to_enable):
        button_to_enable.setEnabled(True)
        self.progress_bar.hide()
        self.statusBar.showMessage("Batch screening complete.", 5000)

    def on_batch_error(self, err_msg):
        self.progress_bar.hide()
        self.predict_btn_cas.setEnabled(True)
        self.predict_btn_smi.setEnabled(True)
        QMessageBox.critical(self, "Prediction Error", f"Batch screening failed:\n{err_msg}")

    def populate_table_row(self, table, row, mol, col_offsets, ag_status, vi_status, ar_call, status_code):
        if mol is not None:
            img = Draw.MolToImage(mol, size=(120, 120))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setCellWidget(row, 0, img_label)
            table.setRowHeight(row, 130)
        else:
            table.setItem(row, 0, QTableWidgetItem("No Image"))

        for idx, val in enumerate(col_offsets):
            col_idx = idx + 1
            item = QTableWidgetItem(val)

            header_label = table.horizontalHeaderItem(col_idx).text()
            if status_code == "FAILED":
                item.setBackground(QColor(250, 219, 216))
                item.setForeground(QColor(120, 40, 31))
            elif status_code == "WARNING":
                item.setBackground(QColor(252, 243, 207))
                item.setForeground(QColor(125, 102, 8))
            else:
                if "Agonist (" in header_label:
                    if ag_status == "ACTIVE":
                        item.setBackground(QColor(212, 239, 223))
                        item.setForeground(QColor(20, 90, 50))
                    else:
                        item.setBackground(QColor(250, 219, 216))
                        item.setForeground(QColor(120, 40, 31))
                elif "Viability" in header_label:
                    if vi_status == "SAFE":
                        item.setBackground(QColor(212, 239, 223))
                        item.setForeground(QColor(20, 90, 50))
                    else:
                        item.setBackground(QColor(250, 219, 216))
                        item.setForeground(QColor(120, 40, 31))
                elif "AR Agonist Call" in header_label:
                    if ar_call == "Confirmed AR Agonist":
                        item.setBackground(QColor(212, 239, 223))
                        item.setForeground(QColor(20, 90, 50))
                    elif ar_call == "Inconclusive (Cytotoxic)":
                        item.setBackground(QColor(252, 243, 207))
                        item.setForeground(QColor(125, 102, 8))
                    else:
                        item.setBackground(QColor(250, 219, 216))
                        item.setForeground(QColor(120, 40, 31))

            table.setItem(row, col_idx, item)

    def export_table_to_csv(self, table, default_filename):
        if table.rowCount() == 0:
            QMessageBox.warning(self, "Export Warning", "There are no prediction results to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Results to CSV", default_filename,
                                              "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        try:
            base_dir = os.path.dirname(path)
            base_name = os.path.splitext(os.path.basename(path))[0]
            img_folder = os.path.join(base_dir, f"{base_name}_images")
            os.makedirs(img_folder, exist_ok=True)

            rows = table.rowCount()
            cols = table.columnCount()
            headers = [table.horizontalHeaderItem(i).text() for i in range(cols)]

            smiles_col_idx = -1
            for i in range(cols):
                if "SMILES" in headers[i]:
                    smiles_col_idx = i
                    break

            data = []
            for r in range(rows):
                row_data = []
                smiles_str = None

                for c in range(cols):
                    item = table.item(r, c)
                    if item is not None:
                        val = item.text()
                        row_data.append(val)
                        if c == smiles_col_idx:
                            smiles_str = val
                    else:
                        row_data.append("")

                img_filename = ""
                if smiles_str and smiles_str not in ["Resolution Failed", "-", "FAILED"]:
                    mol = Chem.MolFromSmiles(smiles_str)
                    if mol is not None:
                        img_filename = f"structure_{r + 1}.png"
                        img_path = os.path.join(img_folder, img_filename)
                        img = Draw.MolToImage(mol, size=(300, 300))
                        img.save(img_path)

                row_data[0] = img_filename if img_filename else "No Image"
                data.append(row_data)

            df_export = pd.DataFrame(data, columns=headers)
            df_export.to_csv(path, index=False)

            QMessageBox.information(
                self, "Success",
                f"Results successfully exported to:\n{path}\n\nStructure images saved in folder:\n{img_folder}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QSARApp()
    window.show()
    sys.exit(app.exec())