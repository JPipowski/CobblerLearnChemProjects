import sys
import io
import os
import numpy as np
import pandas as pd
import urllib.request
import urllib.parse
from rdkit import Chem
from rdkit.Chem import Draw, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTabWidget, QMessageBox,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt


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


class QSARApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QSAR Bioassay Prediction Dashboard (PyQt6)")
        self.resize(1100, 850)

        # Train model on startup
        self.init_model()

        # Setup UI
        self.init_ui()

    def init_model(self):
        try:
            file_path = resource_path("LUC_Master_with_Fingerprints.csv")
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

            self.opt_thresh_ag = find_optimal_threshold(y_true_ag, y_prob_ag)
            self.opt_thresh_vi = find_optimal_threshold(y_true_vi, y_prob_vi)

            # Train Final Model
            base_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
            self.multi_rf = MultiOutputClassifier(base_rf, n_jobs=-1)

            base_xgb = xgb.XGBClassifier(n_estimators=100, eval_metric="aucpr", random_state=42, n_jobs=-1)
            self.multi_xgb = MultiOutputClassifier(base_xgb, n_jobs=-1)

            self.multi_rf.fit(X_train_full, Y_train_full)
            self.multi_xgb.fit(X_train_full, Y_train_full)

            # RDKit Generators
            self.ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
            self.fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=3, fpSize=2048,
                atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            )
            self.salt_remover = SaltRemover.SaltRemover()
            self.uncharger = rdMolStandardize.Uncharger()
            self.fragment_chooser = rdMolStandardize.LargestFragmentChooser()

        except Exception as e:
            QMessageBox.critical(None, "Model Error", f"Failed to load dataset or train model:\n{str(e)}")
            sys.exit(1)

    def init_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

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

        predict_btn_cas = QPushButton("Run CAS Predictions")
        predict_btn_cas.clicked.connect(self.run_batch_cas_predictions)
        btn_layout_cas.addWidget(predict_btn_cas)

        export_btn_cas = QPushButton("Export to CSV & Save Images")
        export_btn_cas.clicked.connect(lambda: self.export_table_to_csv(self.table_cas, "cas_predictions.csv"))
        btn_layout_cas.addWidget(export_btn_cas)

        layout1.addLayout(btn_layout_cas)

        self.table_cas = QTableWidget()
        self.table_cas.setColumnCount(6)
        self.table_cas.setHorizontalHeaderLabels(
            ["Structure", "CAS RN", "Resolved SMILES", "Agonist Prob", "Viability Prob", "Status"]
        )
        self.table_cas.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout1.addWidget(self.table_cas)
        tab1.setLayout(layout1)
        tabs.addTab(tab1, "Multi-CAS Screening")

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

        predict_btn_smi = QPushButton("Run SMILES Predictions")
        predict_btn_smi.clicked.connect(self.run_batch_smi_predictions)
        btn_layout_smi.addWidget(predict_btn_smi)

        export_btn_smi = QPushButton("Export to CSV & Save Images")
        export_btn_smi.clicked.connect(lambda: self.export_table_to_csv(self.table_smi, "smiles_predictions.csv"))
        btn_layout_smi.addWidget(export_btn_smi)

        layout2.addLayout(btn_layout_smi)

        self.table_smi = QTableWidget()
        self.table_smi.setColumnCount(5)
        self.table_smi.setHorizontalHeaderLabels(
            ["Structure", "Input SMILES", "Agonist Prob", "Viability Prob", "Status"]
        )
        self.table_smi.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout2.addWidget(self.table_smi)
        tab2.setLayout(layout2)
        tabs.addTab(tab2, "Multi-SMILES Screening")

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

    def process_prediction(self, smiles_str):
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None, None, None, None, None, "INVALID SMILES"

        try:
            mol = self.salt_remover.StripMol(mol)
            mol = self.uncharger.uncharge(mol)
            mol = self.fragment_chooser.choose(mol)
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        ecfp = self.ecfp_gen.GetFingerprintAsNumPy(mol)
        fcfp = self.fcfp_gen.GetFingerprintAsNumPy(mol)
        X_new = np.hstack([ecfp, fcfp]).reshape(1, -1)

        probs_rf = self.multi_rf.predict_proba(X_new)
        probs_xgb = self.multi_xgb.predict_proba(X_new)

        prob_ag = 0.5 * probs_rf[0][:, 1][0] + 0.5 * probs_xgb[0][:, 1][0]
        prob_vi = 0.5 * probs_rf[1][:, 1][0] + 0.5 * probs_xgb[1][:, 1][0]

        ag_status = "ACTIVE" if prob_ag >= self.opt_thresh_ag else "INACTIVE"
        vi_status = "CYTOTOXIC" if prob_vi >= self.opt_thresh_vi else "SAFE"

        return mol, f"{prob_ag:.4f} ({ag_status})", ag_status, f"{prob_vi:.4f} ({vi_status})", vi_status, "SUCCESS"

    def populate_table_row(self, table, row, mol, col_offsets, ag_status, vi_status):
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
            if "Agonist" in header_label:
                if ag_status == "ACTIVE":
                    item.setBackground(QColor(212, 239, 223))  # Soft Green
                    item.setForeground(QColor(20, 90, 50))
                else:
                    item.setBackground(QColor(250, 219, 216))  # Soft Red
                    item.setForeground(QColor(120, 40, 31))
            elif "Viability" in header_label:
                if vi_status == "SAFE":
                    item.setBackground(QColor(212, 239, 223))  # Soft Green
                    item.setForeground(QColor(20, 90, 50))
                else:
                    item.setBackground(QColor(250, 219, 216))  # Soft Red
                    item.setForeground(QColor(120, 40, 31))

            table.setItem(row, col_idx, item)

    def run_batch_cas_predictions(self):
        cas_list = [edit.text().strip() for _, edit in self.cas_rows if edit.text().strip()]
        if not cas_list:
            QMessageBox.warning(self, "Input Error", "Please enter at least one valid CAS RN.")
            return

        self.table_cas.setRowCount(0)
        for cas in cas_list:
            row = self.table_cas.rowCount()
            self.table_cas.insertRow(row)

            smiles = cas_to_smiles(cas)
            if not smiles:
                self.table_cas.setItem(row, 0, QTableWidgetItem("Failed"))
                self.table_cas.setItem(row, 1, QTableWidgetItem(cas))
                self.table_cas.setItem(row, 2, QTableWidgetItem("Resolution Failed"))
                self.table_cas.setItem(row, 3, QTableWidgetItem("-"))
                self.table_cas.setItem(row, 4, QTableWidgetItem("-"))
                self.table_cas.setItem(row, 5, QTableWidgetItem("FAILED"))
                continue

            mol, p_ag_str, ag_status, p_vi_str, vi_status, status = self.process_prediction(smiles)
            self.populate_table_row(
                self.table_cas, row, mol,
                [cas, smiles, p_ag_str, p_vi_str, status],
                ag_status, vi_status
            )

    def run_batch_smi_predictions(self):
        smi_list = [edit.text().strip() for _, edit in self.smi_rows if edit.text().strip()]
        if not smi_list:
            QMessageBox.warning(self, "Input Error", "Please enter at least one valid SMILES string.")
            return

        self.table_smi.setRowCount(0)
        for smi in smi_list:
            row = self.table_smi.rowCount()
            self.table_smi.insertRow(row)

            mol, p_ag_str, ag_status, p_vi_str, vi_status, status = self.process_prediction(smi)
            self.populate_table_row(
                self.table_smi, row, mol,
                [smi, p_ag_str, p_vi_str, status],
                ag_status, vi_status
            )

    def export_table_to_csv(self, table, default_filename):
        if table.rowCount() == 0:
            QMessageBox.warning(self, "Export Warning", "There are no prediction results to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Results to CSV", default_filename, "CSV Files (*.csv);;All Files (*)")
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