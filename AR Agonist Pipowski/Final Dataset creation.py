import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

MASTER_CSV = "LUC_Master_with_Fingerprints.csv"
BATCH_CSV = "Batch-Search-chemical-list-2026-11-20.csv"
OUTPUT_CSV = "LUC_Master_With_Fingerprints_Enriched.csv"

_ECFP6_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
_FCFP6_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)
_SALT_REMOVER = SaltRemover.SaltRemover()
_UNCHARGER = rdMolStandardize.Uncharger()
_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()


def clean_dtxsid(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("https://comptox.epa.gov/dashboard/chemical/details/", "", regex=False)
        .str.strip()
    )
    return cleaned.where(series.notna() & (~cleaned.isin(["", "nan", "None"])), other=pd.NA)


def is_missing(series: pd.Series) -> pd.Series:
    stripped = series.astype(str).str.strip()
    return series.isna() | stripped.isin(["", "nan", "None"])


def to_qsar_smiles(smiles: str) -> str | None:
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        mol = _SALT_REMOVER.StripMol(mol)
        mol = _LARGEST_FRAG.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_fp_strings(smiles: str) -> tuple[str | None, str | None]:
    if pd.isna(smiles) or not str(smiles).strip():
        return None, None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, None
    try:
        Chem.SanitizeMol(mol)
        ecfp = ";".join(map(str, _ECFP6_GEN.GetFingerprintAsNumPy(mol).astype(int)))
        fcfp = ";".join(map(str, _FCFP6_GEN.GetFingerprintAsNumPy(mol).astype(int)))
        return ecfp, fcfp
    except Exception:
        return None, None


def build_enriched_dataset(
    master_path: str = MASTER_CSV,
    batch_path: str = BATCH_CSV,
    output_path: str = OUTPUT_CSV,
) -> pd.DataFrame:
    master_df = pd.read_csv(master_path)
    batch_df = pd.read_csv(batch_path)

    master_ids = clean_dtxsid(master_df["DTXSID"])
    batch_ids = clean_dtxsid(batch_df["DTXSID"])

    out = master_df.copy()

    # 1. Fill ONLY missing MOLECULAR FORMULA from batch search CSV
    if "MOLECULAR FORMULA" in batch_df.columns:
        valid_mf = batch_df[~is_missing(batch_df["MOLECULAR FORMULA"])]
        mf_lookup = dict(zip(batch_ids.loc[valid_mf.index], valid_mf["MOLECULAR FORMULA"].astype(str).str.strip()))
        fill_mf_mask = is_missing(out["MOLECULAR FORMULA"]) & master_ids.isin(mf_lookup)
        out.loc[fill_mf_mask, "MOLECULAR FORMULA"] = master_ids.loc[fill_mf_mask].map(mf_lookup)

    # 2. Fill ONLY missing SMILES from batch search CSV if available
    if "SMILES" in batch_df.columns and "SMILES" in out.columns:
        valid_smi = batch_df[~is_missing(batch_df["SMILES"])]
        smi_lookup = dict(zip(batch_ids.loc[valid_smi.index], valid_smi["SMILES"].astype(str).str.strip()))
        fill_smi_mask = is_missing(out["SMILES"]) & master_ids.isin(smi_lookup)
        out.loc[fill_smi_mask, "SMILES"] = master_ids.loc[fill_smi_mask].map(smi_lookup)

    # 3. Generate missing QSAR_READY_SMILES where SMILES exists
    if "QSAR_READY_SMILES" not in out.columns:
        out["QSAR_READY_SMILES"] = pd.NA
    qsar_candidates = out.index[is_missing(out["QSAR_READY_SMILES"]) & (~is_missing(out["SMILES"]))]
    for idx in qsar_candidates:
        qsar_smi = to_qsar_smiles(out.at[idx, "SMILES"])
        if qsar_smi:
            out.at[idx, "QSAR_READY_SMILES"] = qsar_smi

    # 4. Generate missing ECFP6_Str and FCFP6_Str from QSAR_READY_SMILES (fallback to SMILES)
    for fp_col in ["ECFP6_Str", "FCFP6_Str"]:
        if fp_col not in out.columns:
            out[fp_col] = pd.NA

    need_fp_mask = is_missing(out["ECFP6_Str"]) | is_missing(out["FCFP6_Str"])
    for idx in out.index[need_fp_mask]:
        qsar_val = out.at[idx, "QSAR_READY_SMILES"]
        raw_val = out.at[idx, "SMILES"]
        src_smi = qsar_val if (pd.notna(qsar_val) and str(qsar_val).strip() not in {"", "nan", "None"}) else raw_val
        ecfp_str, fcfp_str = smiles_to_fp_strings(src_smi)
        if ecfp_str and fcfp_str:
            if is_missing(pd.Series([out.at[idx, "ECFP6_Str"]])).iloc[0]:
                out.at[idx, "ECFP6_Str"] = ecfp_str
            if is_missing(pd.Series([out.at[idx, "FCFP6_Str"]])).iloc[0]:
                out.at[idx, "FCFP6_Str"] = fcfp_str

    out.to_csv(output_path, index=False)
    print(f"Saved enriched dataset to {output_path} ({len(out):,} rows)")
    return out


if __name__ == "__main__":
    build_enriched_dataset()