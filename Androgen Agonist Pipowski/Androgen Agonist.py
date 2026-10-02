import re
import pandas as pd

# Define the source files and their assay identifiers
files = {
    "LUC_2": "AID_2283393_datatable_all_LUC_2.csv",
    "BLA": "AID_2283588_datatable_all_BLA.csv",
    "Viability": "AID_2283763_datatable_all_Viability.csv",
    "LUC": "AID_2283761_datatable_all_LUC.csv",
}

dfs = []

# 1. Load and process each dataset
for assay_name, file_path in files.items():
    # Read CSV skipping the PubChem metadata rows (rows 1-4)
    df = pd.read_csv(file_path, skiprows=[1, 2, 3, 4])

    # Extract DTXSID from the PubChem activity URL
    df["DTXSID"] = df["PUBCHEM_ACTIVITY_URL"].astype(str).str.extract(r"(DTXSID\d+)")

    # Select key chemical identifiers and endpoint columns
    cols_to_keep = [
        "DTXSID",
        "PUBCHEM_SID",
        "PUBCHEM_CID",
        "PUBCHEM_EXT_DATASOURCE_SMILES",
        "PUBCHEM_ACTIVITY_OUTCOME",
        "PUBCHEM_ACTIVITY_SCORE",
        "AC50",
        "HITC",
        "BMD",
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Prefix assay-specific endpoint columns to prevent name collisions
    rename_dict = {
        "PUBCHEM_ACTIVITY_OUTCOME": f"Outcome_{assay_name}",
        "PUBCHEM_ACTIVITY_SCORE": f"Score_{assay_name}",
        "AC50": f"AC50_{assay_name}",
        "HITC": f"HITC_{assay_name}",
        "BMD": f"BMD_{assay_name}",
    }
    df = df.rename(columns=rename_dict)

    # Remove duplicate DTXSIDs within individual assay tables
    df = df.drop_duplicates(subset=["DTXSID"])
    dfs.append(df)

# 2. Outer join datasets by DTXSID into a master DataFrame
master_df = dfs[0]
for next_df in dfs[1:]:
    master_df = pd.merge(
        master_df, next_df, on="DTXSID", how="outer", suffixes=("", "_dup")
    )

    # Fill missing structural/chemical values across datasets
    for col in ["PUBCHEM_SID", "PUBCHEM_CID", "PUBCHEM_EXT_DATASOURCE_SMILES"]:
        if f"{col}_dup" in master_df.columns:
            master_df[col] = master_df[col].fillna(master_df[f"{col}_dup"])
            master_df.drop(columns=[f"{col}_dup"], inplace=True)

# 3. Export to CSV file
output_filename = "master_dataset.csv"
master_df.to_csv(output_filename, index=False)

print(
    f"Successfully exported master dataset to '{output_filename}' "
    f"({len(master_df)} chemicals, {len(master_df.columns)} columns)."
)