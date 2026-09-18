import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def get_pubchem_data(compound_name):
    """Query PubChem for SMILES string and XLogP value."""
    try:
        results = pcp.get_compounds(compound_name, 'name')
        if results:
            compound = results[0]
            return {
                "smiles": compound.connectivity_smiles,
                "xlogp_pubchem": compound.xlogp
            }
    except Exception as e:
        print(f"PubChem lookup error: {e}")
    return None


def calculate_molecular_properties(mol):
    """Calculate physicochemical descriptors including RDKit Crippen LogP."""
    # RDKit's local LogP approximation (Wildman-Crippen method)
    logp_rdkit = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rotors = Lipinski.NumRotatableBonds(mol)

    # Delaney (ESOL) calculation using RDKit LogP
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    heavy_atoms = mol.GetNumHeavyAtoms()
    ap = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0

    log_solubility = 0.16 - (0.63 * logp_rdkit) - (0.0062 * mw) + (0.066 * rotors) - (0.74 * ap)

    return {
        "exact_mw": Descriptors.ExactMolWt(mol),
        "logp_rdkit": logp_rdkit,
        "h_donors": Lipinski.NumHDonors(mol),
        "h_acceptors": Lipinski.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "log_s": log_solubility
    }


def main():
    print("--- Molecule Property & XLogP Analyzer ---")
    print("Type 'quit' or 'exit' at any time to stop.\n")

    while True:
        compound_name = input("Enter compound name: ").strip()

        if compound_name.lower() in ['quit', 'exit']:
            print("Exiting program. Goodbye!")
            break

        if not compound_name:
            print("Please enter a valid compound name.\n")
            continue

        pubchem_data = get_pubchem_data(compound_name)
        if not pubchem_data:
            print(f"Error: Compound '{compound_name}' could not be found on PubChem.\n")
            continue

        smiles = pubchem_data["smiles"]
        pubchem_xlogp = pubchem_data["xlogp_pubchem"]

        print(f"Retrieved SMILES for {compound_name}: {smiles}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Error: RDKit could not parse the provided SMILES string.\n")
            continue

        props = calculate_molecular_properties(mol)

        print("\n--- Physicochemical Properties ---")
        print(f"PubChem XLogP3:                  {pubchem_xlogp if pubchem_xlogp is not None else 'N/A'}")
        print(f"RDKit MolLogP (Crippen):        {props['logp_rdkit']:.2f}")
        print(f"Exact Molecular Weight:         {props['exact_mw']:.4f}")
        print(f"Hydrogen Bond Donors:           {props['h_donors']}")
        print(f"Hydrogen Bond Acceptors:        {props['h_acceptors']}")
        print(f"TPSA:                           {props['tpsa']:.2f} Å²")
        print(f"Estimated Aqueous Sol. (LogS):  {props['log_s']:.2f} log(mol/L)\n")


if __name__ == "__main__":
    main()