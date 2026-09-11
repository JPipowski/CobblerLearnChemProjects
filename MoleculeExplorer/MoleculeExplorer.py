import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def get_smiles_from_pubchem(compound_name):
    """Query PubChem and return the canonical SMILES string for a compound."""
    results = pcp.get_compounds(compound_name, 'name')
    if results:
        return results[0].connectivity_smiles
    return None


def calculate_molecular_properties(mol):
    """Calculate various physicochemical descriptors using RDKit."""
    # Components for Delaney (ESOL) solubility estimation
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rotors = Lipinski.NumRotatableBonds(mol)

    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    heavy_atoms = mol.GetNumHeavyAtoms()
    ap = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0

    log_solubility = 0.16 - (0.63 * logp) - (0.0062 * mw) + (0.066 * rotors) - (0.74 * ap)

    return {
        "exact_mw": Descriptors.ExactMolWt(mol),
        "h_donors": Descriptors.NumHDonors(mol),
        "h_acceptors": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "log_s": log_solubility
    }


def main():
    print("--- Molecule Property Analyzer ---")
    print("Type 'quit' or 'exit' at any time to stop.\n")

    while True:
        compound_name = input("Enter compound name: ").strip()

        if compound_name.lower() in ['quit', 'exit']:
            print("Exiting program. Goodbye!")
            break

        if not compound_name:
            print("Please enter a valid compound name.\n")
            continue

        # Fetch SMILES from PubChem
        smiles = get_smiles_from_pubchem(compound_name)
        if not smiles:
            print(f"Error: Compound '{compound_name}' could not be found on PubChem.\n")
            continue

        print(f"Retrieved SMILES for {compound_name}: {smiles}")

        # Parse into RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Error: RDKit could not parse the provided SMILES string.\n")
            continue

        # Calculate and display properties
        props = calculate_molecular_properties(mol)
        print(f"Exact Molecular Weight: {props['exact_mw']:.4f}")
        print(f"Number of Hydrogen Bond Donors: {props['h_donors']}")
        print(f"Number of Hydrogen Bond Acceptors: {props['h_acceptors']}")
        print(f"TPSA: {props['tpsa']:.2f} Å²")
        print(f"Estimated Aqueous Solubility (LogS): {props['log_s']:.2f} mol/L\n")


if __name__ == "__main__":
    main()