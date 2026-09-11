import pubchempy as pcp
from pubchempy import PubChemPyError
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def get_smiles_from_pubchem(compound_name):
    """Query PubChem and return the isomeric SMILES string for a compound."""
    try:
        results = pcp.get_compounds(compound_name.strip(), 'name')
        if results:
            return results[0].smiles  # Using isomeric SMILES to preserve stereochemistry
    except (PubChemPyError, OSError):
        pass
    return None


def calculate_molecular_properties(mol):
    """Calculate various physicochemical descriptors using RDKit."""
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rotors = Lipinski.NumRotatableBonds(mol)

    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    heavy_atoms = mol.GetNumHeavyAtoms()
    ap = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0

    log_solubility = 0.16 - (0.63 * logp) - (0.0062 * mw) + (0.066 * rotors) - (0.74 * ap)

    return {
        "mw": mw,
        "exact_mw": Descriptors.ExactMolWt(mol),
        "h_donors": Descriptors.NumHDonors(mol),
        "h_acceptors": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "log_s": log_solubility
    }


def main():
    print("--- Molecular Weight & Property Comparison Tool ---")
    print("Enter multiple compound names separated by commas (e.g., aspirin, caffeine, ibuprofen).")
    print("Type 'quit' or 'exit' at any time to stop.\n")

    while True:
        user_input = input("Enter compound names: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Exiting program. Goodbye!")
            break

        if not user_input:
            print("Please enter at least one compound name.\n")
            continue

        # Split input string by commas into a list of names
        compound_names = [name.strip() for name in user_input.split(',') if name.strip()]
        comparison_data = []

        for name in compound_names:
            print(f"Fetching '{name}' from PubChem...")
            smiles = get_smiles_from_pubchem(name)
            if not smiles:
                print(f"  -> Warning: Could not find '{name}' on PubChem. Skipping.")
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  -> Warning: Could not parse SMILES for '{name}'. Skipping.")
                continue

            props = calculate_molecular_properties(mol)
            comparison_data.append({
                "name": name,
                "mw": props["mw"],
                "exact_mw": props["exact_mw"],
                "h_donors": props["h_donors"],
                "h_acceptors": props["h_acceptors"],
                "tpsa": props["tpsa"],
                "log_s": props["log_s"]
            })

        if not comparison_data:
            print("No valid compounds could be processed. Try again.\n")
            continue

        # Sort the results by molecular weight (ascending) for clear comparison
        comparison_data.sort(key=lambda x: x["mw"])

        # Display formatted comparison table
        print("\n" + "=" * 92)
        header = f"{'Compound':<20} | {'MW (g/mol)':<12} | {'Exact MW':<12} | {'H-Donors':<10} | {'H-Acceptors':<12} | {'TPSA (Å²)':<10}"
        print(header)
        print("-" * 92)
        for item in comparison_data:
            print(f"{item['name']:<20} | {item['mw']:<12.2f} | {item['exact_mw']:<12.4f} | {item['h_donors']:<10} | {item['h_acceptors']:<12} | {item['tpsa']:<10.2f}")
        print("=" * 92 + "\n")


if __name__ == "__main__":
    main()