import math
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def get_smiles_from_pubchem(compound_name):
    """Query PubChem and return the canonical SMILES string for a compound."""
    try:
        results = pcp.get_compounds(compound_name, 'name')
        if results:
            return results[0].connectivity_smiles
    except Exception as e:
        print(f"PubChem lookup error: {e}")
    return None


def calculate_flexibility_vector(mol):
    """Calculate normalized feature vector representing molecular flexibility."""
    num_rotatable = Lipinski.NumRotatableBonds(mol)
    num_bonds = mol.GetNumBonds()
    f_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    mw = Descriptors.MolWt(mol)

    rotatable_ratio = num_rotatable / num_bonds if num_bonds > 0 else 0.0
    flexibility_index = (num_rotatable / mw) * 100 if mw > 0 else 0.0

    return {
        "num_rotatable": num_rotatable,
        "f_sp3": f_sp3,
        "rotatable_ratio": rotatable_ratio,
        "num_rings": num_rings,
        "flexibility_index": flexibility_index,
        # Feature vector normalized to similar scales for distance calculation
        "vector": [
            num_rotatable / 10.0,  # Scale rotatable bonds
            f_sp3,  # Already 0.0 - 1.0
            rotatable_ratio,  # Already 0.0 - 1.0
            num_rings / 5.0,  # Scale ring count
            flexibility_index / 5.0  # Scale flexibility index
        ]
    }


def calculate_flexibility_similarity(vec1, vec2):
    """Calculate Euclidean distance-based similarity score (0% to 100%)."""
    # Euclidean Distance between feature vectors
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    # Convert distance to a similarity percentage using an exponential decay function
    similarity = math.exp(-distance) * 100
    return similarity


def main():
    print("--- Molecule Flexibility Similarity Comparison ---")
    print("Type 'quit' or 'exit' at any time to stop.\n")

    while True:
        cmp1 = input("Enter first compound name: ").strip()
        if cmp1.lower() in ['quit', 'exit']:
            break

        cmp2 = input("Enter second compound name: ").strip()
        if cmp2.lower() in ['quit', 'exit']:
            break

        if not cmp1 or not cmp2:
            print("Please enter valid compound names.\n")
            continue

        smiles1 = get_smiles_from_pubchem(cmp1)
        smiles2 = get_smiles_from_pubchem(cmp2)

        if not smiles1 or not smiles2:
            print("Error: Could not retrieve SMILES for one or both compounds.\n")
            continue

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            print("Error: RDKit could not parse one of the SMILES strings.\n")
            continue

        data1 = calculate_flexibility_vector(mol1)
        data2 = calculate_flexibility_vector(mol2)

        similarity_score = calculate_flexibility_similarity(data1["vector"], data2["vector"])

        print(f"\n--- Flexibility Analysis ---")
        print(f"{'Metric':<25} | {cmp1:<18} | {cmp2:<18}")
        print("-" * 65)
        print(f"{'Rotatable Bonds':<25} | {data1['num_rotatable']:<18} | {data2['num_rotatable']:<18}")
        print(f"{'Rotatable Bond Ratio':<25} | {data1['rotatable_ratio']:<18.2%} | {data2['rotatable_ratio']:<18.2%}")
        print(f"{'Fraction sp3 (Fsp3)':<25} | {data1['f_sp3']:<18.2f} | {data2['f_sp3']:<18.2f}")
        print(f"{'Ring Count':<25} | {data1['num_rings']:<18} | {data2['num_rings']:<18}")
        print(f"{'Flexibility Index':<25} | {data1['flexibility_index']:<18.2f} | {data2['flexibility_index']:<18.2f}")
        print("-" * 65)
        print(f"Flexibility Similarity Score: {similarity_score:.2f}%\n")


if __name__ == "__main__":
    main()