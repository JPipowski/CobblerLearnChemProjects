import pubchempy as pcp

compounds = pcp.get_compounds('theobromine', 'name')

if compounds:
    compound = compounds[0]
    print(f"Molecular Formula: {compound.molecular_formula}")
    print(f"Molecular Weight: {compound.molecular_weight}")
    print(f"SMILES String: {compound.canonical_smiles}")
else:
    print("Compound not found.")