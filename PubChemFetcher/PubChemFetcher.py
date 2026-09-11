import pubchempy as pcp

compound_name = input("Enter compound name: ").strip()
compounds = pcp.get_compounds(compound_name, 'name')

if compounds:
    compound = compounds[0]
    print(f"\nCompound: {compound_name.title()}")
    print(f"IUPAC Name: {compound.iupac_name}")
    print(f"Molecular Formula: {compound.molecular_formula}")
    print(f"Molecular Weight: {compound.molecular_weight}")
    print(f"SMILES String: {compound.connectivity_smiles}")

else:
    print(f"Compound '{compound_name}' not found.")