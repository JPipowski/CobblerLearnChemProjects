import pubchempy as pcp

properties = pcp.get_properties(['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES'], 'theobromine', 'name')

if properties:
    compound_props = properties[0]
    print(f"Molecular Formula: {compound_props.get('MolecularFormula')}")
    print(f"Molecular Weight: {compound_props.get('MolecularWeight')}")
    print(f"SMILES String: {compound_props.get('CanonicalSMILES')}")
else:
    print("Compound not found.")