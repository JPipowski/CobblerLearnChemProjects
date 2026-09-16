import pubchempy as pcp

print("PubChem Compound Search (type 'exit' or 'q' to quit)")

while True:
    compound_name = input("\nEnter compound name: ").strip()

    # Check for exit command
    if compound_name.lower() in ["exit", "q"]:
        print("Exiting search. Goodbye!")
        break

    # Skip empty input
    if not compound_name:
        continue

    compounds = pcp.get_compounds(compound_name, "name")

    if compounds:
        compound = compounds[0]
        print(f"\nCompound: {compound_name.title()}")
        print(f"IUPAC Name: {compound.iupac_name}")
        print(f"Molecular Formula: {compound.molecular_formula}")
        print(f"Molecular Weight: {compound.molecular_weight}")
        print(f"SMILES String: {compound.connectivity_smiles}")
        print(f"H-Bond Donors: {compound.h_bond_donor_count}")
        print(f"H-Bond Acceptors: {compound.h_bond_acceptor_count}")
        print(f"XLogP: {compound.xlogp}")
    else:
        print(f"Compound '{compound_name}' not found.")