# PyMOL Superposition Visualization Script
# Experimental (Reference): Blue
# AlphaFold Prediction (Aligned): Red

reinitialize

# Load structures
load C:/Users/pipow/OneDrive/Documents/1TRZ_chain_A_cleaned.pdb, experimental
load aligned_comparison_output.pdb, alphafold_model

# Center view
center experimental

# Set representations to cartoon
as cartoon
color blue, experimental
color red, alphafold_model

# Smooth background and lighting
bg_color white
set ray_shadow, 0
zoom

print("Loaded and styled successfully!")
