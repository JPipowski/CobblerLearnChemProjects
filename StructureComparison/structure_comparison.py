import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Bio.PDB import PDBParser, Superimposer, PDBIO
import os
import subprocess
import json


def select_files():
    """Opens a file dialog forced to the front to select the experimental and comparison PDB files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main root window
    root.attributes('-topmost', True)  # Force dialog box to the front of all windows
    root.lift()
    root.focus_force()

    print("Please select the Experimental PDB file (TRZ)...")
    exp_file = filedialog.askopenfilename(
        master=root,
        title="Select Experimental PDB File (TRZ)",
        filetypes=[("PDB Files", "*.pdb *.ent"), ("All Files", "*.*")]
    )
    if not exp_file:
        print("Experimental file selection cancelled.")
        root.destroy()
        return None, None

    print("Please select the Comparison PDB file...")
    model_file = filedialog.askopenfilename(
        master=root,
        title="Select Comparison PDB File",
        filetypes=[("PDB Files", "*.pdb *.ent"), ("All Files", "*.*")]
    )
    if not model_file:
        print("Comparison file selection cancelled.")
        root.destroy()
        return None, None

    root.destroy()
    return exp_file, model_file


def analyze_structure(structure, name):
    """Computes basic statistics for a Biopython structure object using NumPy."""
    atoms = list(structure.get_atoms())
    residues = list(structure.get_residues())

    coords = np.array([atom.get_coord() for atom in atoms])
    b_factors = np.array([atom.get_bfactor() for atom in atoms])

    center = np.mean(coords, axis=0)
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)

    stats = {
        'Name': name,
        'Total Atoms': len(atoms),
        'Total Residues': len(residues),
        'Geometric Center': f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})",
        'Bounding Box Min': f"({min_coords[0]:.2f}, {min_coords[1]:.2f}, {min_coords[2]:.2f})",
        'Bounding Box Max': f"({max_coords[0]:.2f}, {max_coords[1]:.2f}, {max_coords[2]:.2f})",
        'Average B-Factor': f"{np.mean(b_factors):.2f}",
        'Max B-Factor': f"{np.max(b_factors):.2f}"
    }
    return stats, coords, b_factors


def extract_plddt(structure):
    """Extracts per-residue pLDDT confidence scores from C-alpha atoms (stored in B-factor)."""
    res_labels = []
    plddt_scores = []

    for res in structure.get_residues():
        if "CA" in res:
            ca_atom = res["CA"]
            plddt_scores.append(ca_atom.get_bfactor())
            res_labels.append(f"{res.get_resname()}{res.get_id()[1]}")

    return res_labels, np.array(plddt_scores)


def generate_pymol_script(ref_filename, aligned_filename, script_name="visualize_superposition.pml"):
    """Generates a PyMOL script to load structures, color Experimental blue, and Model red."""
    script_content = f"""# PyMOL Superposition Visualization Script
# Experimental (Reference): Blue
# AlphaFold Prediction (Aligned): Red

reinitialize

# Load structures
load {ref_filename}, experimental
load {aligned_filename}, alphafold_model

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
"""
    with open(script_name, "w", encoding="utf-8") as f:
        f.write(script_content)
    print(f"  • PyMOL script successfully generated: {script_name}")


def generate_html_visualization(ref_filename, aligned_filename, html_name="structure_comparison.html"):
    """Generates an HTML viewer using a clean template to prevent formatting bugs."""
    try:
        with open(ref_filename, "r", encoding="utf-8") as f:
            ref_data = f.read()
        with open(aligned_filename, "r", encoding="utf-8") as f:
            aligned_data = f.read()
    except Exception as e:
        print(f"  • Note: Could not embed files into HTML automatically: {e}")
        return

    ref_data_json = json.dumps(ref_data)
    aligned_data_json = json.dumps(aligned_data)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Protein Structure Superposition Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f4f4f9; text-align: center; }
        h1 { color: #333; font-size: 20px; margin-bottom: 5px; }
        #debug-status { background: #f6ffed; border: 1px solid #b7eb8f; padding: 10px; margin: 10px auto; width: 880px; font-size: 14px; color: #52c41a; border-radius: 6px; font-weight: bold; }
        .legend { font-size: 14px; margin-bottom: 15px; font-weight: bold; }
        .blue { color: #0053D6; }
        .red { color: #FF7D45; }
        #container { width: 900px; height: 550px; margin: 0 auto; position: relative; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background: white; }
    </style>
</head>
<body>
    <h1>Protein Superposition: Experimental vs. Prediction</h1>
    <div id="debug-status">Status: 3D Structures rendered successfully.</div>
    <div class="legend">
        <span class="blue">&#9632; Experimental Structure (Blue)</span> | 
        <span class="red">&#9632; AlphaFold Prediction (Red)</span>
    </div>
    <div id="container"></div>

    <script>
        window.addEventListener('load', function () {
            let element = document.getElementById('container');
            let config = { backgroundColor: 'white' };
            let viewer = $3Dmol.createViewer(element, config);

            viewer.addModel(__REF_DATA_JSON__, "pdb");
            viewer.setStyle({model: 0}, {cartoon: {color: '#0053D6'}});

            viewer.addModel(__ALIGNED_DATA_JSON__, "pdb");
            viewer.setStyle({model: 1}, {cartoon: {color: '#FF7D45'}});

            viewer.zoomTo();
            viewer.render();
        });
    </script>
</body>
</html>
"""
    html_content = html_template.replace("__REF_DATA_JSON__", ref_data_json).replace("__ALIGNED_DATA_JSON__",
                                                                                     aligned_data_json)

    with open(html_name, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  • Browser-based HTML viewer successfully generated: {html_name}")


def open_in_brave(html_path):
    """Directly launches Brave browser via subprocess, bypassing Windows registry errors."""
    abs_path = os.path.abspath(html_path)

    username = os.environ.get('USERNAME', '')
    brave_paths = [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        rf"C:\Users\{username}\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"
    ]

    launched = False
    for path in brave_paths:
        if os.path.exists(path):
            try:
                subprocess.Popen([path, abs_path])
                print("  • Opened interactive 3D viewer in Brave browser.")
                launched = True
                break
            except Exception:
                pass

    if not launched:
        try:
            os.startfile(abs_path)
            print("  • Opened interactive 3D viewer via system default handler.")
        except Exception:
            print(f"  • Could not auto-open browser. Please open this file manually: {abs_path}")


def main():
    # 1. File Selection Dialog (Forced to front)
    exp_path, model_path = select_files()
    if not exp_path or not model_path:
        return

    # 2. Parse PDB files using Biopython
    parser = PDBParser(QUIET=True)
    exp_structure = parser.get_structure("Experimental", exp_path)
    model_structure = parser.get_structure("Comparison", model_path)

    # 3. Analyze basic statistics
    exp_stats, exp_coords, exp_bf = analyze_structure(exp_structure, "Experimental Structure (TRZ)")
    model_stats, model_coords, model_bf = analyze_structure(model_structure, "Comparison Structure")

    # Print Report
    print("\n" + "=" * 50)
    print(" PDB STRUCTURE ANALYSIS REPORT")
    print("=" * 50)
    for stats in [exp_stats, model_stats]:
        print(f"\n[{stats['Name']}]")
        for k, v in stats.items():
            if k != 'Name':
                print(f"  • {k}: {v}")

    # 4. Extract pLDDT confidence scores for the comparison structure
    model_res_labels, model_plddt = extract_plddt(model_structure)
    if len(model_plddt) > 0:
        print(f"\n[AlphaFold pLDDT Confidence Analysis]")
        print(f"  • Mean pLDDT Score: {np.mean(model_plddt):.2f}")
        print(f"  • Max pLDDT Score:  {np.max(model_plddt):.2f}")
        print(f"  • Min pLDDT Score:  {np.min(model_plddt):.2f}")

    # 5. Structural Alignment & RMSD Calculation
    exp_atoms = [atom for atom in exp_structure.get_atoms() if atom.get_name() == 'CA']
    model_atoms = [atom for atom in model_structure.get_atoms() if atom.get_name() == 'CA']

    per_residue_distances = []
    residue_labels = []

    output_pdb = "aligned_comparison_output.pdb"
    if len(exp_atoms) == len(model_atoms):
        super_imposer = Superimposer()
        super_imposer.set_atoms(exp_atoms, model_atoms)
        super_imposer.apply(model_structure.get_atoms())

        print(f"\n[Alignment & RMSD Statistics]")
        print(f"  • Overall C-alpha RMSD: {super_imposer.rms:.3f} Å")

        for exp_a, mod_a in zip(exp_atoms, model_atoms):
            dist = np.linalg.norm(exp_a.get_coord() - mod_a.get_coord())
            per_residue_distances.append(dist)
            res = exp_a.get_parent()
            residue_labels.append(f"{res.get_resname()}{res.get_id()[1]}")

        io = PDBIO()
        io.set_structure(model_structure)
        io.save(output_pdb)
        print(f"  • Aligned structure successfully saved to: {output_pdb}")
    else:
        print("\n[Alignment Note] Atom count mismatch for C-alpha atoms; saving unaligned comparison structure.")
        io = PDBIO()
        io.set_structure(model_structure)
        io.save(output_pdb)

    # 6. Generate PyMOL Script & HTML Viewer Files
    print("\n[External Visualization File Generation]")
    generate_pymol_script(exp_path, output_pdb)
    generate_html_visualization(exp_path, output_pdb)
    open_in_brave("structure_comparison.html")

    # 7. Matplotlib Multi-Plot Generation (Figure 1: Standard Plots)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 11))

    # Plot 1: B-Factor Comparison
    axes[0].plot(exp_bf, label='Experimental (TRZ)', color='blue', alpha=0.7)
    axes[0].plot(model_bf, label='Comparison', color='orange', alpha=0.7)
    axes[0].set_title('Atomic B-Factor Comparison Across Structures')
    axes[0].set_xlabel('Atom Index')
    axes[0].set_ylabel('B-Factor (A^2)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Per-Residue Distance Profile
    if per_residue_distances:
        axes[1].plot(residue_labels, per_residue_distances, marker='o', color='purple', linestyle='-', linewidth=1.5)
        axes[1].set_title(r'Per-Residue C-$\alpha$ Distance Deviation After Alignment')
        axes[1].set_xlabel('Residue')
        axes[1].set_ylabel(r'Distance ($\AA$)')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # Plot 3: AlphaFold pLDDT Confidence Profile along the sequence
    if len(model_plddt) > 0:
        colors = []
        for score in model_plddt:
            if score > 90:
                colors.append('#0053D6')  # Very high
            elif score > 70:
                colors.append('#65CBF3')  # Confident
            elif score > 50:
                colors.append('#FFDB13')  # Low
            else:
                colors.append('#FF7D45')  # Very low

        axes[2].bar(model_res_labels, model_plddt, color=colors, width=0.8)
        axes[2].axhline(90, color='blue', linestyle='--', alpha=0.5, label='Very high (>90)')
        axes[2].axhline(70, color='cyan', linestyle='--', alpha=0.5, label='Confident (70-90)')
        axes[2].axhline(50, color='orange', linestyle='--', alpha=0.5, label='Low (50-70)')

        axes[2].set_title('AlphaFold pLDDT Confidence Score per Residue')
        axes[2].set_xlabel('Residue')
        axes[2].set_ylabel('pLDDT Score')
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right', fontsize=8)

        legend_handles = [
            mpatches.Patch(color='#0053D6', label='Very high (>90)'),
            mpatches.Patch(color='#65CBF3', label='Confident (70–90)'),
            mpatches.Patch(color='#FFDB13', label='Low (50–70)'),
            mpatches.Patch(color='#FF7D45', label='Very low (<50)')
        ]
        axes[2].legend(handles=legend_handles, loc='lower right', fontsize=8)

    fig.tight_layout()

    # 8. Advanced Analysis: Correlation & Distance Difference Matrix Heatmap (Figure 2)
    if len(model_plddt) > 0 and per_residue_distances:
        fig2, (ax_corr, ax_heat) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        min_len = min(len(model_plddt), len(per_residue_distances))
        plddt_subset = model_plddt[:min_len]
        dist_subset = np.array(per_residue_distances[:min_len])

        # Correlation Scatter Plot (pLDDT vs C-alpha Deviation)
        ax_corr.scatter(plddt_subset, dist_subset, color='teal', alpha=0.7, edgecolors='k')
        m, b = np.polyfit(plddt_subset, dist_subset, 1)
        ax_corr.plot(plddt_subset, m * plddt_subset + b, color='red', linestyle='--',
                     label=f'Trendline (y={m:.2f}x+{b:.2f})')

        corr_coef = np.corrcoef(plddt_subset, dist_subset)[0, 1]
        ax_corr.set_title(r'pLDDT vs. C-$\alpha$ Deviation (Pearson r = %.2f)' % corr_coef)
        ax_corr.set_xlabel('AlphaFold pLDDT Confidence Score')
        ax_corr.set_ylabel(r'C-$\alpha$ Deviation ($\AA$)')
        ax_corr.grid(True, linestyle='--', alpha=0.5)
        ax_corr.legend(loc='upper right')

        # Distance Difference Matrix Heatmap
        exp_coords_ca = np.array([a.get_coord() for a in exp_atoms[:min_len]])
        model_coords_ca = np.array([a.get_coord() for a in model_atoms[:min_len]])

        dist_matrix_exp = np.linalg.norm(exp_coords_ca[:, np.newaxis, :] - exp_coords_ca[np.newaxis, :, :], axis=2)
        dist_matrix_model = np.linalg.norm(model_coords_ca[:, np.newaxis, :] - model_coords_ca[np.newaxis, :, :],
                                           axis=2)
        dist_diff_matrix = np.abs(dist_matrix_exp - dist_matrix_model)

        cax = ax_heat.imshow(dist_diff_matrix, cmap='plasma', interpolation='nearest')
        fig2.colorbar(cax, ax=ax_heat, label=r'Absolute Distance Difference ($\AA$)')
        ax_heat.set_title('Pairwise Distance Difference Matrix Heatmap')
        ax_heat.set_xlabel('Residue Index')
        ax_heat.set_ylabel('Residue Index')

        fig2.tight_layout()

    print("\nDisplaying Matplotlib analysis plots & correlation heatmap...")
    plt.show()


if __name__ == "__main__":
    main()