import subprocess
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# ==========================================
# 1. CREATE BINARY IMAGE GRID (ASTERISK)
# ==========================================
height, width = 7, 7
grid = np.zeros((height, width), dtype=int)
center_y, center_x = height // 2, width // 2
grid[center_y, :] = 1  # Horizontal bar
grid[:, center_x] = 1  # Vertical line
np.fill_diagonal(grid, 1)  # Main diagonal
np.fill_diagonal(np.fliplr(grid), 1)  # Anti-diagonal

print("--- Binary Image Grid (Asterisk) ---")
print(grid)

# ==========================================
# 2. DEFINE FILTERS (KERNELS)
# ==========================================
filters = {
    "Horizontal Edge": np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]),
    "Vertical Edge": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    "Diagonal Edge": np.array([
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]
    ]),
    "Junction": np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
}

# ==========================================
# 3. APPLY FILTERS AND GENERATE HEATMAPS
# ==========================================
for name, kernel in filters.items():
    # Convolve filter with grid
    output = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)

    print(f"\n--- {name} Convolution Output ---")
    print(output)

    # Generate Heatmap Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(output, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Response Intensity')

    filename = name.lower().replace(" ", "_") + "_heatmap.png"
    plt.title(f'{name} Filter Response Heatmap')
    plt.xlabel('X Pixel Coordinate')
    plt.ylabel('Y Pixel Coordinate')

    # Annotate matrix values inside cells
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            plt.text(j, i, f"{output[i, j]}", ha='center', va='center', color='black', fontsize=9)

    plt.grid(False)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] '{filename}' generated successfully.")


# ==========================================
# 4. AUTOMATED GIT COMMIT AND PUSH TO GITHUB
# ==========================================
def commit_and_push_heatmaps():
    try:
        print("\n--- Pushing heatmap plots to GitHub ---")

        # Stage all generated heatmap PNG files matching the pattern
        subprocess.run(["git", "add", "*_heatmap.png"], check=True)

        # Check if there are changes to commit
        status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        if not status_result.stdout.strip():
            print("No new changes in heatmap files to commit.")
            return

        # Commit changes with a clear message
        commit_message = "Auto-update image convolution filter response heatmaps"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push to remote repository (origin)
        subprocess.run(["git", "push"], check=True)
        print("[Success] Heatmap plots successfully committed and pushed to GitHub!")

    except subprocess.CalledProcessError as e:
        print(f"[Error] Git command failed: {e}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred during git operations: {e}")


# Run the git automation
commit_and_push_heatmaps()