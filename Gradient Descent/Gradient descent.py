import os
import numpy as np
import matplotlib.pyplot as plt
from git import Repo


def generate_scatter_plot(filename='scatter_plot.png'):
    """Generates a sample scatter plot."""
    x = np.random.rand(100)
    y = np.random.rand(100)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, c='blue', edgecolors='none')
    plt.title('Sample Scatter Plot')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def generate_loss_landscape(filename='loss_landscape.png'):
    """Generates a 3D loss landscape plot."""
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)
    # Simulated loss function (e.g., quadratic landscape with local minima)
    Loss = W1 ** 2 + W2 ** 2 + 0.5 * np.sin(3 * W1) * np.cos(3 * W2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W1, W2, Loss, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Loss Landscape 3D')
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss')

    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def push_to_github(repo_path, file_list, commit_message="Update plots"):
    """Commits and pushes specified files to GitHub using GitPython."""
    try:
        repo = Repo(repo_path)

        # Add files
        for file in file_list:
            repo.index.add([file])

        # Commit
        repo.index.commit(commit_message)

        # Push to default origin/main (or master)
        origin = repo.remote(name='origin')
        origin.push()
        print("Successfully pushed plots to GitHub!")

    except Exception as e:
        print(f"An error occurred while pushing to GitHub: {e}")


if __name__ == "__main__":
    # 1. Define filenames
    scatter_file = 'scatter_plot.png'
    loss_file = 'loss_landscape.png'

    # 2. Generate plots
    generate_scatter_plot(scatter_file)
    generate_loss_landscape(loss_file)

    # 3. Push to GitHub (ensure repo_path points to your local repo directory)
    repo_directory = os.getcwd()  # Or pass full path to your git repo folder
    files_to_push = [scatter_file, loss_file]

    push_to_github(repo_directory, files_to_push, commit_message="Add scatter plot and loss landscape")