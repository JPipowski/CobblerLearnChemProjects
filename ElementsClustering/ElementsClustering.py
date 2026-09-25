import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 1. Load the dataset from CSV
df = pd.read_csv("group_1_elements.csv")

# Set display options so all columns show cleanly in terminal print
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("--- Group 1 Elements Dataset Loaded ---")
print(df)

# Palette of colors for clusters (Cluster 0: Blue, Cluster 1: Pink, etc.)
cluster_colors = ["#1f77b4", "#e377c2", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Custom offsets for annotations to prevent label overlap
custom_offsets = {
    "H": (8, 0, "left", "center"),
    "Li": (8, 5, "left", "bottom"),
    "Na": (8, 5, "left", "bottom"),
    "K": (8, 5, "left", "bottom"),
    "Rb": (8, 5, "left", "bottom"),
    "Cs": (-12, -12, "right", "top"),
    "Fr": (10, 8, "left", "bottom"),
}

X = df[["Atomic Radius (pm)", "First Ionization Energy (kJ/mol)"]]

# 2. Interactive Loop to enter new k values
while True:
    user_input = input("\nEnter number of clusters (k) or 'stop'/'q' to quit: ").strip()

    # Exit condition
    if user_input.lower() in ["stop", "q", "quit", "exit"]:
        print("Exiting interactive clustering loop.")
        break

    # Input validation
    if not user_input.isdigit() or int(user_input) <= 0:
        print("Please enter a valid positive integer for k (e.g., 2, 3, 4).")
        continue

    k = int(user_input)
    if k > len(df):
        print(f"k cannot exceed the total number of elements ({len(df)}).")
        continue

    # 3. Fit KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Print updated DataFrame to console
    print(f"\n--- Group 1 Elements with K-Means Clusters (k={k}) ---")
    print(df)

    # 4. Create Scatter Plot
    plt.figure(figsize=(9, 6))

    # Plot cluster points (Pink & Blue for k=2, expanding palette for higher k)
    for cluster_id in range(k):
        cluster_data = df[df["Cluster"] == cluster_id]
        c_color = cluster_colors[cluster_id % len(cluster_colors)]

        plt.scatter(
            cluster_data["Atomic Radius (pm)"],
            cluster_data["First Ionization Energy (kJ/mol)"],
            color=c_color,
            label=f"Cluster {cluster_id}",
            s=100,
            zorder=3,
        )

    # Plot ALL centroids as BLACK 'X' markers
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="black",
        marker="X",
        s=250,
        label="Centroids",
        zorder=4,
    )

    # Annotate element symbols
    for idx, row in df.iterrows():
        symbol = row["Symbol"]
        x_off, y_off, ha, va = custom_offsets.get(symbol, (8, 5, "left", "bottom"))
        plt.annotate(
            symbol,
            (row["Atomic Radius (pm)"], row["First Ionization Energy (kJ/mol)"]),
            textcoords="offset points",
            xytext=(x_off, y_off),
            ha=ha,
            va=va,
            fontsize=10,
            weight="bold",
        )

    # Chart styling
    plt.title(f"K-Means Clustering of Group 1 Elements (k={k})")
    plt.xlabel("Atomic Radius (pm)")
    plt.ylabel("First Ionization Energy (kJ/mol)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6, zorder=0)

    plt.xlim(40, 320)
    plt.ylim(320, 1380)

    # Save high-resolution plot and display
    plot_filename = f"atomic_radius_vs_ionization_k{k}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved as '{plot_filename}'.")

    plt.show()