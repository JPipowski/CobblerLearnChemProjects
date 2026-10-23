from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ==========================================
# PART 1: Load Dataset
# ==========================================
print("Loading Iris dataset...")
iris = load_iris(as_frame=True)
features_df = iris.data


# ==========================================
# PART 2: 3-Cluster K-Means (2D Petal Plot)
# ==========================================
print("Running 3-Cluster K-Means...")
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels_3 = kmeans_3.fit_predict(features_df)

# Create 2D Plot for 3 Clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(3):
    subset = features_df[labels_3 == cluster_id]
    plt.scatter(
        subset['petal length (cm)'],
        subset['petal width (cm)'],
        label=f'Cluster {cluster_id}',
        edgecolor='k',
        s=60,
        alpha=0.8
    )

plt.xlabel('Petal Length (cm)', fontsize=11)
plt.ylabel('Petal Width (cm)', fontsize=11)
plt.title('K-Means (3 Clusters): Petal Length vs. Petal Width', fontsize=13, fontweight='bold')
plt.legend(title='Clusters', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Save 2D Plot
plt.savefig('iris_clusters_3_petal.png', dpi=300, bbox_inches='tight')
print("-> Saved 2D plot as 'iris_clusters_3_petal.png'")
plt.close() # Close figure so plots don't overlap


# ==========================================
# PART 3: 6-Cluster K-Means (Normalized 3D Plot)
# ==========================================
print("Running 6-Cluster K-Means with Normalization...")
# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

kmeans_6 = KMeans(n_clusters=6, random_state=42, n_init='auto')
labels_6 = kmeans_6.fit_predict(scaled_features)

# Create 3D Plot for 6 Clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x_col = 'sepal length (cm)'
y_col = 'petal length (cm)'
z_col = 'petal width (cm)'

for cluster_id in range(6):
    subset = features_df[labels_6 == cluster_id]
    ax.scatter(
        subset[x_col],
        subset[y_col],
        subset[z_col],
        label=f'Cluster {cluster_id}',
        s=50,
        edgecolor='k',
        alpha=0.8
    )

ax.set_xlabel(x_col, fontsize=10)
ax.set_ylabel(y_col, fontsize=10)
ax.set_zlabel(z_col, fontsize=10)
ax.set_title('3D K-Means Clustering (6 Clusters, Normalized Features)', fontsize=12, fontweight='bold', pad=15)
ax.legend(title='Clusters', loc='upper left')

# Save 3D Plot
plt.savefig('iris_3d_clusters_6.png', dpi=300, bbox_inches='tight')
print("-> Saved 3D plot as 'iris_3d_clusters_6.png'")

# Display the 3D plot window
plt.show()
print("All tasks completed successfully!")