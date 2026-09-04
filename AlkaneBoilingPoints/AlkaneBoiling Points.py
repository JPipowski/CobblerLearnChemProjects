import matplotlib.pyplot as plt

# 1. Data for the first 10 linear alkanes
carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
boiling_points_c = [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.6, 150.8, 174.1]

# 2. Create the scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(carbons, boiling_points_c, color='crimson', edgecolor='black', s=60, zorder=3)

# 3. Add titles and labels
plt.title("Boiling Point vs. Number of Carbons in Linear Alkanes", fontsize=14, pad=12)
plt.xlabel("Number of Carbon Atoms", fontsize=12)
plt.ylabel("Boiling Point (°C)", fontsize=12)

# 4. Styling enhancements
plt.xticks(carbons)
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
plt.tight_layout()

# 5. Save the plot to the current working directory
# Note: Always place savefig() before show(), as show() clears the figure context.
plt.savefig("alkane_boiling_points.png", dpi=300, bbox_inches='tight')

# 6. Display the plot
plt.show()