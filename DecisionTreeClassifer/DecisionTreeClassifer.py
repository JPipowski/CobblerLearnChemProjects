import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Create the DataFrame
data = {
    "Molecule": [f"Molecule {i}" for i in range(1, 13)],
    "Molecular Weight": [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    "Hydrogen Bond Donors": [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    "Hydrogen Bond Acceptors": [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    "Water Solubility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
}

df = pd.DataFrame(data).set_index("Molecule")

# 2. Separate features (X) and target variable (y)
X = df[["Molecular Weight", "Hydrogen Bond Donors", "Hydrogen Bond Acceptors"]]
y = df["Water Solubility"]

# 3. Initialize and fit the Decision Tree Classifier with max_depth
# Adjust max_depth as needed (e.g., 2, 3, or None for unlimited depth)
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
clf.fit(X, y)

# 4. Plot the decision tree
plt.figure(figsize=(10, 6))
annotations = plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Insoluble (0)", "Soluble (1)"],
    filled=True,
    rounded=True,
    fontsize=10,
)

# 5. Customize tree colors: Red for Insoluble, Green for Soluble
for artist in annotations:
    text = artist.get_text()
    if "value =" in text:
        # Extract the sample counts array [insoluble_count, soluble_count]
        value_line = [line for line in text.split("\n") if "value =" in line][0]
        values = [
            float(x)
            for x in value_line.split("=")[1].strip(" []").split(",")
            if x.strip()
        ]

        total = sum(values)
        if total > 0:
            p_insoluble = values[0] / total
            p_soluble = values[1] / total

            # Pure red (#d9534f) for insoluble, Pure green (#5cb85c) for soluble
            if p_soluble > p_insoluble:
                alpha = p_soluble
                color = mcolors.to_rgba("#5cb85c", alpha=alpha)
            elif p_insoluble > p_soluble:
                alpha = p_insoluble
                color = mcolors.to_rgba("#d9534f", alpha=alpha)
            else:
                # 50/50 mix node (white/light gray)
                color = (0.95, 0.95, 0.95, 1.0)

            artist.get_bbox_patch().set_facecolor(color)
            artist.get_bbox_patch().set_edgecolor("#333333")

plt.title(
    f"Decision Tree for Predicting Water Solubility (max_depth={clf.max_depth})"
)

# 6. Save the figure
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
print("Image successfully saved as 'decision_tree.png' in current directory.")

plt.show()

# 7. View Feature Importances
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": clf.feature_importances_}
).sort_values("Importance", ascending=False)

print("\n--- Feature Importances ---")
print(importance_df.to_string(index=False))