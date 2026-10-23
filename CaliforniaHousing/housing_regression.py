import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Import all four models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

print("Loading dataset and running 5-fold cross-validation... Please wait.")

# 1. Load the California Housing dataset (using ALL features)
housing = fetch_california_housing(as_frame=True)
X = housing.data  # Automatically grabs all 8 features
y = housing.target

# 2. Define all models in a dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "SVM Regressor": make_pipeline(StandardScaler(), SVR())
}

# 3. Evaluate each model using 5-Fold Cross-Validation
results = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    results.append({
        "Model": name,
        "Mean R² Score": cv_scores.mean(),
        "Std Dev": cv_scores.std()
    })

# Create and sort the performance DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Mean R² Score", ascending=False).reset_index(drop=True)

print("\n" + "="*50)
print(" 5-FOLD CROSS-VALIDATION PERFORMANCE RESULTS ")
print("="*50)
print(results_df.to_string(index=False))

# 4. Train the final Random Forest model on the FULL dataset
print("\nTraining final Random Forest model on full dataset...")
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X, y)

# 5. Extract and display Feature Importances
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance Score": rf_model.feature_importances_
}).sort_values(by="Importance Score", ascending=False).reset_index(drop=True)

print("\n" + "="*50)
print(" RANDOM FOREST FEATURE IMPORTANCES (Impact Ranking) ")
print("="*50)
print(feature_importances.to_string(index=False))

# Plot and save feature importances as a bar chart
plt.figure(figsize=(10, 6))
plt.barh(feature_importances["Feature"][::-1], feature_importances["Importance Score"][::-1], color="skyblue", edgecolor="black")
plt.xlabel("Importance Score", fontsize=12)
plt.title("Feature Impact on California Housing Predictions (Random Forest)", fontsize=14)
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=300, bbox_inches="tight")
print("\nFeature importance chart saved as 'feature_importances.png'.")

# Save and plot histograms for all features once
X.hist(bins=30, figsize=(12, 10), edgecolor="black")
plt.suptitle("Distribution of All California Housing Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("all_features_distributions.png", dpi=300, bbox_inches="tight")
print("Histogram saved as 'all_features_distributions.png'.")

# 6. Interactive Multi-Prediction Loop
feature_prompts = [
    ("MedInc", "Median income in block group ($10k units, e.g., 5.5 for $55,000): "),
    ("HouseAge", "Median house age in years (e.g., 25): "),
    ("AveRooms", "Average number of rooms per household (e.g., 5.8): "),
    ("AveBedrms", "Average number of bedrooms per household (e.g., 1.02): "),
    ("Population", "Block group population (e.g., 1200): "),
    ("AveOccup", "Average number of household members (e.g., 2.9): "),
    ("Latitude", "Latitude coordinate (e.g., 34.05): "),
    ("Longitude", "Longitude coordinate (e.g., -118.24): ")
]

while True:
    print("\n" + "="*50)
    print(" NEW HOUSE PRICE PREDICTION (Type 'q' to quit) ")
    print("="*50)

    user_inputs = []
    quit_flag = False

    for col_name, prompt_text in feature_prompts:
        while True:
            val_input = input(prompt_text)
            if val_input.strip().lower() == 'q':
                quit_flag = True
                break
            try:
                val = float(val_input)
                user_inputs.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number or 'q' to quit.")
        if quit_flag:
            break

    if quit_flag:
        print("\nExiting prediction loop. Have a great day!")
        break

    # Run prediction using the fully trained Random Forest model
    sample_neighborhood = pd.DataFrame([user_inputs], columns=X.columns)
    predicted_price = rf_model.predict(sample_neighborhood)

    print("\n" + "-"*40)
    print(f"💰 Predicted House Price: ${predicted_price[0] * 100000:,.2f}")
    print("-" * 40)