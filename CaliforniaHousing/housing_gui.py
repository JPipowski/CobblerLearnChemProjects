import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

print("Loading dataset and running 5-fold cross-validation...")
housing = fetch_california_housing(as_frame=True)

# 1. Keep only the requested features mapped to the model columns
selected_features = ['MedInc', 'HouseAge', 'AveBedrms', 'Latitude', 'Longitude']
X = housing.data[selected_features]
y = housing.target

# 2. Initialize and evaluate the Random Forest Regressor using 5-Fold CV
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

print(f"Mean CV R² Score: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# 3. Train the final model on the selected features
rf_model.fit(X, y)
print("Model fully trained! Launching GUI...")

# 4. Create the Tkinter Window
root = tk.Tk()
root.title("California Housing Price Predictor")
root.geometry("480x620")
root.resizable(False, False)

# Title Label inside GUI
title_label = tk.Label(root, text="California Housing Price Predictor", font=("Arial", 14, "bold"))
title_label.pack(pady=12)

# Frame for feature sliders
form_frame = tk.Frame(root)
form_frame.pack(pady=5)

# Feature specifications: (Col Name, Label Text, Min, Max, Resolution, Default Value)
features_info = [
    ("MedInc", "Median Income:", 0.5, 15.0, 0.1, 5.5),
    ("HouseAge", "House Age (years):", 1.0, 52.0, 1.0, 25.0),
    ("AveBedrms", "Number of Bedrooms:", 0.5, 6.0, 0.01, 1.02),
    ("Latitude", "Latitude Coordinate:", 32.5, 42.0, 0.01, 34.05),
    ("Longitude", "Longitude Coordinate:", -124.3, -114.3, 0.01, -118.24)
]

sliders = {}

for i, (col, label_text, min_val, max_val, res, default_val) in enumerate(features_info):
    # Label for the slider
    lbl = tk.Label(form_frame, text=label_text, anchor="w", width=22, font=("Arial", 10))
    lbl.grid(row=i, column=0, sticky="w", padx=5, pady=6)

    # Horizontal Slider (Scale)
    slider = tk.Scale(
        form_frame,
        from_=min_val,
        to=max_val,
        resolution=res,
        orient=tk.HORIZONTAL,
        length=200,
        font=("Arial", 9)
    )
    slider.set(default_val)
    slider.grid(row=i, column=1, padx=5, pady=6)
    sliders[col] = slider

# Frame for Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=15)


# 5. Prediction Function
def make_prediction():
    try:
        input_data = []
        for col, _, _, _, _, _ in features_info:
            val = sliders[col].get()
            input_data.append(val)

        # Create a single-row DataFrame with the 5 input values
        sample_df = pd.DataFrame([input_data], columns=X.columns)

        # Predict price (multiply by 100,000 and format with no decimals)
        predicted_price = rf_model.predict(sample_df)[0] * 100000

        # Update result display
        result_label.config(text=f"House Price: ${predicted_price:,.0f}", fg="#2e7d32")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")


# 6. Reset Function
def reset_fields():
    for col, _, _, _, _, default_val in features_info:
        sliders[col].set(default_val)
    result_label.config(text="House Price: $0", fg="#1f3b4d")


# Predict Button
predict_btn = tk.Button(button_frame, text="Predict Price", font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
                        command=make_prediction, width=12, cursor="hand2")
predict_btn.grid(row=0, column=0, padx=8)

# Reset Button
reset_btn = tk.Button(button_frame, text="Reset", font=("Arial", 10, "bold"), bg="#f44336", fg="white",
                      command=reset_fields, width=12, cursor="hand2")
reset_btn.grid(row=0, column=1, padx=8)

# --- Boxed Result Container ---
result_container = tk.Frame(root, relief="solid", bd=1, bg="#f4f6f8")
result_container.pack(pady=15, padx=25, fill="x")

# Result Output Label (Bigger font, placed inside the box)
result_label = tk.Label(
    result_container,
    text="House Price: $0",
    font=("Arial", 16, "bold"),
    fg="#1f3b4d",
    bg="#f4f6f8"
)
result_label.pack(pady=14, padx=10)

# Run the Tkinter main event loop
root.mainloop()