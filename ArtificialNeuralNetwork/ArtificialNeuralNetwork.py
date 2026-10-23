import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. CREATE DATASET
# ==========================================
data = {
    'Compound': [
        'Methane', 'Water', 'Propane', 'Ethanol', 'Formic Acid',
        'Acetic Acid', 'Butane', 'Acetone', 'Benzene', 'Toluene', 'Octane'
    ],
    'Formula': [
        'CH4', 'H2O', 'C3H8', 'C2H5OH', 'HCOOH',
        'CH3COOH', 'C4H10', '(CH3)2CO', 'C6H6', 'C7H8', 'C8H18'
    ],
    'MW (g/mol)': [16, 18, 44, 46, 46, 60, 58, 58, 78, 92, 114],
    'BP (°C)': [-161, 100, -42, 78, 101, 118, -1, 56, 80, 111, 125]
}

df = pd.DataFrame(data)

# ==========================================
# 2. SPLIT OUT BUTANE FOR TESTING
# ==========================================
butane_row = df[df['Compound'] == 'Butane']
df_train = df[df['Compound'] != 'Butane'].reset_index(drop=True)

# Training Data (All compounds except Butane)
X_train = df_train[['MW (g/mol)']]
Y_train = df_train['BP (°C)']

# Test Data (Butane)
X_butane = butane_row[['MW (g/mol)']]
real_butane_bp = butane_row['BP (°C)'].values[0]

# ==========================================
# 3. LINEAR REGRESSION MODEL (Retrained)
# ==========================================
print("--- Training Linear Regression Model (Without Butane) ---")
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
Y_pred_lr_train = lr_model.predict(X_train)

# Metrics on Training Set
mae_lr = mean_absolute_error(Y_train, Y_pred_lr_train)
mse_lr = mean_squared_error(Y_train, Y_pred_lr_train)
r2_lr = r2_score(Y_train, Y_pred_lr_train)

print(f"Training MAE: {mae_lr:.2f}°C")
print(f"Training MSE: {mse_lr:.2f}")
print(f"Training R²:  {r2_lr:.2f}\n")

pred_butane_lr = lr_model.predict(X_butane)[0]

# ==========================================
# 4. NEURAL NETWORK MODEL (Retrained)
# ==========================================
print("--- Training Neural Network (MLPRegressor) Model (Without Butane) ---")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# MLP Configuration: 2 hidden layers with 10 neurons each, ReLU, 5000 epochs max
mlp_model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    max_iter=5000,
    random_state=42,
    solver='adam'
)

mlp_model.fit(X_train_scaled, Y_train)
Y_pred_mlp_train = mlp_model.predict(X_train_scaled)

print(f"Neural Network training epochs actually used: {mlp_model.n_iter_}")

# Metrics on Training Set
mae_mlp = mean_absolute_error(Y_train, Y_pred_mlp_train)
mse_mlp = mean_squared_error(Y_train, Y_pred_mlp_train)
r2_mlp = r2_score(Y_train, Y_pred_mlp_train)

print(f"Training MAE: {mae_mlp:.2f}°C")
print(f"Training MSE: {mse_mlp:.2f}")
print(f"Training R²:  {r2_mlp:.2f}\n")

X_butane_scaled = scaler.transform(X_butane)
pred_butane_mlp = mlp_model.predict(X_butane_scaled)[0]

# ==========================================
# 5. COMPARISON & METRICS SUMMARY OUTPUT
# ==========================================
print("==========================================")
print("       FINAL METRICS & COMPARISON         ")
print("==========================================")
print(f"Metric          | Linear Reg | Neural Net")
print(f"------------------------------------------")
print(f"Training MAE    | {mae_lr:.2f}°C    | {mae_mlp:.2f}°C")
print(f"Training MSE    | {mse_lr:.2f}   | {mse_mlp:.2f}")
print(f"Training R²     | {r2_lr:.2f}       | {r2_mlp:.2f}")
print("==========================================")
print(f"Butane Real BP:        {real_butane_bp:.2f}°C")
print(f"Linear Reg Prediction: {pred_butane_lr:.2f}°C")
print(f"Neural Net Prediction: {pred_butane_mlp:.2f}°C")
print("==========================================")

# ==========================================
# 6. SAVE PLOTS TO DIRECTORY
# ==========================================

# A. Combined Models Prediction Graph
plt.figure(figsize=(9, 6))
plt.scatter(X_train, Y_train, color='blue', label='Training Data', zorder=5)
plt.scatter(X_butane, real_butane_bp, color='orange', marker='*', s=200, label='Butane (Actual)', zorder=6)

X_dense = np.linspace(df['MW (g/mol)'].min(), df['MW (g/mol)'].max(), 200).reshape(-1, 1)
Y_dense_lr = lr_model.predict(X_dense)
Y_dense_mlp = mlp_model.predict(scaler.transform(X_dense))

plt.plot(X_dense, Y_dense_lr, color='red', linewidth=2, label='Linear Model Fit')
plt.plot(X_dense, Y_dense_mlp, color='green', linewidth=2, linestyle='-', label='Neural Network Fit (MLP)')

plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('Boiling Point (°C)')
plt.title('Comparison of Linear Regression vs Neural Network Models')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('models_comparison_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# B. ReLU Activation Function Visualization Plot
weights = mlp_model.coefs_[0]
biases = mlp_model.intercepts_[0]
X_dense_scaled = scaler.transform(X_dense)

plt.figure(figsize=(9, 6))
for i in range(weights.shape[1]):
    z = X_dense_scaled * weights[0, i] + biases[i]
    relu_out = np.maximum(0, z)
    plt.plot(X_dense, relu_out, label=f'Neuron {i + 1}')

plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('ReLU Output Value')
plt.title('ReLU Activation Outputs across First Hidden Layer Neurons')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('relu_activation_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# C. Real vs Predicted Comparison Plot for Butane
plt.figure(figsize=(7, 7))
plt.scatter([real_butane_bp], [pred_butane_lr], color='red', label=f'Linear Reg ({pred_butane_lr:.1f}°C)', s=150,
            zorder=5)
plt.scatter([real_butane_bp], [pred_butane_mlp], color='green', label=f'Neural Net ({pred_butane_mlp:.1f}°C)', s=150,
            zorder=5)

min_v = min(real_butane_bp, pred_butane_lr, pred_butane_mlp) - 30
max_v = max(real_butane_bp, pred_butane_lr, pred_butane_mlp) + 30
plt.plot([min_v, max_v], [min_v, max_v], color='black', linestyle='--', label='Ideal Prediction (y = x)')

plt.xlabel('Real Boiling Point (°C)')
plt.ylabel('Predicted Boiling Point (°C)')
plt.title('Butane: Real vs Predicted Boiling Point Comparison')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(min_v, max_v)
plt.ylim(min_v, max_v)
plt.savefig('prediction_comparison_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# D. Individual Fit and Residual Plots
plt.figure(figsize=(8, 5))
plt.scatter(X_train, Y_train, color='blue', label='Training Data (Excluding Butane)')
plt.scatter(X_butane, real_butane_bp, color='orange', marker='*', s=150, label='Butane (Actual)')
plt.plot(X_train, Y_pred_lr_train, color='red', linewidth=2, label='Linear Fit')
plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('Boiling Point (°C)')
plt.title('Linear Regression: Molecular Weight vs Boiling Point')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()

residuals_lr = Y_train - Y_pred_lr_train
plt.figure(figsize=(8, 5))
plt.scatter(Y_pred_lr_train, residuals_lr, color='purple', edgecolors='k')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Predicted Boiling Point (°C)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Linear Regression Residuals Plot')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(X_train, Y_train, color='blue', label='Training Data (Excluding Butane)')
plt.scatter(X_butane, real_butane_bp, color='orange', marker='*', s=150, label='Butane (Actual)')
plt.plot(X_dense, Y_dense_mlp, color='green', linewidth=2, label='MLP Fit (2x10, ReLU)')
plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('Boiling Point (°C)')
plt.title('Neural Network (MLPRegressor): Molecular Weight vs Boiling Point')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('mlp_regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()

residuals_mlp = Y_train - Y_pred_mlp_train
plt.figure(figsize=(8, 5))
plt.scatter(Y_pred_mlp_train, residuals_mlp, color='orange', edgecolors='k')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Predicted Boiling Point (°C)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('MLP Residuals Plot')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('mlp_residuals_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[Success] All models, predictions, and plots generated successfully!")


# ==========================================
# 7. AUTOMATED GIT COMMIT AND PUSH TO GITHUB
# ==========================================
def commit_and_push_plots():
    try:
        print("\n--- Pushing plots to GitHub ---")

        # Stage all PNG image files
        subprocess.run(["git", "add", "*.png"], check=True)

        # Check if there are changes to commit
        status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        if not status_result.stdout.strip():
            print("No new changes in plot files to commit.")
            return

        # Commit changes
        commit_message = "Auto-update model evaluation plots and neural network analysis"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push to remote repository (defaults to current branch / origin)
        subprocess.run(["git", "push"], check=True)
        print("[Success] Plots successfully committed and pushed to GitHub!")

    except subprocess.CalledProcessError as e:
        print(f"[Error] Git command failed: {e}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred during git operations: {e}")


# Run the git automation
commit_and_push_plots()