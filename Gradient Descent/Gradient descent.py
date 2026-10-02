import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Set random seed for reproducibility
np.random.seed(42)

# Generate x values in range 0 - 10
X = np.linspace(0, 10, 100).reshape(-1, 1)

# True line: y = 2x + 5
true_y = 2 * X.ravel() + 5

# Add random Gaussian noise (mean=0, std=1.5)
noise = np.random.normal(0, 1.5, size=X.ravel().shape)
y = true_y + noise

# 1. Fit Scikit-Learn LinearRegression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

fitted_w = model.coef_[0]
fitted_b = model.intercept_

print(f'Fitted Slope (w): {fitted_w:.4f}')
print(f'Fitted Intercept (b): {fitted_b:.4f}')

# 2. Compute Loss Landscape Grid over parameters (w, b)
w_vals = np.linspace(1.0, 3.0, 100)
b_vals = np.linspace(3.0, 7.0, 100)
W, B = np.meshgrid(w_vals, b_vals)

# MSE Loss: L(w, b) = (1/N) * sum((y - (w*X + b))^2)
X_flat = X.ravel()
y_hat = B[:, :, np.newaxis] + W[:, :, np.newaxis] * X_flat
loss = np.mean((y - y_hat) ** 2, axis=2)

# 3. Compute Analytical Gradients for Vector Field
w_coarse = np.linspace(1.1, 2.9, 15)
b_coarse = np.linspace(3.2, 6.8, 15)
W_c, B_c = np.meshgrid(w_coarse, b_coarse)

y_hat_c = B_c[:, :, np.newaxis] + W_c[:, :, np.newaxis] * X_flat
dL_dw = -2 * np.mean(X_flat * (y - y_hat_c), axis=2)
dL_db = -2 * np.mean(y - y_hat_c, axis=2)

# 4. Plotting: Subplots for Fit and Loss Landscape
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Regression Fit
ax1.scatter(
    X, y, color='crimson', alpha=0.7, edgecolors='k', label='Noisy Data Points'
)
ax1.plot(
    X,
    true_y,
    color='navy',
    linestyle='--',
    linewidth=2,
    label='True Line ($y = 2x + 5$)',
)
ax1.plot(
    X,
    y_pred,
    color='forestgreen',
    linestyle='-',
    linewidth=2,
    label=f'Fitted Line ($y = {fitted_w:.2f}x + {fitted_b:.2f}$)',
)
ax1.set_title('1. Data & Linear Regression Fit', fontsize=13, fontweight='bold')
ax1.set_xlabel('X', fontsize=11)
ax1.set_ylabel('Y', fontsize=11)
ax1.set_xlim(0, 10)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=10)

# Subplot 2: Loss Landscape Contour & Gradient Field
# Reversing viridis (viridis_r) maps low values to yellow and high values to purple
contour = ax2.contourf(W, B, loss, levels=30, cmap='viridis_r', alpha=0.85)
fig.colorbar(contour, ax=ax2, label='Mean Squared Error (MSE)')

# Overlay gradient vectors using quiver (dark colored vector field for visibility against yellow center)
ax2.quiver(
    W_c,
    B_c,
    dL_dw,
    dL_db,
    color='black',
    alpha=0.7,
    scale=50,
    width=0.004,
    label='Gradient Field ($\\nabla L$)',
)

# Highlight fitted minimum and true parameters
ax2.plot(
    fitted_w,
    fitted_b,
    'r*',
    markersize=14,
    markeredgecolor='black',
    label=f'Fitted Min ($w={fitted_w:.2f}, b={fitted_b:.2f}$)',
)
ax2.plot(
    2.0,
    5.0,
    'cyan',
    marker='o',
    linestyle='None',
    markersize=8,
    markeredgecolor='black',
    label='True Parameters ($w=2, b=5$)',
)

ax2.set_title(
    '2. Loss Landscape & Gradient Field ($MSE(w, b)$)',
    fontsize=13,
    fontweight='bold',
)
ax2.set_xlabel('Weight / Slope ($w$)', fontsize=11)
ax2.set_ylabel('Bias / Intercept ($b$)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.4, color='gray')
ax2.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('loss_landscape_reversed.png', dpi=300)
plt.show()