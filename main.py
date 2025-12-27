"""
CO2 Emissions Prediction
Linear Regression using:
1. Gradient Descent (from scratch)
2. Scikit-learn LinearRegression

Dataset: CO2 Emissions_Canada.csv
Features: Engine Size, Cylinders, Fuel Consumption (City, Hwy, Comb)
Target: CO2 Emissions (g/km)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
dataset_path = './CO2 Emissions_Canada.csv'
dataset = pd.read_csv(dataset_path)

features = [
    "Engine Size(L)",
    "Cylinders",
    "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)",
    "Fuel Consumption Comb (L/100 km)"
]
target = "CO2 Emissions(g/km)"

X = dataset[features].values  # shape (n_samples, 5)
y = dataset[target].values    # shape (n_samples,)

# -------------------------------
# 2️⃣ Feature scaling (Z-score normalization)
# Important for gradient descent
# -------------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# -------------------------------
# 3️⃣ Split data: 80% train, 20% validation
# Shuffle data
# -------------------------------
np.random.seed(42)
indices = np.arange(X_scaled.shape[0])
np.random.shuffle(indices)

X_scaled = X_scaled[indices]
y = y[indices]

split_idx = int(0.8 * X_scaled.shape[0])
X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# -------------------------------
# 4️⃣ Gradient Descent Implementation
# -------------------------------
def f_wb(X, w, b):
    """Predict y using linear model: y = Xw + b"""
    return X @ w + b

def compute_cost(X, y, w, b):
    """Compute Mean Squared Error cost"""
    m = X.shape[0]
    y_pred = f_wb(X, w, b)
    return (1/(2*m)) * np.sum((y_pred - y)**2)

def compute_gradients(X, y, w, b):
    """Compute gradients of the cost function w.r.t weights and bias"""
    m = X.shape[0]
    y_pred = f_wb(X, w, b)
    error = y_pred - y
    dw = (1/m) * (X.T @ error)
    db = (1/m) * np.sum(error)
    return dw, db

# Hyperparameters
alpha = 0.01      # Learning rate
iterations = 2000 # Number of iterations

# Initialize weights and bias
w = np.zeros(X_train.shape[1])
b = 0
cost_history = []

# Gradient Descent Loop
for i in range(iterations):
    dw, db = compute_gradients(X_train, y_train, w, b)
    w -= alpha * dw
    b -= alpha * db
    if i % 100 == 0 or i == iterations-1:
        cost_history.append(compute_cost(X_train, y_train, w, b))

print("Gradient Descent Trained weights:", w)
print("Gradient Descent Trained bias:", b)

# Predictions on validation set
y_val_pred_gd = f_wb(X_val, w, b)

# Evaluation
mse_gd = np.mean((y_val_pred_gd - y_val)**2)
ss_res = np.sum((y_val - y_val_pred_gd)**2)
ss_tot = np.sum((y_val - np.mean(y_val))**2)
r2_gd = 1 - (ss_res / ss_tot)

print("Gradient Descent MSE on validation set:", mse_gd)
print("Gradient Descent R² score on validation set:", r2_gd)

# -------------------------------
# 5️⃣ Scikit-learn Linear Regression
# -------------------------------
scaler = StandardScaler()
X_train_scaled_skl = scaler.fit_transform(X_train)
X_val_scaled_skl = scaler.transform(X_val)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled_skl, y_train)

weights_skl = lr_model.coef_
bias_skl = lr_model.intercept_

y_val_pred_skl = lr_model.predict(X_val_scaled_skl)

mse_skl = mean_squared_error(y_val, y_val_pred_skl)
r2_skl_val = r2_score(y_val, y_val_pred_skl)

print("\n--- scikit-learn Linear Regression ---")
print("Trained weights:", weights_skl)
print("Trained bias:", bias_skl)
print("MSE on validation set (sklearn):", mse_skl)
print("R² score on validation set (sklearn):", r2_skl_val)

# -------------------------------
# 6️⃣ Compare Predictions Visually
# -------------------------------

plt.figure(figsize=(8,6))

# Plot actual vs predicted values
plt.scatter(y_val, y_val_pred_gd, alpha=0.5, color='blue', label='Gradient Descent')
plt.scatter(y_val, y_val_pred_skl, alpha=0.5, color='green', label='scikit-learn')

# Plot ideal line
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Ideal')

# Labels and title
plt.xlabel("Actual CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.title("Actual vs Predicted CO2 Emissions")
plt.legend()
plt.grid(True)
plt.show()

