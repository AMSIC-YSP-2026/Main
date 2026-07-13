# import csv
# import numpy as np

# CSV_PATH = ".\\MachineLearning\\data.csv"      # <-- put your CSV file path here
# X_COLUMN = None            # e.g. "time" or "0" (None = first column)
# Y_COLUMN = None            # e.g. "value" or "1" (None = second column)
# mse = 0.0                  # Initialize mean squared error
# mse_gradient = np.array([0.0, 0.0])  # Initialize gradient for mean squared error
# theta = np.array([0.0, 0.0])  # Initial parameters (intercept and slope)

# with open(CSV_PATH, 'r') as file:
#     reader = csv.reader(file)
#     row_count = sum(1 for row in reader)
# print("Total points (including header):", row_count)

# # print points
# for i in range(0, row_count+1):
#     #with open(CSV_PATH, 'r') as file:
#         #reader = csv.reader(file)
#         # row_count = sum(1 for row in reader)
#     for j, row in enumerate(reader):
#         if j == i - 1:  
#             x_value = float(row[0])  
#             y_value = float(row[1]) 
#             mse = mse + np.mean((y_value - (theta[0] + theta[1] * x_value)) ** 2)
#             mse_gradient = mse_gradient + (-2 / row_count) * (y_value - (theta[0] + theta[1] * x_value)) * np.array([1, x_value])
#             theta = theta - 0.01 * mse_gradient  # Update parameters using gradient descent
#             break


import csv
import numpy as np

CSV_PATH = ".\\MachineLearning\\data.csv"   # <-- put your CSV file path here
X_COLUMN = None       # e.g. "time" or "0" (None = first column)
Y_COLUMN = None       # e.g. "value" or "1" (None = second column)
LEARNING_RATE = 0.01
EPOCHS = 1000

# --- Load data ---
x_vals = []
y_vals = []

with open(CSV_PATH, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # skip header row

    # Resolve column indices
    x_idx = 0 if X_COLUMN is None else (
        int(X_COLUMN) if X_COLUMN.isdigit() else header.index(X_COLUMN)
    )
    y_idx = 1 if Y_COLUMN is None else (
        int(Y_COLUMN) if Y_COLUMN.isdigit() else header.index(Y_COLUMN)
    )

    for row in reader:
        if not row:
            continue
        x_vals.append(float(row[x_idx]))
        y_vals.append(float(row[y_idx]))

x = np.array(x_vals)
y = np.array(y_vals)
n = len(x)
print("Total data points:", n)

# --- Gradient descent ---
theta = np.array([0.0, 0.0])  # [intercept, slope]

for epoch in range(EPOCHS):
    predictions = theta[0] + theta[1] * x
    errors = predictions - y

    mse = np.mean(errors ** 2)
    grad_intercept = (2 / n) * np.sum(errors)
    grad_slope = (2 / n) * np.sum(errors * x)
    gradient = np.array([grad_intercept, grad_slope])

    theta = theta - LEARNING_RATE * gradient

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch}: MSE = {mse:.6f}, theta = {theta}")

np.save(".\\MachineLearning\\theta.npy", theta)
print(f"Saved theta to theta.npy: intercept={theta[0]:.6f}, slope={theta[1]:.6f}")

print("\nFinal parameters:")
print(f"Intercept: {theta[0]:.6f}")
print(f"Slope:     {theta[1]:.6f}")

