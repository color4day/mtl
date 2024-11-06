import numpy as np
import pandas as pd
import os
print(os.getcwd())
mypath = os.getcwd()+"/folds"

# Set random seed for reproducibility
np.random.seed(42)

# Number of points and tasks
n_points = 200

# Generate input data (e.g., time points or spatial coordinates)
X = np.linspace(0, 10, n_points).reshape(-1, 1)

# Simulate two tasks with correlated outputs
# Task 1: Y1 = sin(X) + some noise
Y1 =(X*(10-X)/10 * np.sin(X)).ravel() + np.random.normal(0, 0.1, n_points)

# Task 2: Y2 = cos(X) + some noise, with correlation to Y1
Y2 = np.cos(X*(10-X)/10).ravel() + np.random.normal(0, 0.1, n_points)

# Combine data into a DataFrame for better readability
data = pd.DataFrame({
    'X': X.ravel(),
    'Y1': Y1,
    'Y2': Y2
})

# Save data as a CSV file
data.to_csv(mypath + '/synthetic_data2.csv', index=False)
# Load data from CSV file
data = np.loadtxt(mypath + '/synthetic_data2.csv', delimiter=',', skiprows=1)
# Print the loaded data
print(data)


