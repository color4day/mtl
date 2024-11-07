import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
print(os.getcwd())
mypath = os.getcwd()+"/folds"

# Set random seed for reproducibility
np.random.seed(42)

# Number of points and tasks
n_points = 200
n_classes = 3

def nonlinear_function(X0, labels1, labels2):
    dist = (max(X0) - min(X0)) * 4
    k = 1
    phi0 = np.pi/6
    noise1 = 0.5*np.random.randn(labels1.shape[0])
    noise2 = 0.0*np.random.randn(labels1.shape[0])
    noise3 = 0.2*np.random.randn(labels1.shape[0])
    A = (labels1*3 + 1 + noise1)
    x = A * np.cos( phi0 + (X0/dist) * 2*np.pi + 2*np.pi * (labels2+noise2) /n_classes + noise3)
    y = A * np.sin( phi0 + (X0/dist) * 2*np.pi + 2*np.pi *  (labels2+noise2) /n_classes + noise3)
    X = np.vstack((x, y)).T
    return X

# Generate input data (e.g., time points or spatial coordinates)
X0 = np.linspace(0, 10, n_points)
labels1 = np.random.randint(0, n_classes, n_points)
labels2 = np.random.randint(0, n_classes, n_points)
X = nonlinear_function(X0, labels1, labels2)

# Combine data and labels into a DataFrame for better readability
data = pd.DataFrame({
    'X1': X[:, 0],
    'X2': X[:, 1],
    'labels1': labels1,
    'labels2': labels2
})

# Save data as a CSV file
data.to_csv(mypath + '/classification_data.csv', index=False)
# Load data from CSV file
data = pd.read_csv(mypath + '/classification_data.csv')
# Print the loaded data
print(data)

fig, ax = plt.subplots(1,2, figsize=(10, 5))
ax[0].scatter(data['X1'], data['X2'], c=data['labels1'])
ax[1].scatter(data['X1'], data['X2'], c=data['labels2'])
plt.show()

