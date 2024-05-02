# Code from another k-nn assignment

# Common imports
import numpy as np
import matplotlib.pyplot as plt

# Figures plotted inside the notebook
%matplotlib inline
# High quality figures
%config InlineBackend.figure_format = 'retina'
# For fancy table Display
%load_ext google.colab.data_table

import warnings
warnings.filterwarnings("ignore")

# Count the number of samples belonging to malignant tumors (class 0)
num_malignant_samples = np.count_nonzero(y == 0)

# Count the number of samples belonging to benign tumors (class 1)
num_benign_samples = np.count_nonzero(y == 1)

# Print the results
print("Number of samples belonging to malignant tumors:", num_malignant_samples)
print("Number of samples belonging to benign tumors:", num_benign_samples)

import matplotlib.pyplot as plt

# Count the number of samples belonging to each class
class_counts = [num_malignant_samples, num_benign_samples]

# Define class labels
class_labels = ['Malignant (0)', 'Benign (1)']

# Plot the bar chart
plt.bar(class_labels, class_counts, color=['red', 'green'])
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Breast Cancer Wisconsin Dataset')
plt.show()

from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
random_seed = 42

# Split the data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

# Print the shape of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.preprocessing import StandardScaler

# Create an instance of StandardScaler
scaler = StandardScaler()

# Fit the scaler with the training data (X_train)
scaler.fit(X_train)

# Create standardized matrices
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

#Let's check out that everything is ok!
print('TRAINING SET')
print('Mean of each feature: ', np.round(np.mean(X_train_s,0),2))
print('Std of each feature: ', np.round(np.std(X_train_s,0),2))

print('\nTEST SET')
print('Mean of each feature: ', np.round(np.mean(X_test_s,0),2))
print('Std of each feature: ', np.round(np.std(X_test_s,0),2))

#Fine-tuning hyperparameters
from sklearn.neighbors import KNeighborsClassifier

# Define the hyperparameters to be tuned
param_grid_knn = {'n_neighbors': list(range(1, 11)), 'weights': ['uniform', 'distance']}

# Create an instance of KNeighborsClassifier
knn_model = KNeighborsClassifier()

# Create an instance of GridSearchCV
grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5)

# Fit the GridSearchCV to the standardized training data
grid_search_knn.fit(X_train_s, y_train)

# Get the best hyperparameters
best_hyperparameters_knn = grid_search_knn.best_params_

# Print the best hyperparameters
print("Best Hyperparameters for k-NN Classifier:", best_hyperparameters_knn)

#Check the best model with the test set

# Get the best estimator
best_knn_model = grid_search_knn.best_estimator_

# Calculate accuracy rate on the test set
accuracy_test_knn = best_knn_model.score(X_test_s, y_test) * 100

# Print the result
print("Accuracy rate on the test set for k-NN Classifier:", round(accuracy_test_knn, 2), "%")