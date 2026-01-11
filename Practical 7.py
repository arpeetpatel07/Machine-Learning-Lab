
Aim : To perform Dimensionality Reduction using PCA(Principal Component Analysis)
"""

import numpy as np
from sklearn.decomposition import PCA

# Sample data with 6 samples and 2 features
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# Initialize PCA, reducing to 2 components (in this case, no reduction)
pca = PCA(n_components=2)

# Fit PCA on the data and transform it
X_pca = pca.fit_transform(X)

# Print the transformed data
print(X_pca)

# You can also specify a different number of components, e.g., reducing to
pca_one_component = PCA(n_components=1)
X_pca_one = pca_one_component.fit_transform(X)

# Print the transformed data with one component
print("\nTransformed data with 1 component:")
print(X_pca_one)

import matplotlib.pyplot as plt

# Visualize the 2-component PCA transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2-Component PCA of Sample Data')
plt.grid(True)
plt.axhline(0, color='grey', linewidth=0.8)
plt.axvline(0, color='grey', linewidth=0.8)
plt.show()
print('\nExplained Variance Ratio (2 components):')
print(pca.explained_variance_ratio_)
print('Cumulative Explained Variance (2 components):', pca.explained_variance_ratio_.sum())

print('\nExplained Variance Ratio (1 component):')
print(pca_one_component.explained_variance_ratio_)
print('Cumulative Explained Variance (1 component):', pca_one_component.explained_variance_ratio_.sum())
