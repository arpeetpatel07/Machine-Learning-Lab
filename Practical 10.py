# -*- coding: utf-8 -*-


Aim: Implement KMeans Clustering Algorithm
"""

from sklearn.cluster import KMeans

# Define optimal_k as it was not defined previously
optimal_k = 3 # Placeholder value, consider determining this using methods like the Elbow method

# Instantiate KMeans with optimal_k and random_state
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')

# Fit the KMeans model to the scaled RFM data
kmeans.fit(rfm_scaled_array)

# Store the cluster labels
cluster_labels = kmeans.labels_

print("K-Means clustering completed and labels stored in 'cluster_labels'.")
print(f"First 10 cluster labels: {cluster_labels[:10]}")

from sklearn.preprocessing import StandardScaler

# Select the features for scaling (assuming log-transformed features are preferred for clustering)
rfm_features = ['Recency_log', 'Frequency_log', 'Monetary_log']

# Instantiate a StandardScaler
scaler = StandardScaler()

# Scale the selected RFM features
rfm_scaled_array = scaler.fit_transform(rfm_df[rfm_features])

print("RFM features scaled successfully and stored in 'rfm_scaled_array'.")
print(f"Shape of scaled array: {rfm_scaled_array.shape}")

rfm_df['Cluster'] = cluster_labels

print("Cluster labels successfully added to 'rfm_df'.")
print(rfm_df.head())

from sklearn.decomposition import PCA
import pandas as pd

# Instantiate PCA with 2 components
pca = PCA(n_components=2, random_state=42)

# Fit PCA to the scaled RFM data and transform it
rfm_pca = pca.fit_transform(rfm_scaled_array)

# Create a DataFrame for the PCA components
rfm_pca_df = pd.DataFrame(data=rfm_pca, columns=['PCA1', 'PCA2'])

print("PCA applied and components stored in 'rfm_pca_df'.")
print(rfm_pca_df.head())

rfm_pca_df['Cluster'] = cluster_labels

print("Cluster labels successfully added to 'rfm_pca_df'.")
print(rfm_pca_df.head())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=rfm_pca_df, palette='viridis', s=100, alpha=0.8)
plt.title('K-Means Clusters Visualized with PCA', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
