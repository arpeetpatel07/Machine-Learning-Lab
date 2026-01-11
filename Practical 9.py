# -*- coding: utf-8 -*-

Aim : Apply Naive Bayes Classifier on the Dataset and analyse the prediction accuracy.
"""

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from google.colab import files

# Step 2: Upload and Load Dataset
uploaded = files.upload()  # Upload 'Social_Network_Ads.csv'
df = pd.read_csv('Social_Network_Ads.csv')
df.head()

# Step 3: Select Features and Target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Step 4: Split Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Step 5: Handle Missing Values (if any)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train[['Age', 'EstimatedSalary']])
X_train[['Age', 'EstimatedSalary']] = imputer.transform(X_train[['Age', 'EstimatedSalary']])
X_test[['Age', 'EstimatedSalary']] = imputer.transform(X_test[['Age', 'EstimatedSalary']])

# Step 6: Feature Scaling (Naive Bayes can work without scaling but scaling doesn't harm)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Step 8: Evaluate Model Accuracy
train_acc = nb.score(X_train, y_train)
test_acc = nb.score(X_test, y_test)
print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# Step 9: Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report

y_pred = nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)
)
plt.contourf(X1, X2, nb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('lightcoral', 'lightgreen')))
plt.scatter(X_set[:, 0], X_set[:, 1], c = y_set, cmap = ListedColormap(('red', 'green')))
plt.title('Naive Bayes Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()
