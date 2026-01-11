# Machine Learning Classification Assignment - Pima Indians Diabetes Dataset

# Goal: Predict whether a patient has diabetes (1) or not (0)

# The Libraries that we are using and are needed are as follows:
# pip install pandas numpy scikit-learn matplotlib seaborn

# Here we are importing all the stated libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# here we are loading our dataset

import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/MyDrive/ai ml dataset/diabetes.csv')

# displaying the content from our dataset

print('First 5 rows of the dataset:')
print(data.head())

print('\nBasic Information:')
print(data.info())

print('\nCheck for missing values:')
print(data.isnull().sum())

print('\nStatistical Summary:')
print(data.describe())

sns.scatterplot(data=data, x='Age', y='Glucose', hue='Outcome', palette='coolwarm')

# how to handel missing or zero values or outliers
# as some columns like Glucose, BloodPressure, SkinThickness, Insulin, and BMI can have zeros which are not realistic values. We will replace them with NaN.

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)

# Now, replace missing (NaN) values with the median of each column.
imputer = SimpleImputer(strategy='median')
data[cols_with_zero] = imputer.fit_transform(data[cols_with_zero])

print('\nMissing values after imputation:')
print(data.isnull().sum())

# here we are splitting the data into features and target

X = data.drop('Outcome', axis=1)  # Independent variables
y = data['Outcome']               # Target variable

# Split data into 80% training and 20% testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('\nTraining and testing data shapes:')
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)

# we are now doing Feature scaling as scaling helps models like Logistic Regression and SVM perform better.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# we are building and training classification models; We will build 4 models: Logistic Regression, Decision Tree, Random Forest, and SVM.

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Support Vector Machine (SVM)
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# now we will evaluate models using parameters like accuracy, precision, recall, F1-score

def evaluate_model(y_test, y_pred, model_name):
    print(f'\nPerformance of {model_name}:')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred))
    print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('\nClassification Report:\n', classification_report(y_test, y_pred))

# here call the function for each model
evaluate_model(y_test, y_pred_log, 'Logistic Regression')
evaluate_model(y_test, y_pred_tree, 'Decision Tree')
evaluate_model(y_test, y_pred_rf, 'Random Forest')
evaluate_model(y_test, y_pred_svm, 'SVM')

# here we are doing the comparison between ROC Curve and AUC

models = {
    'Logistic Regression': log_model,
    'Decision Tree': tree_model,
    'Random Forest': rf_model,
    'SVM': svm_model
}

plt.figure(figsize=(8, 6))
for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest']:
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()


# here is the complete Summary and Conclusion

print('\nSummary:')
print('- Logistic Regression and SVM usually perform well on this dataset after scaling.')
print('- Decision Tree is easy to interpret but can overfit.')
print('- Random Forest generally gives a balanced and accurate result.')
print('- SVM may achieve the highest accuracy if features are scaled properly.')

print('\nTips for improvement:')
print('- Try hyperparameter tuning (GridSearchCV) for better accuracy.')
print('- Experiment with other models like k-NN, Naive Bayes, or XGBoost.')
print('- Use cross-validation to confirm model stability.')

