from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
import pandas as pd
data = pd.read_csv("../Regression Analysis/cleaned_dataset.csv")

# Set up features (X) and target (y)
X = data.drop(columns=['host_is_superhost', 'price_usd'])  # Remove target and irrelevant columns
y = data['host_is_superhost']  # Classification target

# Normalize the feature set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Initialize and train the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate performance
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and display ROC-AUC
y_prob = dt_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
'''
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
'''
print(f"ROC-AUC Score: {roc_auc:.2f}")


'''
The model performs well for Class 0 (non-superhost) but struggles with Class 1 (superhost).
Imbalance in the dataset might be impacting performance (many more Class 0 samples compared to Class 1).
Moderate AUC suggests room for improvement.
'''

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
dt = DecisionTreeClassifier(random_state=42)

# Define parameter grid for grid search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.01, 0.1]
}

# Perform grid search
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Best parameters and performance
print("Best parameters found:", grid_search.best_params_)
best_dt = grid_search.best_estimator_
y_pred_balanced = best_dt.predict(X_test)

# Evaluate the new model
conf_matrix = confusion_matrix(y_test, y_pred_balanced)
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_balanced))

# ROC-AUC
y_prob_balanced = best_dt.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob_balanced)
fpr, tpr, thresholds = roc_curve(y_test, y_prob_balanced)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Optimized Decision Tree (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print(f"ROC-AUC Score: {roc_auc:.2f}")




param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 15, None],  # Fewer depth values
    'min_samples_split': [2, 5],  # Focus on fewer split thresholds
    'max_features': [None, 'sqrt'],  # Limit feature selection
    'ccp_alpha': [0.0, 0.01]  # Fewer pruning options
}

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Randomized Grid Search
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': np.linspace(0.0, 0.1, 5)  # Continuous range for pruning
}

random_search = RandomizedSearchCV(
    estimator=dt,
    param_distributions=param_dist,
    n_iter=50,  # Test 50 random combinations
    scoring='roc_auc',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_balanced, y_train_balanced)

# Best parameters and performance
print("Best parameters found:", random_search.best_params_)
best_dt = random_search.best_estimator_
y_pred_balanced = best_dt.predict(X_test)

# Evaluate the new model
conf_matrix = confusion_matrix(y_test, y_pred_balanced)
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_balanced))


# ROC-AUC for Optimized Model
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_prob_balanced = best_dt.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob_balanced)
fpr, tpr, thresholds = roc_curve(y_test, y_prob_balanced)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Optimized Decision Tree (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print(f"ROC-AUC Score: {roc_auc:.2f}")
