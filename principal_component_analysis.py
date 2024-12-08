
'''
Apply PCA for Dimensionality Reduction
Now that we've reduced the dataset to important features, we can apply PCA to check
if further dimensionality reduction is possible without significant loss of variance.
PCA will help us identify the most influential principal components for clustering, regression, or classification tasks.
'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the reduced dataset
file_path = 'reduced_dataset.csv'
data_reduced = pd.read_csv(file_path)

# Separate features and target variable
target = 'price_usd'
features = [col for col in data_reduced.columns if col != target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_reduced[features])

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# Print cumulative variance explained by components
cumulative_variance = explained_variance.cumsum()
print("Cumulative Variance Explained by Components:")
for i, var in enumerate(cumulative_variance):
    print(f"Component {i+1}: {var:.4f}")

'''
Analysis of PCA Output
Variance Retention:

The first 10 components explain approximately 65.87% of the variance.
The first 15 components explain about 84.11% of the variance.
The first 20 components explain around 96.63% of the variance.
By retaining 15 components, you capture most of the variance while achieving significant dimensionality reduction.
Elbow Method:

From the cumulative variance plot, the elbow point (where the curve starts to flatten significantly) is around 10 components, 
suggesting diminishing returns after this.
Next Steps: Retain the Top Components
Based on this, let’s retain the top 20 components to balance dimensionality reduction with variance retention. 
The reduced dataset can then be used for regression, clustering, or classification tasks.
'''
# Retain the top 15 components
# Exclude target variables ('price_usd' and 'host_is_superhost') for PCA
features_for_pca = [
    col for col in data_reduced.columns if col not in ['price_usd', 'host_is_superhost']
]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_reduced[features_for_pca])

# Perform PCA
n_components = 20  # Adjust based on cumulative variance from earlier analysis
pca = PCA(n_components=n_components)
X_pca_transformed = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA-transformed data
pca_columns = [f'PC{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(X_pca_transformed, columns=pca_columns)

# Add the target variables back for specific tasks
pca_df['price_usd'] = data_reduced['price_usd']  # For regression
pca_df['host_is_superhost'] = data_reduced['host_is_superhost']  # For classification

# Save the PCA-reduced dataset
pca_df.to_csv('pca_reduced_dataset_no_targets.csv', index=False)
print(f"PCA-reduced dataset with {n_components} components saved as 'pca_reduced_dataset_no_targets.csv'")

