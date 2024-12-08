from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'processed_airbnb_data_with_categories.csv'
df = pd.read_csv(file_path, encoding='latin1')
data = df

# Select numerical features excluding irrelevant columns
numerical_columns = [
    col for col in data.select_dtypes(include=['float64', 'int64']).columns
    if col not in ['listing_id', 'host_id']
]

# Compute correlation matrix
corr_matrix = data[numerical_columns].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Numerical Features (Excluding IDs)")
plt.show()

# Display pairs with high correlation (absolute correlation > 0.8)
high_corr_pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
high_corr_pairs = high_corr_pairs[high_corr_pairs > 0.8]
print("Highly Correlated Feature Pairs (|correlation| > 0.8):")
print(high_corr_pairs)

'''
Low Multicollinearity Overall:

Aside from host_is_superhost having perfect self-correlation (expected), there are no strong correlations (|correlation| > 0.8) between the other numerical features.
The review_scores_* features (e.g., rating, accuracy, cleanliness) exhibit moderate correlations (~0.6-0.7), but these are not severe enough to require immediate removal.
No Redundant Features Detected:

Since no pairs have a correlation above 0.8 (other than diagonal elements), we don't need to drop features solely based on multicollinearity at this stage.
'''


'''
Random Forest Feature Importance
Since the correlation analysis does not suggest strong multicollinearity,
 the next step is to evaluate feature importance using Random Forest. 
 This will help identify which features contribute the most to predicting a specific target variable (e.g., price_usd).
'''
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Define features and target
target = 'price_usd'  # Target variable for regression
features = [
    col for col in numerical_columns
    if col != target  # Exclude the target from the feature list
]

# Prepare feature matrix (X) and target vector (y)
X_rf = data[features]
y_rf = data[target]

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)

# Calculate feature importance
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display feature importances
print("Feature Importances from Random Forest:")
print(feature_importances)

# Plot feature importances
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance at the top
plt.show()

'''
Observations from Random Forest Feature Importance:
Top Features:

The most important features for predicting price_usd include:
maximum_nights
host_total_listings_count
minimum_nights
bedrooms
exchange_rate
accommodates
Moderately Important Features:

Features like Laundry, review_scores_rating, Accessibility, and Safety Features also show reasonable importance but contribute less compared to the top six features.
Low-Importance Features:

Features such as Child-Friendly, Kitchen Essentials, review_scores_checkin, and host_has_profile_pic have minimal contributions and can be candidates for exclusion, especially if dimensionality reduction is a priority.
Next Steps: Selecting Features
Based on the feature importance results, we can make the following decisions:

Retain Key Features:

Retain the top features (maximum_nights, host_total_listings_count, minimum_nights, bedrooms, exchange_rate, accommodates) as they contribute the most to predicting price_usd.
Drop Low-Importance Features:

Drop features with extremely low importance (e.g., host_has_profile_pic, review_scores_checkin, Kitchen Essentials).
Optional: Run PCA on Remaining Features:

After reducing features based on importance, PCA can help further reduce dimensionality while retaining most of the variance.
'''

# Retain features with importance above a threshold (e.g., 0.01)
important_features = feature_importances[feature_importances['Importance'] > 0.01]['Feature'].tolist()

# Subset the data to include only important features
data_reduced = data[important_features + ['price_usd']]  # Add the target variable back

print(f"Reduced dataset shape: {data_reduced.shape}")
print("Retained features:")
print(important_features)

# Save the reduced dataset for further use
data_reduced.to_csv('reduced_dataset.csv', index=False)
print("Reduced dataset saved as 'reduced_dataset.csv'")

