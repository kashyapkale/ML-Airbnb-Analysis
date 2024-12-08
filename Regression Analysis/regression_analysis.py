from sklearn.model_selection import train_test_split
import pandas as pd




'''
Step 1: Split Data into Training and Testing Sets
We'll start by preparing the data for regression analysis using price_usd as the dependent variable.
'''
# Load the PCA-reduced dataset
file_path = '../pca_reduced_dataset_no_targets.csv'
data = pd.read_csv(file_path)


# Calculate the IQR for 'price_usd'
Q1 = data['price_usd'].quantile(0.25)
Q3 = data['price_usd'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to exclude outliers
data_cleaned = data[(data['price_usd'] >= lower_bound) & (data['price_usd'] <= upper_bound)]

print(f"Original dataset size: {data.shape[0]}")
print(f"Cleaned dataset size: {data_cleaned.shape[0]}")

# Save the cleaned dataset
data_cleaned.to_csv('cleaned_dataset.csv', index=False)
print("Cleaned dataset saved as 'cleaned_dataset.csv'")

data = pd.read_csv('cleaned_dataset.csv')

# Define independent variables (exclude target)
X = data[[col for col in data.columns if col.startswith('PC')]]  # Use PCA-transformed components
y = data['price_usd']  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

'''
Step 2: Build the Initial Multiple Linear Regression Model
We'll use statsmodels to fit the regression model, which provides tools for evaluating metrics like R-squared, 
adjusted R-squared, and confidence intervals.
'''
import statsmodels.api as sm

# Add a constant to the features (for the intercept)
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit the regression model
model = sm.OLS(y_train, X_train_const).fit()

# Print model summary
print(model.summary())

# Predict on test data
y_pred = model.predict(X_test_const)

# Save predictions for plotting later
predictions = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).reset_index(drop=True)

'''
Step 3: Plot Train, Test, and Predicted Values
This step visualizes how the model performs by plotting the actual vs. predicted values for the test set.
'''
import matplotlib.pyplot as plt

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(predictions['Actual'].values, label='Actual', alpha=0.7)
plt.plot(predictions['Predicted'].values, label='Predicted', alpha=0.7)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price (USD)')
plt.ylim([-100, 600])
plt.legend()
plt.show()

'''
Step 4: Evaluate Regression Metrics
We'll calculate and present the following metrics:

1. R-squared and Adjusted R-squared
2. Mean Squared Error (MSE)
3. Akaike Information Criterion (AIC)
4. Bayesian Information Criterion (BIC)
'''
from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# Extract AIC and BIC from the model
aic = model.aic
bic = model.bic

# Prepare a metrics summary
metrics_summary = pd.DataFrame({
    'Metric': ['R-squared', 'Adjusted R-squared', 'AIC', 'BIC', 'MSE'],
    'Value': [model.rsquared, model.rsquared_adj, aic, bic, mse]
})

print("Regression Metrics Summary:")
print(metrics_summary)


# Extract t-tests and confidence intervals from the model
t_tests = pd.DataFrame({
    'Coefficient': model.params,
    'Standard Error': model.bse,
    't-value': model.tvalues,
    'p-value': model.pvalues
})

# Confidence intervals
confidence_intervals = model.conf_int()
confidence_intervals.columns = ['Lower 95%', 'Upper 95%']
t_tests = pd.concat([t_tests, confidence_intervals], axis=1)

print("T-Test and Confidence Interval Analysis:")
print(t_tests)


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Use Recursive Feature Elimination (RFE) for stepwise regression
estimator = LinearRegression()
rfe = RFE(estimator, n_features_to_select=10)  # Retain top 10 features
rfe = rfe.fit(X_train, y_train)

# Display selected features
selected_features = X.columns[rfe.support_]
print("Selected Features from Stepwise Regression:")
print(selected_features)

# Refit the model with selected features
X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]

X_train_reduced_const = sm.add_constant(X_train_reduced)
X_test_reduced_const = sm.add_constant(X_test_reduced)

stepwise_model = sm.OLS(y_train, X_train_reduced_const).fit()
print(stepwise_model.summary())
