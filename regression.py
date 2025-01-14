'''
Regression Analysis
'''
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns

def run_regression():
    # Load the PCA-reduced dataset
    file_path = 'pca_reduced_dataset_no_targets.csv'  # Update path if needed
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

    # Reload cleaned dataset
    data = pd.read_csv('cleaned_dataset.csv')

    # Define independent variables (exclude target)
    X = data[[col for col in data.columns if col.startswith('PC')]]  # Use PCA-transformed components
    y = data['price_usd']  # Target variable

    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

    ### Step 1: Build the Initial Multiple Linear Regression Model ###
    # Add a constant to the features (for the intercept)
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Fit the regression model
    model = sm.OLS(y_train, X_train_const).fit()

    # Print model summary
    print("\n--- Initial Multiple Linear Regression Summary ---")
    print(model.summary())

    # Predict on test data
    y_pred = model.predict(X_test_const)

    ### Step 2: Plot Train, Test, and Predicted Values ###
    # Visualize Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual', alpha=0.7, color='blue')
    plt.plot(y_pred.values, label='Predicted', alpha=0.7, color='orange')
    plt.title('Actual vs. Predicted Prices (Regression)')
    plt.xlabel('Index')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolor='k')
    plt.axhline(0, color='red', linestyle='--', label='Zero Residuals')
    plt.title('Residuals vs. Predicted Prices')
    plt.xlabel('Predicted Price (USD)')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid()
    plt.show()

    # Distribution of Residuals
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    ### Step 3: Evaluate Regression Metrics ###
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    aic = model.aic
    bic = model.bic

    metrics_summary = pd.DataFrame({
        'Metric': ['R-squared', 'Adjusted R-squared', 'AIC', 'BIC', 'MSE'],
        'Value': [model.rsquared, model.rsquared_adj, aic, bic, mse]
    })

    print("\n--- Regression Metrics Summary ---")
    print(metrics_summary)

    ### Step 4: T-Test and Confidence Interval Analysis ###
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

    print("\n--- T-Test and Confidence Interval Analysis ---")
    print(t_tests)

    ### Step 5: Recursive Feature Elimination (RFE) ###
    # Recursive Feature Elimination for Top Features
    estimator = LinearRegression()
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X_train, y_train)

    # Selected Features
    selected_features = X.columns[rfe.support_]
    print("\n--- Selected Features from Stepwise Regression ---")
    print(selected_features)

    # Refit with selected features
    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]

    X_train_reduced_const = sm.add_constant(X_train_reduced)
    X_test_reduced_const = sm.add_constant(X_test_reduced)

    stepwise_model = sm.OLS(y_train, X_train_reduced_const).fit()
    print("\n--- Stepwise Regression Model Summary ---")
    print(stepwise_model.summary())

    # Residuals for RFE Model
    residuals_rfe = y_test - stepwise_model.predict(X_test_reduced_const)

    # Residual Plot for RFE Model
    plt.figure(figsize=(8, 6))
    plt.scatter(stepwise_model.predict(X_test_reduced_const), residuals_rfe, alpha=0.7, edgecolor='k')
    plt.axhline(0, color='red', linestyle='--', label='Zero Residuals')
    plt.title('Residuals vs. Predicted Prices (Stepwise Model)')
    plt.xlabel('Predicted Price (USD)')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid()
    plt.show()
