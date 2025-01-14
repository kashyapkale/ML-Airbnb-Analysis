'''
Feature Selection
'''
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feature_selection_and_pca():
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


    '''
    PCA
    '''

    '''
    Apply PCA for Dimensionality Reduction
    Now that we've reduced the dataset to important features, we can apply PCA to check
    if further dimensionality reduction is possible without significant loss of variance.
    PCA will help us identify the most influential principal components for clustering, regression, or classification tasks.
    '''
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
