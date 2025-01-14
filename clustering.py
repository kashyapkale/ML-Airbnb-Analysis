
'''
Clustering Analysis
'''

def perform_clustering():
    '''
    K-means Clustering
    '''
    # Import necessary libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    # Load the dataset (adjust path if needed)
    file_path = 'pca_reduced_dataset_no_targets.csv'  # Use PCA-reduced dataset
    data = pd.read_csv(file_path)

    # Select only PCA-transformed features for clustering
    features = [col for col in data.columns if col.startswith('PC')]
    X = data[features]

    # Standardize the data (important for clustering algorithms)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ### Step 1: Elbow Method for Optimal k ###
    wcss = []
    k_values = range(2, 11)  # Test k values from 2 to 10
    for k in k_values:
        print(k)
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Plot WCSS to find the Elbow
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid()
    plt.show()

    ### Step 2: Silhouette Analysis ###
    silhouette_scores = []
    for k in k_values:
        print(k)
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot Silhouette Scores
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='orange')
    plt.title('Silhouette Analysis for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.grid()
    plt.show()

    ### Step 3: Apply K-Means with Optimal k ###
    # Select the optimal k based on Elbow Method and Silhouette Analysis
    optimal_k = 3  # Update this based on the plots
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, init='k-means++')
    cluster_labels = kmeans.fit_predict(X_scaled)
    print("Step 3 completed....")

    # Add cluster labels to the dataset
    data['Cluster'] = cluster_labels

    # Save the clustered dataset
    data.to_csv('kmeans_clustered_dataset.csv', index=False)
    print(f"K-Means clustering completed with {optimal_k} clusters. Results saved as 'kmeans_clustered_dataset.csv'.")

    ### Step 4: Visualize Clusters (Optional) ###
    # Plot clusters using the first two principal components
    plt.figure(figsize=(8, 6))
    for cluster in np.unique(cluster_labels):
        plt.scatter(
            X_scaled[cluster_labels == cluster, 0],
            X_scaled[cluster_labels == cluster, 1],
            label=f'Cluster {cluster}'
        )
    print("Step 4 completed....")
    plt.title(f'K-Means Clustering with {optimal_k} Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.show()


    '''
    DB-Scan
    '''
    # Import necessary libraries
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the dataset (adjust path if needed)
    file_path = 'pca_reduced_dataset_no_targets.csv'  # Use PCA-reduced dataset
    data = pd.read_csv(file_path)

    # Select only PCA-transformed features for clustering
    features = [col for col in data.columns if col.startswith('PC')]
    X = data[features]

    # Standardize the data (important for DBSCAN)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ### Step 1: Find Optimal eps using Nearest Neighbors ###
    # Compute the k-nearest neighbors distance for k = min_samples
    k = 1000  # Adjust based on dataset size
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    # Sort distances and plot
    distances = np.sort(distances[:, k - 1])
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.title("K-Distance Graph for DBSCAN")
    plt.xlabel("Points sorted by distance to {}th nearest neighbor".format(k))
    plt.ylabel("Distance to {}th nearest neighbor".format(k))
    plt.grid()
    plt.show()

    # Choose an optimal eps based on the "knee" of the plot
    optimal_eps = 1.5  # Adjust based on the graph

    ### Step 2: Apply DBSCAN ###
    dbscan = DBSCAN(eps=optimal_eps, min_samples=k)
    cluster_labels = dbscan.fit_predict(X_scaled)

    # Add cluster labels to the dataset
    data['Cluster'] = cluster_labels

    # Save the clustered dataset
    data.to_csv('dbscan_clustered_dataset.csv', index=False)
    print("DBSCAN clustering completed. Results saved as 'dbscan_clustered_dataset.csv'.")

    ### Step 3: Visualize Clusters ###
    # Visualize DBSCAN clusters using the first two principal components
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(cluster_labels)
    for cluster in unique_labels:
        if cluster == -1:  # Noise points
            plt.scatter(
                X_scaled[cluster_labels == cluster, 0],
                X_scaled[cluster_labels == cluster, 1],
                label='Noise',
                alpha=0.5,
                c='gray'
            )
        else:
            plt.scatter(
                X_scaled[cluster_labels == cluster, 0],
                X_scaled[cluster_labels == cluster, 1],
                label=f'Cluster {cluster}'
            )
    plt.title("DBSCAN Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid()
    plt.show()

    ### Step 4: Evaluate DBSCAN Clustering ###
    # Count the number of clusters (excluding noise)
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)

    print(f"Number of clusters: {num_clusters}")
    print(f"Number of noise points: {num_noise}")


    '''
    Association Rule Mining
    '''
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules
    import matplotlib.pyplot as plt

    # Load the dataset
    file_path = 'processed_airbnb_data_with_categories.csv'
    df = pd.read_csv(file_path, encoding='latin1')

    # Select the relevant columns for association analysis
    columns = [
        'WiFi', 'Kitchen Essentials', 'Safety Features', 'Entertainment',
        'Comfort Items', 'Laundry', 'Outdoor', 'Child-Friendly',
        'Accessibility', 'Workspace', 'Parking'
    ]

    # Check for missing values in the selected columns
    if df[columns].isnull().any().any():
        print("Missing values found. Filling missing values with False.")
        df[columns] = df[columns].fillna(False)

    # Convert selected columns to boolean (True/False)
    df_encoded = df[columns].astype(bool)

    # Verify the encoding
    print("Data Encoding Verification:")
    print(df_encoded.head())

    # Apply the Apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

    # Verify frequent itemsets
    print("\nFrequent Itemsets:")
    print(frequent_itemsets.head())

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    # Verify the generated rules
    print("\nAssociation Rules:")
    print(rules.head())

    # Show top 10 rules sorted by lift
    top_rules = rules.sort_values(by='lift', ascending=False).head(10)
    print("\nTop 10 Association Rules by Lift:")
    print(top_rules)

    # Optional: Visualize the top rules
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(top_rules)), top_rules['lift'], align='center')
    plt.xticks(range(len(top_rules)), top_rules['antecedents'].apply(lambda x: ', '.join(list(x))), rotation=90)
    plt.ylabel('Lift')
    plt.title('Top 10 Association Rules by Lift')
    plt.tight_layout()
    plt.show()
