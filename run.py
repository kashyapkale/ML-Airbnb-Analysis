import os
import pandas as pd
from data_cleaning import clean_data
from eda import perform_eda
from feature_selection_pca import feature_selection_and_pca
from regression import run_regression
from classification import run_classification
from clustering import perform_clustering


input_file = os.path.join('', "Listings.csv")

# Step 1: Data Cleaning
data = pd.read_csv(input_file, encoding="latin1")
data_cleaned = clean_data(data)

# Step 2: Exploratory Data Analysis
perform_eda(data_cleaned)

# Step 3: Feature Selection and PCA
feature_selection_and_pca()

# Step 4: Regression Analysis
run_regression()

# Step 5: Classification Analysis
run_classification()

# Step 6: Clustering
perform_clustering()

print("Workflow complete! All results are displayed above.")
