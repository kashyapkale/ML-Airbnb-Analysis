import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'processed_airbnb_data_with_categories.csv'
df = pd.read_csv(file_path, encoding='latin1')
data = df

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())
'''
# Define broader categories for amenities
broader_categories = {
    "WiFi": ["Wifi", "Pocket wifi"],
    "Kitchen Essentials": ["Kitchen", "Refrigerator", "Microwave", "Cooking basics", "Oven", "Stove", "Dishes and silverware"],
    "Safety Features": ["Smoke alarm", "Fire extinguisher", "Carbon monoxide alarm", "First aid kit"],
    "Entertainment": ["TV", "Cable TV", "Game console", "Sound system"],
    "Comfort Items": ["Air conditioning", "Heating", "Extra pillows and blankets", "Room-darkening shades"],
    "Laundry": ["Washer", "Dryer", "Drying rack for clothing"],
    "Outdoor": ["Patio or balcony", "BBQ grill", "Garden or backyard", "Outdoor furniture"],
    "Child-Friendly": ["Crib", "High chair", "Children’s books and toys", "Children’s dinnerware", "Baby bath"],
    "Accessibility": ["Elevator", "Single level home", "EV charger"],
    "Workspace": ["Dedicated workspace", "Ethernet connection"],
    "Parking": ["Free parking on premises", "Paid parking on premises", "Free street parking"]
}

# Count occurrences of each broader category
amenities_sum = df[[col for col in df.columns if col in broader_categories]].sum()



# 1. Price Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['price_usd'], kde=True, bins=50)
plt.title("Distribution of Price in USD")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")
plt.show()

# 2. City-wise Price Analysis
plt.figure(figsize=(12, 6))
sns.barplot(x='city', y='price_usd', data=df, estimator='mean')
plt.title("Average Price by City")
plt.xticks(rotation=45)
plt.ylabel("Average Price (USD)")
plt.show()

# 3. Room Type Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='room_type', y='price_usd', data=df)
plt.title("Room Type vs Price")
plt.ylabel("Price (USD)")
plt.show()

# 4. Amenities Analysis
amenities_sum = df[[col for col in df.columns if col in broader_categories]].sum()
amenities_sum.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title("Counts of Listings with Specific Amenities")
plt.ylabel("Count")
plt.show()

# 5. Superhost Impact on Price
plt.figure(figsize=(8, 6))
sns.barplot(x='host_is_superhost', y='price_usd', data=df, estimator='mean')
plt.title("Superhost vs Non-Superhost - Average Price")
plt.ylabel("Average Price (USD)")
plt.xlabel("Superhost (1=Yes, 0=No)")
plt.show()

# 6. Review Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['review_scores_rating'], kde=True, bins=30)
plt.title("Distribution of Review Scores")
plt.xlabel("Review Scores")
plt.ylabel("Frequency")
plt.show()

# 7. Correlation Heatmap
# Exclude specific columns from the correlation matrix
columns_to_exclude = ['listing_id', 'host_id']
numeric_data = df.select_dtypes(include=['float64', 'int64']).drop(columns=columns_to_exclude)

# Compute and plot the correlation matrix
# Compute and rearrange the correlation matrix
correlation_matrix = numeric_data.corr()
sorted_corr = correlation_matrix['price_usd'].sort_values(ascending=False)

# Plot heatmap focused on price_usd correlations
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix.loc[sorted_corr.index, sorted_corr.index], cmap='mako', annot=True, fmt=".2f")
plt.title("Correlation Heatmap (Focused on Price USD)")
plt.show()


# 8. Superhost Percentage by City
superhost_data = df.groupby(['city', 'host_is_superhost'])['listing_id'].count().unstack()
superhost_percentage = (superhost_data[1] / superhost_data.sum(axis=1)) * 100

plt.figure(figsize=(12, 6))
superhost_percentage.sort_values().plot(kind='bar', color='skyblue')
plt.title("Percentage of Superhosts by City")
plt.ylabel("Percentage of Superhosts (%)")
plt.xlabel("City")
plt.show()

'''