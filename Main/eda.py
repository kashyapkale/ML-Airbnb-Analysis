
'''
Exploratory Data Analysis
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def perform_eda(df):
    data = df

    # Detect and remove outliers in 'price_usd' using IQR
    Q1, Q3 = data['price_usd'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['price_usd'] >= lower_bound) & (data['price_usd'] <= upper_bound)]

    # Display basic information about the dataset
    print("Dataset Info:")
    print(data.info())

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



    # 2. City-wise Price Analysis
    plt.figure(figsize=(12, 6))
    sns.barplot(x='city', y='price_usd', data=df, estimator='mean')
    plt.title("Average Price by City")
    plt.xticks(rotation=45)
    plt.ylabel("Average Price (USD)")
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

    # 9.
    # Identify the top 10 most occurring property types
    top_property_types = data['property_type'].value_counts().head(10).index

    # Filter the data to include only the top 10 property types
    filtered_data = data[data['property_type'].isin(top_property_types)]

    # Plot the boxplot for the top 10 property types
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='property_type', y='price_usd', data=filtered_data)
    plt.xticks(rotation=45)
    plt.title("Price Distribution by Top 10 Property Types")
    plt.ylabel("Price (USD)")
    plt.show()


    # 10.
    city_amenities = df.groupby('city')[[col for col in broader_categories]].sum()
    city_amenities.T.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='coolwarm')
    plt.title("Amenities Availability by City")
    plt.ylabel("Count")
    plt.xlabel("Amenities")
    plt.legend(title="City")
    plt.show()

    # 12.
    # Convert 'host_since' to a datetime column
    data['host_since_year'] = pd.to_datetime(data['host_since']).dt.year

    # Group by host_since_year and calculate average price
    host_experience_price = data.groupby('host_since_year')['price_usd'].mean().reset_index()


    # 13.
    price_accommodation = data.groupby('accommodates')['price_usd'].mean()
    plt.figure(figsize=(10, 6))
    price_accommodation.plot(kind='line', marker='o')
    plt.title("Average Price vs Accommodation Capacity")
    plt.xlabel("Number of Guests Accommodated")
    plt.ylabel("Average Price (USD)")
    plt.grid()
    plt.show()


    amenities_impact = data.groupby('WiFi')['price_usd'].mean()
    plt.figure(figsize=(8, 6))
    amenities_impact.plot(kind='bar', color='orchid')
    plt.title("Impact of Amenities on Average Price")
    plt.xlabel("Amenity")
    plt.ylabel("Average Price (USD)")
    plt.xticks(rotation=45)
    plt.show()

    superhost_roomtype = data.groupby(['room_type', 'host_is_superhost']).size().unstack()
    plt.figure(figsize=(8, 6))
    sns.heatmap(superhost_roomtype, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Superhost Distribution Across Room Types")
    plt.xlabel("Superhost (0=No, 1=Yes)")
    plt.ylabel("Room Type")
    plt.show()

    '''
    '''
    city_price = data.groupby('city')['price_usd'].mean().reset_index()
    city_listings = data['city'].value_counts().reset_index()
    city_listings.columns = ['city', 'listings']

    # Merge datasets to include the number of listings as bubble size
    city_price = city_price.merge(city_listings, on='city')

    plt.figure(figsize=(12, 6))
    plt.scatter(city_price['city'], city_price['price_usd'], s=city_price['listings'], alpha=0.6, c=city_price['price_usd'], cmap='viridis')
    plt.colorbar(label='Average Price (USD)')
    plt.title("City-Wise Price Analysis with Bubble Size Representing Listings")
    plt.xticks(rotation=45)
    plt.ylabel("Average Price (USD)")
    plt.xlabel("City")
    plt.show()

    '''
    '''
    city_amenities = df.groupby('city')[[col for col in broader_categories]].sum()
    city_amenities = city_amenities.T

    plt.figure(figsize=(14, 7))
    city_amenities.plot(kind='area', stacked=True, figsize=(14, 7), colormap='Spectral')
    plt.title("Amenities Availability by City (Stacked Area Plot)")
    plt.ylabel("Total Count of Amenities")
    plt.xlabel("Amenities")
    plt.legend(title="City", loc='upper left')
    plt.grid(alpha=0.5)
    plt.show()

    '''
    '''
    host_experience_price = data.groupby('host_since_year')['price_usd'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.fill_between(host_experience_price['host_since_year'], host_experience_price['price_usd'], color='skyblue', alpha=0.4)
    plt.plot(host_experience_price['host_since_year'], host_experience_price['price_usd'], color='blue', linewidth=2)
    plt.title("Average Price vs Host Experience (Years)")
    plt.xlabel("Host Since Year")
    plt.ylabel("Average Price (USD)")
    plt.grid(alpha=0.5)
    plt.show()


    '''
    '''
    superhost_roomtype = data.groupby(['room_type', 'host_is_superhost']).size().reset_index(name='count')
    superhost_roomtype['host_is_superhost'] = superhost_roomtype['host_is_superhost'].map({0: 'Non-Superhost', 1: 'Superhost'})

    fig = px.sunburst(superhost_roomtype, path=['room_type', 'host_is_superhost'], values='count',
                      color='count', color_continuous_scale='RdYlBu',
                      title="Superhost Distribution Across Room Types")
    fig.show()


    '''
    '''
    amenities_impact = data.groupby('WiFi')['price_usd'].mean()

    plt.figure(figsize=(10, 6))
    amenities_impact.sort_values().plot(kind='barh', color='mediumseagreen')
    plt.title("Impact of Amenities on Average Price (Horizontal Bar)")
    plt.xlabel("Average Price (USD)")
    plt.ylabel("Amenities")
    plt.show()
