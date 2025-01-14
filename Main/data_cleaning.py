import pandas as pd
import numpy as np

def clean_data(df):
    data = df

    '''
    1. Check for duplicates
    2. If duplicates exist, remove them
    3. Checking for missing values
    4. Calculate percentage of missing values
    5. Create a DataFrame with missing values count and percentage
    '''
    # Check for duplicates
    duplicates = data.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")

    # If duplicates exist, remove them
    data.drop_duplicates(inplace=True)
    print(f"Dataset shape after removing duplicates: {data.shape}")

    df.dropna(subset=['price', 'city', 'room_type'], inplace=True)

    # Check for missing values
    missing_values = data.isnull().sum()
    total_rows = data.shape[0]

    # Calculate percentage of missing values
    missing_percentage = (missing_values / total_rows) * 100

    # Create a DataFrame with missing values count and percentage
    missing_values_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percentage
    }).sort_values(by='Percentage (%)', ascending=False)

    print(missing_values_df.head(15))


    # Drop columns with excessive missing data
    columns_to_drop = ['district', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'neighbourhood', 'latitude', 'longitude']
    data.drop(columns=columns_to_drop, inplace=True)
    print(f"Columns dropped: {columns_to_drop}")
    print(f"Dataset shape after column drop: {data.shape}")

    # Impute missing values
    # Numeric columns: Fill missing with median
    df.dropna(subset=['city', 'listing_id', 'name', 'host_id'], inplace=True)

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Categorical columns: Fill missing with mode
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Verify if any missing values remain
    print("\nRemaining Missing Values per Column:")
    print(data.isnull().sum())


    print(f"Dataset shape after handling missing values drop: {data.shape}")

    # Converting prices from local currency to USD
    city_prices = df.groupby('city')['price'].describe()
    print(city_prices)

    exchange_rates = {
        'Bangkok': 0.030,         # USD
        'Cape Town': 0.057,
        'Hong Kong': 0.13,
        'Istanbul': 0.029,
        'Mexico City':0.050,
        'New York':1,
        'Paris':1.08,
        'Rio de Janeiro':0.175,
        'Rome':1.08,
        'Sydney':0.66
    }

    df['exchange_rate'] = df['city'].map(exchange_rates)
    df['price_usd'] = df['price'] * df['exchange_rate']

    city_prices_usd = df.groupby('city')['price_usd'].describe()
    print(city_prices_usd)


    # Outlier detection using z-score
    # Calculate the z-scores for the column 'price_usd'
    data['z_score'] = (data['price_usd'] - data['price_usd'].mean()) / data['price_usd'].std()

    # Set a threshold for z-scores (commonly 3 or 3.5 for extreme outliers)
    z_threshold = 3

    # Identify outliers
    outliers = data[np.abs(data['z_score']) > z_threshold]

    # Print the number of outliers
    print(f"Number of outliers detected by z-score: {len(outliers)}")

    data_cleaned = data[np.abs(data['z_score']) <= z_threshold].drop(columns=['z_score'])

    # Output the cleaned DataFrame
    print(f"Original DataFrame shape: {data.shape}")
    print(f"Cleaned DataFrame shape: {data_cleaned.shape}")

    # Drop columns with excessive missing data
    columns_to_drop = ['z_score', 'price']
    data.drop(columns=columns_to_drop, inplace=True)

    #Checking top aminities
    from collections import Counter

    # Step 1: Clean and split the 'amenities' column
    amenities = df['amenities'].str.replace('[\[\]"]', '', regex=True).str.split(', ')

    # Step 2: Flatten the list of amenities and count frequencies
    amenity_counts = Counter([item for sublist in amenities.dropna() for item in sublist])

    # Step 3: Sort amenities by frequency (descending)
    sorted_amenities = sorted(amenity_counts.items(), key=lambda x: x[1], reverse=True)

    # Step 4: Display the sorted list of amenities with frequencies
    for amenity, count in sorted_amenities:
        print(f"{amenity}: {count}")


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

    # Create binarized columns for each broader category
    for category, keywords in broader_categories.items():
        df[category] = df['amenities'].apply(
            lambda x: 1 if any(keyword in str(x) for keyword in keywords) else 0
        )

    # Drop the original 'amenities' column as it's no longer needed
    df.drop(columns=['amenities'], inplace=True)

    # Display the updated DataFrame with new binarized columns
    print("Updated DataFrame with Binarized Amenities:")
    print(df.info())

    # Identify columns that can be boolean based on the data type and unique values
    boolean_columns = []

    # Define possible boolean-like values
    boolean_values = ['yes', 'no', 'true', 'false', 't', 'f']

    # Iterate through columns to check for possible boolean values
    for col in df.columns:
        # For object columns (which may contain 'yes'/'no'/'t'/'f' or 'true'/'false')
        if df[col].dtype == 'object' and df[col].isin(boolean_values).any():
            # Replace 'yes', 'true', 't' with 1 and 'no', 'false', 'f' with 0
            df[col] = df[col].replace({'yes': 1, 'no': 0, 'true': 1, 'false': 0, 't': 1, 'f': 0})
            boolean_columns.append(col)

        # For numeric columns that contain only 0s and 1s
        elif df[col].dtype in ['int64', 'float64'] and df[col].isin([0, 1]).all():
            boolean_columns.append(col)

    # Print the updated DataFrame and boolean columns
    print("Updated DataFrame with 1/0 values:")
    print(df)

    print("\nBoolean columns in the dataset:")
    print(boolean_columns)


    # Save the processed DataFrame for later use
    df.to_csv("processed_airbnb_data_with_categories.csv", index=False)
    print("Processed data saved to 'processed_airbnb_data_with_categories.csv'")
    return df
