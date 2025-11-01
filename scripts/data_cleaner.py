import pandas as pd
import re
import numpy as np

def clean_price(price_str):
    """Removes currency symbols and commas, then converts to numeric."""
    if isinstance(price_str, str):
        price_str = price_str.replace('₦', '').replace(',', '').strip()
        try:
            return pd.to_numeric(price_str)
        except ValueError:
            return np.nan
    return price_str

def extract_bedrooms(text):
    """Extracts the number of bedrooms from a string."""
    match = re.search(r'(\d+)\s*bdrm', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return np.nan

def extract_bathrooms(text):
    """Extracts the number of bathrooms from a string."""
    match = re.search(r'(\d+)\s*bath', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return np.nan

def extract_property_type(text):
    """Extracts the property type from a string."""
    if 'duplex' in text.lower():
        return 'Duplex'
    if 'bungalow' in text.lower():
        return 'Bungalow'
    if 'flat' in text.lower():
        return 'Flat'
    if 'house' in text.lower():
        return 'House'
    return 'Other'

def clean_data(input_file='jiji_ibadan_properties.csv', output_file='cleaned_ibadan_properties.csv'):
    """
    Loads the raw Jiji properties data, cleans it, extracts features, and saves it.
    """
    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Cleaning Price column...")
    df['Price'] = df['Price'].apply(clean_price)

    print("Extracting features from Title and Description...")
    df['Bedrooms'] = df['Title'].apply(extract_bedrooms)
    df['Bathrooms'] = df['Description'].apply(extract_bathrooms)
    df['Property_Type'] = df['Title'].apply(extract_property_type)

    # Fill missing bedrooms and bathrooms, assuming 1 if not mentioned
    df['Bedrooms'].fillna(1, inplace=True)
    df['Bathrooms'].fillna(1, inplace=True)

    # Drop rows where price could not be converted
    df.dropna(subset=['Price'], inplace=True)

    # Select and reorder columns
    cleaned_df = df[['Property_Type', 'Bedrooms', 'Bathrooms', 'Location', 'Price']].copy()

    print(f"Saving cleaned data to {output_file}...")
    cleaned_df.to_csv(output_file, index=False)

    print("\nData cleaning and feature engineering complete.")
    print(f"Cleaned data has {len(cleaned_df)} rows.")
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())

if __name__ == '__main__':
    clean_data()
