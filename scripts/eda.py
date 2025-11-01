import pandas as pd
import warnings

# Suppress the FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

def perform_eda(input_file='cleaned_ibadan_properties.csv', output_file='eda_summary.txt'):
    """
    Performs EDA on the cleaned dataset and saves the summary.
    """
    try:
        df = pd.read_csv(input_file)

        with open(output_file, 'w') as f:
            f.write("Exploratory Data Analysis Summary\n")
            f.write("===================================\n\n")

            f.write("1. Dataframe Info:\n")
            # Redirecting df.info() output is a bit tricky, so we'll construct it
            f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
            f.write("Data Types and Non-Null Counts:\n")
            for col in df.columns:
                f.write(f"- {col}: {df[col].dtype}, {df[col].notna().sum()} non-null\n")
            f.write("\n")

            f.write("2. Summary Statistics for Numerical Columns:\n")
            f.write(df[['Bedrooms', 'Bathrooms', 'Price']].describe().to_string())
            f.write("\n\n")

            f.write("3. Distribution of Property Types:\n")
            f.write(df['Property_Type'].value_counts().to_string())
            f.write("\n\n")

            f.write("4. Top 15 Most Common Locations:\n")
            f.write(df['Location'].value_counts().nlargest(15).to_string())
            f.write("\n\n")

            f.write("5. Correlation Matrix:\n")
            # Select only numeric columns for correlation matrix
            numeric_df = df.select_dtypes(include=['number'])
            f.write(numeric_df.corr().to_string())
            f.write("\n\n")

            f.write("===================================\n")
            f.write("EDA Complete.\n")

        print(f"EDA summary saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")

if __name__ == '__main__':
    perform_eda()
