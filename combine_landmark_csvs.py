import os
import pandas as pd

# Set the directory containing the csv files
directory = 'landmark_csvs'  # Change this to your directory path

# List of regions to include in the combined csv
regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', '32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T']

# Initialize an empty DataFrame to hold the combined data
combined_df = pd.DataFrame()

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a csv file that ends with '_top_salient.csv'
    if filename.endswith('_top_salient.csv'):
        # Extract the region code from the filename (e.g., '10S' from '10S_top_salient.csv')
        region_code = filename.split('_')[0]
        # Check if the region code is in the list of specified regions
        if region_code in regions:
            # Construct the full path to the file
            file_path = os.path.join(directory, filename)
            # Read the csv file
            df = pd.read_csv(file_path)
            # Add the region code as the first column of the DataFrame
            df['Region'] = region_code
            # Reorder the DataFrame columns to have 'Region' as the first column
            df = df[['Region'] + [col for col in df.columns if col != 'Region']]
            # Append the DataFrame to the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)

            print(f'Region {region_code} added to combined csv file.')

# Save the combined DataFrame to a new csv file
combined_csv_path = os.path.join(directory, '16_regions_top_salient.csv')
combined_df.to_csv(combined_csv_path, index=False)

print(f'Combined csv file saved to {combined_csv_path}.')
