import csv

# File paths
input_file_path = 'prioritized_regions.csv'
output_file_path = 'reduced_prioritized_regions.csv'

# Read the first 20 regions from the input file
first_20_regions = []
with open(input_file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        first_20_regions = row[:20]  # Take the first 20 entries
        break  # Assuming all regions are in the first row

# Save the first 20 regions into a new CSV file
with open(output_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(first_20_regions)  # Write as a single row

print("Saved the first 20 MGRS regions to 'reduced_prioritized_regions.csv'")
