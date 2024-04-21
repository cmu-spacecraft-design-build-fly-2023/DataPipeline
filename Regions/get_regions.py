import csv

# File paths
input_file_path = 'reduced_prioritized_regions.csv'
output_file_path = 'regions_in_separate_rows.csv'

# Read the comma-separated regions from the input file
regions = []
with open(input_file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        regions = row  # Assuming all regions are in the first row

# Save the regions into separate rows in a new CSV file
with open(output_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for region in regions:
        writer.writerow([region])  # Write each region in its own row

print(f"Saved the MGRS regions into separate rows in '{output_file_path}'")