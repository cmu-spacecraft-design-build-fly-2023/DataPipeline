import os
import shutil
import glob

# Directories
source_dir = 'New_RC_Data/17R_S2_train_x/'
target_dir = 'New_RC_Data/17R_S2_train/'

# Extract the region ID from the target directory name
region_id = target_dir.split('/')[-2].split('_')[0]

# Find the highest index in the target directory
target_files = glob.glob(os.path.join(target_dir, '*.tif'))

if target_files:  # Check if the list is not empty
    # Correctly extract the index from file names
    highest_index = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in target_files])
else:
    highest_index = -1  # Start from 0 if no files are found

# Copy files from source to target, starting from the next index
next_index = highest_index + 1
source_files = sorted(glob.glob(os.path.join(source_dir, '*.tif')))

for file in source_files:
    if next_index > 999:
        break  # Stop copying if the index exceeds 2000
    
    # Construct new file name based on the next index and region ID
    new_file_name = f's2_{region_id}_{next_index:05d}.tif'
    new_file_path = os.path.join(target_dir, new_file_name)
    
    # Copy file
    shutil.copy(file, new_file_path)
    print(f'Copied {file} to {new_file_path}')
    
    next_index += 1
