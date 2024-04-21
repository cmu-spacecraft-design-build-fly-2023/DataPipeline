import os

def count_files_in_directory(directory, prefix=''):
    try:
        # List all directory contents
        entries = os.listdir(directory)
        count_files = 0
        directories = []

        # Iterate through each entry in the directory
        for entry in entries:
            full_path = os.path.join(directory, entry)
            # If entry is a directory, add to directories list for later processing
            if os.path.isdir(full_path):
                directories.append(entry)
            else:
                # If entry is a file, increment count
                count_files += 1

        # Print the current directory and its file count
        print(f"{prefix}+-- {os.path.basename(directory)} [{count_files}]")

        # New prefix for subdirectories
        new_prefix = prefix + "|   "
        # Process each subdirectory found
        for i, subdir in enumerate(directories):
            # Adjust the prefix for the last directory
            if i == len(directories) - 1:
                new_prefix = prefix + "    "
            count_files_in_directory(os.path.join(directory, subdir), new_prefix)
    except PermissionError:
        # Skip directories that cannot be accessed
        print(f"{prefix}+-- {os.path.basename(directory)} [Access Denied]")

# Specify the top-level directory
top_level_directory = '/home/argus-vision/vision/VisionTrainingGround/RCnet/16_regions_dataset'
count_files_in_directory(top_level_directory)