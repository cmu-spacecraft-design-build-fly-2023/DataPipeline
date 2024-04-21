import os

def count_images_in_subfolders(parent_folder):
    image_count = {}
    
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            image_count[subfolder] = 0
            
            for file in os.listdir(subfolder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.webp')):
                    image_count[subfolder] += 1
    return image_count

parent_folder_path = 'New_RC_Data'
image_count = count_images_in_subfolders(parent_folder_path)

# Sort the folders by name and print the image count for each
for folder in sorted(image_count):
    print(f"{folder}: {image_count[folder]}")