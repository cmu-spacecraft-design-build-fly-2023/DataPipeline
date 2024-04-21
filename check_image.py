from PIL import Image

# Replace 'path_to_image.jpg' with the path to your image file
image_path = '/home/argus-vision/vision/VisionTrainingGround/RCnet/16_regions_dataset/train/image/10S/s2_10S_00001.png'
image = Image.open(image_path)

# Get the size of the image
width, height = image.size

# Print the size
print("Width:", width, "pixels")
print("Height:", height, "pixels")