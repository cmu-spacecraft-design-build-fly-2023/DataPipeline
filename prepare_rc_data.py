import numpy as np
import glob
import csv
import os
import warnings
import pyproj
import rasterio as rs
from rasterio.windows import Window
from PIL import Image
import imageio
import argparse
import cv2
import yaml
from tqdm import tqdm  # Import tqdm at the beginning of your script
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map  # Import process_map

warnings.filterwarnings('ignore')

class_colors = {
    "10S": (255, 0, 0),  # Red
    "10T": (0, 255, 0),  # Green
    "11R": (0, 0, 255),  # Blue
    # Add more classes and colors as needed
}

# check if box UL and BR corners are in bounds of image
# Note: this is in whole image, including blacked out regions
def check_bounds(long_lat_coords, src):
    pp = pyproj.Proj(init=src.crs)

    xmin, ymax = pp(long_lat_coords[0][0], long_lat_coords[0][1])
    xmax, ymin = pp(long_lat_coords[1][0], long_lat_coords[1][1])

    bounds = src.bounds
    if (bounds.left < xmin < bounds.right and bounds.bottom < ymax < bounds.top
        and bounds.left < xmax < bounds.right and bounds.bottom < ymin < bounds.top):
        if check_box_vals(long_lat_coords, src):
            return True
    return False

# convert long lat coords to pixel xy
def longlat_to_xy(long_lat_coords, src):
    meta = src.meta

    xy_coords = []
    for coord in long_lat_coords:
        pp = pyproj.Proj(init=src.crs)

        lon, lat = pp(coord[0], coord[1])

        py, px = src.index(lon, lat)
        xy_coords.append([px, py])

    return xy_coords

# check if box lies in no-data region of image (invalid)
def check_box_vals(long_lat_coords, src):
    xy_coords = longlat_to_xy(long_lat_coords, src)

    xsize = xy_coords[1][0] - xy_coords[0][0]
    ysize = xy_coords[1][1] - xy_coords[0][1]
    window = Window(xy_coords[0][0], xy_coords[0][1], xsize, ysize)
    arr = src.read(window=window)
    arr_sum = np.sum(arr, axis=0)
    zero = np.count_nonzero(arr_sum == 0) / (xsize*ysize)

    if zero >= 0.5:
      return False
    else:
      return True

# convert tif images to png and save
def convert_tif_to_png(tif_path, png_path):
    with rs.open(tif_path) as src:
        img = src.read()  # Read the image (bands, rows, columns)
        
        # Normalize and scale floating-point images to 8-bit
        if np.issubdtype(img.dtype, np.floating):
            # Normalize the image to 0-1
            img -= img.min()
            if img.max() != 0:
                img /= img.max()
            # Scale to 0-255 and convert to uint8
            img = (255*img).astype(np.uint8)
        
        # Handle single-band (grayscale) images
        if img.shape[0] == 1:
            img_squeezed = np.squeeze(img, axis=0)
        else:
            img_squeezed = img.transpose((1, 2, 0))  # Reorder dimensions for multi-band images
            
        imageio.imwrite(png_path, img_squeezed)


def draw_landmarks(image, landmarks, class_colors):
    """
    Draws landmarks on the image with colors specific to their classes.

    Args:
        image (np.array): The image array.
        landmarks (dict): A dictionary of detected landmarks and their classes.
        class_colors (dict): A dictionary mapping class names to BGR color values.
    """
    for _, landmark in landmarks.items():
        coords, cls = landmark["coords"], landmark["class"]
        center = coords[2]
        color = class_colors.get(cls, (255, 255, 255))  # Default to white if class not found
        # Assuming coords are (x, y), adjust as necessary based on your coordinate system
        cv2.circle(image, center, radius=5, color=color, thickness=-1)

def draw_landmarks_with_legend(image, landmarks, class_colors):
    """
    Draws landmarks on the image with colors specific to their classes and adds a legend for colors
    only for the classes that appeared in the landmarks.

    Args:
        image (np.array): The image array.
        landmarks (dict): A dictionary of detected landmarks and their classes.
        class_colors (dict): A dictionary mapping class names to BGR color values.
    """
    # Initialize a set to keep track of classes present in the detected landmarks
    present_classes = set()

    # First, draw the landmarks as before
    for _, landmark in landmarks.items():
        coords, cls = landmark["coords"], landmark["class"]
        center = coords[2]  # Assuming this is a tuple of (x, y)
        color = class_colors.get(cls, (255, 255, 255))  # Default to white if class not found
        cv2.circle(image, (int(center[0]), int(center[1])), radius=5, color=color, thickness=-1)
        present_classes.add(cls)  # Add the class to the set of present classes
    
    # Now, add a legend for the class colors of the present classes
    text_start_y = 20  # Starting Y position of the first class name
    for cls in present_classes:
        color = class_colors.get(cls, (255, 255, 255))  # Get the color for the current class
        legend_text = f"{cls}:"
        
        # Put the class name text on the image
        cv2.putText(image, legend_text, (10, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2, cv2.LINE_AA)
        
        text_start_y += 20  # Move to the next line for the next class name

# Function to process a single image
def process_single_image(file, data_path, all_landmarks, region):
    detected_boxes = {}
    img_name = os.path.basename(file).split('.')[0]
    detected_boxes[img_name] = {}

    # Open the source file as an image array
    src_im = cv2.imread(file)  # Use cv2.imread instead of rs.open for direct image array access
    src = rs.open(file)

    for idx, landmark in all_landmarks.items():
        box, cls = landmark[:-1], landmark[-1]  # Split the box coordinates and the class
        if check_bounds(box, src):
            xy_coords = longlat_to_xy(box, src)
            detected_boxes[img_name][idx] = {
                "coords": xy_coords,  # Ensure coordinates are integer tuples
                "class": cls
            }

    # Draw landmarks on the image
    draw_landmarks_with_legend(src_im, detected_boxes[img_name], class_colors)

    # Save training image to png
    image_dir = os.path.join(data_path, f"image/{region}")  # Construct the path to the 'image' subdirectory
    os.makedirs(image_dir, exist_ok=True)  # Ensure the 'image' directory exists
    im_out_path = os.path.join(image_dir, img_name + '.png')
    convert_tif_to_png(file, im_out_path)

    # Save image with landmarks for label validation
    landmarks_dir = os.path.join(data_path, f"landmarks/{region}")  # Construct the path to the 'image' subdirectory
    os.makedirs(landmarks_dir, exist_ok=True)  # Ensure the 'image' directory exists
    landmarks_out_path = os.path.join(landmarks_dir, img_name + '.png')
    cv2.imwrite(landmarks_out_path, src_im)  # Use cv2.imwrite to save the image

    return detected_boxes

def generate_single_label(img_name, detected_boxes, label_path, regions, region):
    """
    Generates a label file for a single image based on detected landmark classes.

    Args:
        img_name (str): The name of the image without extension.
        detected_boxes (dict): A dictionary containing detected landmarks and their classes.
        regions (list): A list of region classes to check against.
        label_path (str): The directory path where the label file will be saved.

    Creates a label file named <img_name>.txt in the label_path directory, containing
    a binary vector indicating the presence of each region class among the detected landmarks.
    """
    label_dir = os.path.join(label_path, f"label/{region}")  # Construct the path to the 'image' subdirectory
    os.makedirs(label_dir, exist_ok=True)  # Ensure the 'image' directory exists
    label_file = os.path.join(label_dir, img_name + '.txt')
    # Initialize a label vector with all zeros
    label_vector = [0] * len(regions)

    # Check each detected landmark's class and update the label vector accordingly
    if img_name in detected_boxes:
        for idx, detection in detected_boxes[img_name].items():
            cls = detection["class"]
            if cls in regions:
                label_vector[regions.index(cls)] = 1

    # Write the label vector to a file
    with open(label_file, 'w') as f:
        label_line = ' '.join(map(str, label_vector))
        f.write(label_line)

def process_single_image_wrapper(args):
    return process_single_image(*args)

def generate_single_label_wrapper(args):
    return generate_single_label(*args)

def process_data_with_pool(args):
    val_set = args.val
    test_set = args.test
    data_path = args.data_path
    landmark_path = args.landmark_path
    output_path = args.output_path
    region = args.region
    anno_file = os.path.join(landmark_path, f"16_regions_top_salient.csv")

    # Process the resolution argument
    resolution = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except ValueError:
            raise ValueError("Resolution must be specified as WIDTHxHEIGHT")
    
    # Make sure the images output directory exists
    if test_set is True:
        data_path = os.path.join(data_path, f"{region}_S2_test")
        save_path = os.path.join(output_path, f"test/")
    elif val_set is True:
        data_path = os.path.join(data_path, f"{region}_S2_val")
        save_path = os.path.join(output_path, f"val/")
    else:
        data_path = os.path.join(data_path, f"{region}_S2_train")
        save_path = os.path.join(output_path, f"train/")
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    print("save_path:", save_path)

    # Load landmarks
    all_landmarks = {}
    with open(anno_file, 'r') as f:
        csvReader = csv.DictReader(f)
        for i, row in enumerate(csvReader):
            cls = row["Region"]
            min_long = float(row["Top-Left Longitude"])
            max_lat = float(row["Top-Left Latitude"])
            max_long = float(row["Bottom-Right Longitude"])
            min_lat = float(row["Bottom-Right Latitude"])
            centroid_lon = float(row["Centroid Longitude"])
            centroid_lat = float(row["Centroid Latitude"])
            all_landmarks[i] = [[min_long, max_lat], [max_long, min_lat], [centroid_lon, centroid_lat], cls]

    # Glob .tif files for the test set from multiple directories
    files = glob.glob(os.path.join(data_path, '*.tif'))

    # Use process_map to process images with progress tracking
    process_args = [(file, save_path, all_landmarks, region) for file in files]
    detected_boxes_list = process_map(process_single_image_wrapper, process_args, chunksize=1, max_workers=None, desc="Processing Images")

    # Combine detected_boxes from all images
    detected_boxes = {}
    for boxes in detected_boxes_list:
        detected_boxes.update(boxes)

    regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', '32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T']

    # Use process_map to generate label files with progress tracking
    generate_args = [(img_name, detected_boxes, save_path, regions, region) for img_name in detected_boxes]
    process_map(generate_single_label_wrapper, generate_args, chunksize=1, max_workers=None, desc="Generating Label Files")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates jpgs and labels in yolo format")
    parser.add_argument("--data_path", required=True, help="path to original tiff images")
    parser.add_argument("--landmark_path", required=True, help="path to landmark annotation file")
    parser.add_argument("--output_path", required=True, help="output folder path for images and labels")
    parser.add_argument("--val", type=bool, help="Flag for creating sentinal 2 validation dataset")
    parser.add_argument("--test", type=bool, help="Flag for creating rotated Landsat testing dataset")
    parser.add_argument("--resolution", type=str, help="Resolution to which the image should be resized, specified as WIDTHxHEIGHT", default=None)
    parser.add_argument('--region', type=str, default='17R')
    args = parser.parse_args()

    process_data_with_pool(args)
    