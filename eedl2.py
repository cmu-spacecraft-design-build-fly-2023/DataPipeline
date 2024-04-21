"""
eedl.py
Earth Engine Downloader
A script to download satellite images from the Google Earth Engine API. 
The script can download images from the Landsat 8, Landsat 9, and Sentinel 2 sensors.
The script can download images from a specified geographical region, date range, and cloud cover percentage range.
The script can also download images from a specified MGRS grid region.
The script can also create custom mosaics by selecting random points within a region and creating a mosaic around each point.
Images are downloaded in either the GeoTIFF or PNG format.
Custom mosaic images are saved directly to Google Drive.
Default Landsat and Sentinel images are saved to the local file system.
Author: Kyle McCleary
"""

import argparse
import os
import shutil
from multiprocessing import cpu_count
import ee
import requests
from retry import retry
import numpy as np
import pyproj
from tqdm.contrib.concurrent import process_map
from getMGRS import getMGRS

ee.Initialize()

def get_region_filter_from_bounds(bounds, get_rect=True):
    """
    Creates a filter for a given geographical rectangle defined by longitude and latitude bounds.

    Parameters:
    bounds (list): A list of four elements [left, top, right, bottom] defining the geographical rectangle.
    get_rect (bool): A flag to determine whether to return the rectangle geometry.

    Returns:
    ee.Filter: A filter that selects images intersecting with the defined rectangle.
    ee.Geometry.Rectangle (optional): The rectangle geometry, returned if getRect is True.
    """
    region_left, region_bottom, region_right, region_top = bounds
    rect_from_bounds = ee.Geometry.Rectangle([region_left, region_top, region_right, region_bottom])
    
    # If sensor is Sentinel 2, add 500 km buffer to bounds of grid region to avoid black sections of images.
    if args.sensor == 's2':
        out_rect = rect_from_bounds.buffer(500000)
    
    # If sensor is Landsat, do not add buffer.
    else:
        out_rect = rect_from_bounds
    region_filter_from_bounds = ee.Filter.bounds(out_rect)
    if get_rect:
        return region_filter_from_bounds, rect_from_bounds
    return region_filter_from_bounds

def get_date_filter(i_date, f_date):
    """
    Creates a date filter for selecting images within a specified date range.

    Parameters:
    i_date (str): Initial date of the date range in a format recognizable by the Earth Engine API. (e.g. '2022-01-01')
    f_date (str): Final date of the date range in a format recognizable by the Earth Engine API. (e.g. '2023-01-01')

    Returns:
    ee.Filter: A date filter for the specified date range.
    """
    ee_date_filter = ee.Filter.date(i_date, f_date)
    return ee_date_filter

def get_collection(sensor, ee_region_filter, ee_date_filter, ee_bands = None, cloud_cover_min = 0.0, cloud_cover_max = 30.0, date_sort=True):
    """
    Retrieves a filtered collection of Landsat images based on the specified parameters.

    Parameters:
    sensor (str): The sensor to pull images from. Options are l8, l9, or s2.
    ee_region_filter (ee.Filter): The geographical region filter.
    ee_date_filter (ee.Filter): The date range filter.
    ee_bands (list): A list of band names to include in the collection. Default is ['B4', 'B3', 'B2'].
    cloud_cover_min (float): The minimum cloud cover percentage for the images. Default is 0.0.
    cloud_cover_max (float): The maximum cloud cover percentage for the images. Default is 30.0.
    date_sort (bool): Flag to sort the collection by acquisition date.

    Returns:
    ee.ImageCollection: A collection of Landsat images filtered by the specified parameters.
    """
    if sensor == 'l8':
        collection_string = 'LANDSAT/LC08/C02/T1_TOA'
        cloud_string = 'CLOUD_COVER'
    elif sensor == 'l9':
        collection_string = 'LANDSAT/LC09/C02/T1_TOA'
        cloud_string = 'CLOUD_COVER'
    elif sensor == 's2':
        collection_string = 'COPERNICUS/S2_HARMONIZED'
        cloud_string = 'CLOUDY_PIXEL_PERCENTAGE'
    elif sensor == 'mds':
        collection_string = 'MODIS/061/MOD09GA'
        cloud_string = 'CLOUD_COVER'

    if ee_bands is None:
        ee_bands = ['B4', 'B3', 'B2']

    ee_collection = ee.ImageCollection(collection_string)
    ee_collection = ee_collection.filter(ee_date_filter)
    if not args.custom_mosaics:
        ee_collection = ee_collection.filter(ee_region_filter)
    ee_collection = ee_collection.filter(ee_date_filter)
    ee_collection = ee_collection.filter(ee.Filter.lt(cloud_string, cloud_cover_max))
    ee_collection = ee_collection.filter(ee.Filter.gte(cloud_string, cloud_cover_min))
    ee_collection = ee_collection.select(ee_bands)
    if date_sort:
        ee_collection = ee_collection.sort('DATE_ACQUIRED')
    return ee_collection

def get_points_in_region(ee_region, num_points, pts_scale, pts_seed):
    """
    Selects random points within a specified region, focusing on land areas. Uses MODIS land/water data to filter out water bodies.
    
    Parameters:
    ee_region (ee.Geometry): The region within which to select points.
    num_points (int): The number of random points to select.
    pts_scale (float): The scale to sample points
    seed (int): A seed number for the random point generation to ensure reproducibility.

    Returns:
    list: A list of randomly selected geographical points (longitude and latitude) within the specified region.
    """
    water_land_data = ee.ImageCollection('MODIS/061/MCD12Q1')
    land = water_land_data.select('LW').first()
    mask = land.eq(2)
    selected_points = land.updateMask(mask).stratifiedSample(region=ee_region, scale = pts_scale,
                                                    classBand = 'LW', numPoints = num_points,
                                                    geometries=True,seed = pts_seed)
    return selected_points.aggregate_array('.geo').getInfo()

def make_rectangle(ee_point, h_pt_buffer, v_pt_buffer = None):
    """
    Creates a rectangle geometry around a given point.

    Parameters:
    ee_point (dict): A dictionary containing the 'coordinates' key, which holds the longitude and latitude of the point.
    h_pt_buffer (float): A float value containing the radius in meters to horizontally extend rectangle bounds from the point.
    v_pt_buffer (float): A float value containing the radius in meters to vertically extend rectangle bounds from the point. Defaults to None.

    Returns:
    ee.Geometry.Rectangle: A rectangle geometry centered around the given point with a fixed buffer.
    """
    if v_pt_buffer is None:
        v_pt_buffer = h_pt_buffer
    coords = ee_point['coordinates']

    if args.grid_key is None:
        projection = "EPSG:4326"
    elif args.grid_key[-1] <= 'M':
        projection = "EPSG:327" + args.grid_key[:-1]
    else:
        projection = "EPSG:326" + args.grid_key[:-1]

    transformer = pyproj.Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    transformed_pt = tuple(transformer.transform(coords[0], coords[1]))
    pt_tl_x = transformed_pt[0] - h_pt_buffer
    pt_tl_y = transformed_pt[1] + v_pt_buffer
    pt_br_x = transformed_pt[0] + h_pt_buffer
    pt_br_y = transformed_pt[1] - v_pt_buffer
    pt_rect = ee.Geometry.Rectangle([pt_tl_x, pt_br_y, pt_br_x, pt_tl_y], projection, True, False).bounds()
    return pt_rect

def get_url(index, im_list):
    """
    Generates a download URL for a satellite image from the Earth Engine image collection.

    Parameters:
    index (int): The index of the image in the Earth Engine image list.

    Returns:
    str: A URL string from which the image can be downloaded.
    """
    image = ee.Image(im_list.get(index))
    if args.crs:
        crs = args.crs
    else:
        if args.sensor in ('l8','l9','mds'):
            crs = image.select(0).projection()
        else:
            if args.grid_key[-1] <= 'M':
                crs = "EPSG:327" + args.grid_key[:-1]
            else:
                crs = "EPSG:326" + args.grid_key[:-1]
    if args.sensor in ('l8','l9','mds'):
        image = image.multiply(255/0.3).toByte()
        image = image.clip(image.geometry())
    url = image.getDownloadURL({
        'scale':scale,
        'format':out_format,
        'bands':bands,
        'crs':crs})
    #print('URL',index,'done: ', url)
    return url

@retry(tries=10, delay=1, backoff=2)
def get_and_download_url(index, im_list, out_path, out_format, sensor, region_name):
    """
    Downloads an image from a retrieved URL and saves it to a specified path. 
    This function will retry up to 10 times with increasing delays if the download fails.

    Parameters:
    index (int): The index of the image, used for retrieving the URL and naming the downloaded file.

    Notes:
    The file is saved in either the GeoTIFF or PNG format, depending on the 'out_format' variable.
    The file name is constructed using the sensor, region name, and index.
    """
    url = get_url(index, im_list)
    print('Retrieved URL',index,':',url)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(out_path, 'folder created')
    if out_format == 'GEOTiff':
        ext = '.tif'
    else:
        ext = '.png'
    out_name = sensor + '_' + region_name + '_' + str(index).zfill(5) + ext
    r = requests.get(url, stream=True)
    if r.status_code !=200:
        r.raise_for_status()
    with open(os.path.join(out_path,out_name),'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print('Download',out_name, 'done')

def get_and_download_url_wrapper(args):
    return get_and_download_url(*args)

def argument_parser():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bounds', nargs='+', type=int, default=[-84, 24, -78, 32])
    parser.add_argument('-g', '--grid_key', type=str)
    parser.add_argument('-i', '--idate',type=str, default='2022')
    parser.add_argument('-f', '--fdate',type=str, default='2023')
    parser.add_argument('-s', '--scale', type = float, default = 150.0)
    parser.add_argument('-m', '--maxims', type = int, default = 10)
    parser.add_argument('-se', '--sensor', choices=['l8', 'l9', 's2', 'mds'], type=str, default = 'l8')
    parser.add_argument('-o', '--outpath', type=str, default = 'images')
    parser.add_argument('-r', '--region', type=str, default=None)
    parser.add_argument('-e', '--format', type=str,default = 'GEOTiff', choices=['GEOTiff'])
    parser.add_argument('-sd', '--seed', type=int,default = None)
    parser.add_argument('-c', '--crs', type=str, default = None)
    parser.add_argument('-cc', '--cloud_cover_max',type=float, default = 40.0)
    parser.add_argument('-ccgt', '--cloud_cover_min', type=float, default = 0.0)
    parser.add_argument('-ba','--bands',type=str,nargs='+',default =['B4','B3','B2'])
    parser.add_argument('-cm', '--custom_mosaics', type=bool, default = False)
    parser.add_argument('-vb', '--vertical_buffer', type=float, default = 318816)
    parser.add_argument('-hb', '--horizontal_buffer', type=float, default = 425088)
    parser.add_argument('-gd', '--gdrive', type=bool, default = False)
    parser.add_argument('-np', '--nprocs', type=int, default = None)
    parser.add_argument('-rm', '--region_mosaic', type=bool, default = False)
    parser.add_argument('-rc', '--region_composite', type=bool, default = False)
    parser.add_argument('-ex_16', '--exclude_16_regions', type=bool, default = False)
    parsed_args = parser.parse_args()
    if parsed_args.region is None:
        parsed_args.region = parsed_args.grid_key
    if parsed_args.nprocs is None:
        parsed_args.nprocs = cpu_count()
    return parsed_args

def process_region_composite(args, region_rect, collection, scale, out_path, out_format, region_name):
    """
    Process and export a composite image for a specific region from Landsat data.
    
    Args:
        args: Command line arguments or configuration object.
        region_rect (ee.Geometry): The geometry of the region for which the composite is generated.
        collection (ee.ImageCollection): The initial collection of images.
        scale (float): The scale in meters for the export.
        out_path (str): The output directory path on Google Drive.
        out_format (str): The output format of the image, typically 'GEOTiff'.
        region_name (str): A descriptive name for the region to help name the output file.
    
    Returns:
        list: A list of Earth Engine tasks that can be started.
    """
    task_list = []
    region_rect = region_rect.buffer(10000).bounds()
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1').filterDate('2020','2022').filterBounds(region_rect).filter(ee.Filter.lt('CLOUD_COVER', 5))
    composite = ee.Algorithms.Landsat.simpleComposite(collection).select('B4','B3','B2')
    composite = composite.divide(0.3).toByte()
    out_name = args.sensor + '_' + region_name + '_composite'
    if not args.crs:
        crs = 'EPSG:4326'
    task_config = {
        'scale': scale,
        'fileFormat': out_format,
        'region': region_rect,
        'driveFolder': out_path,
        'crs': crs
    }
    task = ee.batch.Export.image(composite, out_name, task_config)
    task_list.append(task)
    return task_list

def process_region_mosaic(args, region_rect, collection, scale, out_path, out_format, region_name):
    """
    Create and export a mosaic image for a specific region, applying a random sort to image selection.
    
    Args:
        args: Configuration object containing parameters like maximum images.
        region_rect (ee.Geometry): The geometry of the region for which the mosaic is generated.
        collection (ee.ImageCollection): The initial collection of images.
        scale (float): The scale in meters for the export.
        out_path (str): The output directory path on Google Drive.
        out_format (str): The output format of the image.
        region_name (str): A descriptive name for the region to help name the output file.
        
    Returns:
        tuple: A tuple containing a list of tasks and a list of images.
    """
    task_list = []
    im_list = []
    max_ims = args.maxims
    for i in range(max_ims):
        collection_with_random_column = collection.randomColumn('random', np.random.randint(100000))
        collection_with_random_column = collection_with_random_column.sort('random')
        collection_with_random_column = ee.ImageCollection(collection_with_random_column)
        MULTIPLIER = 255 / 0.3
        if args.sensor == 's2':
                MULTIPLIER = MULTIPLIER*0.0001 
        im = collection_with_random_column.mosaic().multiply(MULTIPLIER).toByte()
        out_name = f"{args.sensor}_{region_name}_{str(i).zfill(5)}"
        task_config = {
            'scale': scale,
            'fileFormat': out_format,
            'region': region_rect,
            'driveFolder': out_path,
            'crs': 'EPSG:4326'
        }
        task = ee.batch.Export.image(im, out_name, task_config)
        task_list.append(task)
        im_list.append(im)
    im_list = ee.List(im_list)
    return task_list, im_list

def process_landsat_sensor(args, region_rect, collection, scale, out_path, out_format, region_name):
    """
    Processes and exports images from the Landsat sensors, ensuring they are adjusted for visibility and clipped to the region.
    
    Args:
        args: Configuration object containing parameters such as maximum images and CRS settings.
        region_rect (ee.Geometry): The geometry defining the bounds of the region.
        collection (ee.ImageCollection): The filtered collection of Landsat images.
        scale (float): The scale in meters for the image export.
        out_path (str): The directory path where the images will be saved on Google Drive.
        out_format (str): The format for the output images, typically 'GEOTiff'.
        region_name (str): The name of the region, used for naming the output files.
        
    Returns:
        tuple: A tuple containing a list of export tasks and a list of processed images.
    """
    task_list = []
    max_ims = args.maxims
    collection = collection.filterBounds(region_rect)
    collection_size = collection.size().getInfo()
    if collection_size < max_ims:
        max_ims = collection_size
    im_list = collection.toList(max_ims)

    for i in range(max_ims):
        im = ee.Image(im_list.get(i))
        crs = im.select(0).projection().crs().getInfo() if not args.crs else args.crs
        im = im.multiply(255 / 0.3).toByte()
        im = im.clip(im.geometry())
        out_name = f"{args.sensor}_{region_name}_{str(i).zfill(5)}"
        task_config = {
            'scale': scale,
            'fileFormat': out_format,
            'crs': crs,
            'driveFolder': out_path
        }
        task = ee.batch.Export.image.toDrive(im, out_name, **task_config)
        task_list.append(task)

    return task_list, im_list

def process_sentinel_sensor(args, proj, region_rect, collection, scale, out_path, out_format, region_name):
    """
    Processes and exports images from the Sentinel sensor, focusing on specific points within a region to create mosaics.
    
    Args:
        args: Configuration object containing parameters such as maximum images.
        proj (str): The projection code used for the images.
        region_rect (ee.Geometry): The geometry defining the bounds of the region.
        collection (ee.ImageCollection): The filtered collection of Sentinel images.
        scale (float): The scale in meters for the image export.
        out_path (str): The directory path where the images will be saved on Google Drive.
        out_format (str): The format for the output images.
        region_name (str): The name of the region, used for naming the output files.
        
    Returns:
        tuple: A tuple containing a list of export tasks and a list of processed images.
    """
    im_list = []
    task_list = []
    max_ims = args.maxims
    points = get_points_in_region(region_rect, max_ims, scale, np.random.randint(100000))
    for i, point in enumerate(points):
        clip_rect = make_rectangle(point, 185000 / 2)
        collection_with_random_column = collection.filterBounds(clip_rect).randomColumn('random', np.random.randint(100000))
        collection_with_random_column = collection_with_random_column.sort('random')
        collection_with_random_column = ee.ImageCollection(collection_with_random_column)
        MULTIPLIER = 255/0.3
        if args.sensor == 's2':
            MULTIPLIER = MULTIPLIER*0.0001  
        im = collection_with_random_column.mosaic().multiply(0.0001).divide(0.3).multiply(255).toByte()
        rect_im = im.clip(clip_rect)
        im_list.append(rect_im)

        out_name = f"{args.sensor}_{region_name}_{str(i).zfill(5)}"
        task_config = {
            'scale': scale,
            'fileFormat': out_format,
            'region': clip_rect,
            'driveFolder': out_path,
            'crs': proj
        }
        task = ee.batch.Export.image(rect_im, out_name, task_config)
        task_list.append(task)
    im_list = ee.List(im_list)

    return task_list, im_list

def process_custom_mosaics(args, proj, region_rect, collection, scale, out_path, out_format, region_name):
    """
    Creates custom mosaics by selecting random points within a region and creating a detailed mosaic around each point.
    
    Args:
        args: Configuration object with parameters for processing.
        proj (str): The projection code used for image processing.
        region_rect (ee.Geometry): The geometry defining the bounds of the region.
        collection (ee.ImageCollection): The collection of images to process.
        scale (float): The scale in meters for the image export.
        out_path (str): The directory path where the images will be saved on Google Drive.
        out_format (str): The format for the output images.
        region_name (str): The name of the region, used for naming the output files.
        
    Returns:
        tuple: A tuple containing a list of export tasks and a list of processed images.
    """
    im_list = []
    task_list = []
    points = get_points_in_region(region_rect, args.maxims, scale, np.random.randint(100000))
    
    for i, point in enumerate(points):
        # Create custom rectangle around point and filter collection
        h_buffer = args.horizontal_buffer * 1.25
        v_buffer = args.vertical_buffer * 1.25

        bigger_clip_rect = make_rectangle(point, h_buffer, v_buffer)
        clip_rect = make_rectangle(point, args.horizontal_buffer, args.vertical_buffer)
        collection_with_random_column = collection.filterBounds(bigger_clip_rect)
        collection_with_random_column = collection_with_random_column.randomColumn('random',np.random.randint(100000))
        collection_with_random_column = collection_with_random_column.sort('random')
        collection_with_random_column = ee.ImageCollection(collection_with_random_column)
        if collection_with_random_column.size().getInfo() == 0:
            print('Empty collection, skipping to next iteration')
            continue
        print(f"collection size: {collection_with_random_column.size().getInfo()}")
        MULTIPLIER = 255 / 0.3
        if args.sensor == 's2':
            MULTIPLIER = MULTIPLIER*0.0001 
        im = collection_with_random_column.mosaic().multiply(MULTIPLIER).toByte()
        rect_im = im.clip(bigger_clip_rect)
        im_list.append(rect_im)
        out_name = f"{args.sensor}_{region_name}_{str(i).zfill(5)}"
        task_config = {
            'scale': scale,
            'fileFormat': out_format,
            'region': clip_rect,
            'driveFolder': out_path,
            'crs': proj
        }
        task = ee.batch.Export.image(rect_im, out_name, task_config)
        task_list.append(task)
    im_list = ee.List(im_list)

    return task_list, im_list

def process_exclude_16_regions(args, proj, scale, out_path, out_format):
    task_list = []
    maxtask = args.maxims
    excluded_regions = {'10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', '32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T'}
    grid = getMGRS()
    # Filter out the excluded regions
    valid_regions = {key: value for key, value in grid.items() if key not in excluded_regions}
    print(f"Excluding 16 regions. Total valid region number: {len(valid_regions)}")

    # Iterate over valid regions and download images
    for grid_key, bounds in valid_regions.items():
        args.grid_key = grid_key
        args.maxims = 5
        args.bounds = [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
        
        # Remaining script logic to download images...
        print(f"Processing region {grid_key} | Current tasks: {len(task_list)}")
        
        # Getting region filter and rectangle from bounds
        region_filter, region_rect = get_region_filter_from_bounds(args.bounds, get_rect=True)
        date_filter = get_date_filter(args.idate, args.fdate) 
        
        # Process images as previously defined in the script
        collection = get_collection(args.sensor, region_filter, date_filter, ee_bands=bands, cloud_cover_min = args.cloud_cover_min, cloud_cover_max=args.cloud_cover_max, date_sort=True)
        tasks, _ = process_custom_mosaics(args, proj, region_rect, collection, scale, out_path, out_format, grid_key)
        if len(tasks) > 0:
            start_tasks_to_drive(tasks)
            #start_tasks_to_local(args, 5, im_list, region_name)

        task_list.extend(tasks)
        if len(task_list) > maxtask:
            return task_list
         
    return task_list

def start_tasks_to_drive(task_list):
    """
    Starts a list of export tasks to Google Drive, commonly used for batch processing of image exports.
    
    Args:
        task_list (list): A list of Earth Engine tasks to be started.
    """
    print('Downloading images to Google Drive.')
    print('View status of tasks at: https://code.earthengine.google.com/tasks')
    for task in task_list:
        task.start()
    print(len(task_list), 'tasks started')

def start_tasks_to_local(args, max_ims, im_list, region_name):
    """
    Starts tasks for downloading images locally using multiprocessing, providing a wrapper to handle image downloads efficiently.
    
    Args:
        args: Configuration object with necessary parameters like output path and format.
        max_ims (int): Maximum number of images to download.
        im_list (list): A list of image objects to download.
        region_name (str): The name of the region, used for naming the output files.
    """
    # Create a list of arguments to pass to each worker process
    process_args = [(i, im_list, args.outpath, args.format, args.sensor, region_name) for i in range(max_ims)]
    print('Downloading images.')
    # Use process_map to process images with progress tracking
    process_map(get_and_download_url_wrapper, process_args, max_workers=args.nprocs, chunksize=1)


if __name__ == '__main__':
    # Get command line arguments
    args = argument_parser()

    # Assigning parsed arguments to variables
    scale = args.scale
    max_ims = args.maxims
    out_path = args.outpath
    out_format = args.format
    region_name = args.region
    bands = args.bands
    sensor = args.sensor

    # Adjusting bounds if grid key is provided
    if args.grid_key:
        grid = getMGRS()
        left, bottom, right, top = grid[args.grid_key]
        args.bounds = [float(left), float(bottom), float(right), float(top)]

    if args.region is None:
        args.region = args.grid_key

    # Setting seed for random number generation
    if args.seed:
        seed = args.seed
    else:
        seed = np.random.randint(100000)

    if args.grid_key is None:
        proj = "EPSG:4326"
    elif args.grid_key[-1] <= 'M':
        proj = "EPSG:327" + args.grid_key[:-1]
    else:
        proj = "EPSG:326" + args.grid_key[:-1]

    # Getting region filter and rectangle from bounds
    region_filter, region_rect = get_region_filter_from_bounds(args.bounds, get_rect=True)
    date_filter = get_date_filter(args.idate, args.fdate) # Getting date filter based on input dates

    # Processing image collection based on selected options
    collection = get_collection(args.sensor, region_filter, date_filter, 
                                ee_bands=bands, cloud_cover_min = args.cloud_cover_min,
                                cloud_cover_max=args.cloud_cover_max, date_sort=True)  

    if not args.custom_mosaics:
        # Non Custom Mosaics Processing task
        if args.region_composite:
            task_list = process_region_composite(args, region_rect, collection, scale, out_path, out_format, region_name)
        elif args.region_mosaic:
            task_list, im_list = process_region_mosaic(args, region_rect, collection, scale, out_path, out_format, region_name)
        elif args.sensor in ('l8', 'l9', 'mds'):
            task_list, im_list = process_landsat_sensor(args, region_rect, collection, scale, out_path, out_format, region_name)
        elif args.sensor == 's2':
            task_list, im_list = process_sentinel_sensor(args, proj, region_rect, collection, scale, out_path, out_format, region_name)
        
        # Download mode
        if args.gdrive:
            start_tasks_to_drive(task_list)
        else:
            start_tasks_to_local(args, max_ims, im_list, region_name)
    else:
        # Custom mosaics with 16 regions excluded
        if args.exclude_16_regions:
            process_exclude_16_regions(args, proj, scale, out_path, out_format)
        else:
            # Custom mosaics processing & download to drive
            task_list, _ = process_custom_mosaics(args, proj, region_rect, collection, scale, out_path, out_format, region_name)
            start_tasks_to_drive(task_list)
            
        
        
