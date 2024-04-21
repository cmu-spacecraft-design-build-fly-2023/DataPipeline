import csv
import os
import argparse
import glob

def get_all_landmarks(input_csv):
    landmarks_by_scale = {}
    # read input csv containing all landmarks
    with open(input_csv, encoding='utf-8') as csv_file:
        csvReader = csv.DictReader(csv_file)
        total_landmarks = 0
        for row in csvReader:
            scale = row["scale"]
            # group landmarks by scale
            if scale not in landmarks_by_scale.keys():
                landmarks_by_scale.update({scale:[row]})
            else:
                landmarks_by_scale[scale].append(row)
            total_landmarks +=1
    
    print("total landmarks", total_landmarks)
    
    return landmarks_by_scale, total_landmarks


def get_salient_landmarks_by_scale(num_landmarks, landmarks_by_scale, total_landmarks):
    salient_landmarks = []
    num_scales = len(landmarks_by_scale.keys())
    scale_percent = num_landmarks / total_landmarks
    print("percent per scale", scale_percent)

    # get top x% salient landmarks at each scale
    for scale in landmarks_by_scale.keys():
        scale_total = len(landmarks_by_scale[scale])
        num_per_scale = int(scale_percent*scale_total)
        salient_rows = landmarks_by_scale[scale][:num_per_scale]
        salient_landmarks += salient_rows

    #print(salient_landmarks)
    print("total salient landmarks", len(salient_landmarks))
    return salient_landmarks


def save_salient_landmarks(salient_landmarks, output_csv):
    with open(output_csv, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["Centroid Longitude", "Centroid Latitude", "Top-Left Longitude", "Top-Left Latitude", "Bottom-Right Longitude", "Bottom-Right Latitude"])
        for row in salient_landmarks:
            cent_lon = row["x_center_lon"]
            cent_lat = row["y_center_lat"]
            tl_lon = row["x_min_lon"]	
            tl_lat = row["y_max_lat"]
            br_lon = row["x_max_lon"]
            br_lat = row["y_min_lat"]
            writer.writerow([cent_lon, cent_lat, tl_lon, tl_lat, br_lon, br_lat])

def get_files(args):
    for file in glob.glob(args.input_path + '/*_landmarksf30_opts.csv'):
        region_name = os.path.basename(file).split('_')[0]
        print(region_name)
        output_csv = region_name + '_top_salient.csv'
        output_path = os.path.join(args.input_path, output_csv)
        landmarks_by_scale, total_landmarks = get_all_landmarks(file)
        salient_landmarks = get_salient_landmarks_by_scale(args.num_landmarks, landmarks_by_scale, total_landmarks)
        save_salient_landmarks(salient_landmarks, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates json scene file with relationships")
    parser.add_argument("--input_path", required=True, help="input path containing original landmark csvs")
    #parser.add_argument("--input_csv", required=True, help="input csv with all bboxes and saliency")
    #parser.add_argument("--output_csv", required=True, help="output landmark csv to generate")
    parser.add_argument("--num_landmarks", default=500, help="total number of landmarks to use")
    args = parser.parse_args()

    get_files(args)

    #landmarks_by_scale, total_landmarks = get_all_landmarks(args.input_csv)
    #salient_landmarks = get_salient_landmarks_by_scale(args.num_landmarks, landmarks_by_scale, total_landmarks)
    #save_salient_landmarks(salient_landmarks, args.output_csv)