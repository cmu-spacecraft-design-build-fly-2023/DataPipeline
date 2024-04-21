#!/bin/bash

# Define the list of regions
regions=("10S" "10T" "11R" "16T" "17R" "12R" "18S"  "32S" "32T" "33S" "33T"   "52S" "53S" "54S" "54T")

# Loop through each region
for region in "${regions[@]}"
do
    echo "Processing region $region..."
    python3 prepare_rc_data.py \
        --data_path "Raw_Data/${region}_L8_250/" \
        --landmark_path "landmark_csvs/" \
        --region "$region" \
        --resolution "640x640" \
        --output_path "/home/argus-vision/vision/VisionTrainingGround/RCnet/datasets/${region}" \
    	--test "True"
done

echo "All regions processed."

