#!/bin/bash

# Define the list of regions
# regions=('10S' '10T' '11R' '12R' '16T' '17R' '17T' '18S' '32S' '32T' '33S' '33T' '53S' '52S' '54S' '54T')
# regions=('10S' '16T' '17R' '54T')
regions=('16T' '17R' '17T' '18S' '32S' '33S' '33T' '53S' '52S' '54S' '54T')

# Loop through each region
for region in "${regions[@]}"; do
    # Train
    echo "Training dataset for region $region"
    python prepare_rc_data.py --data_path New_RC_Data --landmark_path landmark_csvs --region "$region" --output_path /home/argus-vision/vision/VisionTrainingGround/RCnet/16_regions_dataset

    echo "Completed processing for region $region"
done

echo "All regions processed."