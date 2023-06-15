# Example pipeline to reproduce CTRL vehicle results. Only for reference.

# Generate train_gt.bin once for all. (waymo bin format).
python ./tools/tracklet_learning/generate_train_gt_bin.py

#********************* Training *********************

# Step 1: Use ImmortalTrack to generate tracking results in training split (bin file format)
# ........ (require external code)

# Step 2: Generate track input for training (bin file path and split need to be specified in fsd6f6e_vehicle_full.yaml)
python ./tools/tracklet_learning/generate_track_input.py ./tools/tracklet_learning/data_configs/fsd6f6e_vehicle_full.yaml --process 8

# Step 3: Assign candidates GT tracks
python ./tools/tracklet_learning/gt_tracklet_candidate_select.py ./tools/tracklet_learning/data_configs/fsd6f6e_vehicle_full.yaml --process 8 

# Step 4: Begin training (need --no-validation arguement)
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --no-validate


#********************* Inference *********************

# Step 1: Use ImmortalTrack to generate tracking results in bin file format
# ........ (require external code)

# Step 2 (Optional): Backtracing and Extension, specific information are specified in the extend_veh_01.yaml
python ./tools/tracklet_learning/extend_tracks.py ./tools/tracklet_learning/data_configs/extend_veh_01.yaml 

# Step 3: Generate track input for inference (bin file path and split need to be specified in fsd6f6e_vehicle_full.yaml)
python ./tools/tracklet_learning/generate_track_input.py ./tools/tracklet_learning/data_configs/fsd6f6e_vehicle_full.yaml --process 8

# Step 4: Begin inference (Track TTA is optional, can be enabled in config)
bash ./tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/latest.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/result"  --eval waymo

# Step 5 (Optional): Remove empty predictions
python ./tools/tracklet_learning/remove_empty.py --bin-path ./$WORK/$CONFIG/result.bin --process 8 --split training --type vehicle