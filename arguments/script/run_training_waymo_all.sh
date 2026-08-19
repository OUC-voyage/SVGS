#!/bin/bash

datasets=(
  "segment-131967"

)

# 第一部分：运行第四个bash
for dataset in "${datasets[@]}"; do
  echo "Processing $dataset (First Script)..."
  python train_copy.py -s "./data2/waymo/$dataset" -m "./SVGS/waymo/$dataset" --eval --flatten_loss --position_lr_init 0 --position_lr_final 0 --scaling_lr 0.001 --percent_dense 0.0005 --dataset waymo --sky_seg --normal_loss --depth_loss --propagation_interval 30 --depth_error_min_threshold 0.8 --depth_error_max_threshold 1.0 --propagated_iteration_begin 1000 --propagated_iteration_after 12000 --patch_size 20 --lambda_l1_normal 0.001 --lambda_cos_normal 0.001 --gpu 1 --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1 --port 6009
done



