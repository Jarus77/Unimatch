#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")

save_path="C:/Users/suraj/OneDrive/sprasad2/unimatch_from_scratch/save_path"

mkdir -p $save_path

export CUDA_VISIBLE_DEVICES=2
export TUNE_DISABLE_STRICT_METRIC_CHECKING=1

python ray_tune.py \
    --project_name=ss2-ssl-idx-12 \
    --model_name=debug-fixmatch-wo-cutmix-2 \
    \
    --dataset=idx_12 \
    --nclass=3 \
    \
    --num_samples=5 \
    --num_epochs=10 \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log
    
    # --use_checkpoint \
    # --enable_logging \