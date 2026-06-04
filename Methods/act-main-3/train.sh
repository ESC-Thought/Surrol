#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/escthought/Surrol

# Get the current date and time
task_name=${1:-bi_peg_transfer}
datetime=$(date +"%Y%m%d_%H%M")_${task_name}
# datetime="20250215_2011_bi_peg_transfer"

output_root="/home/escthought/Surrol/Methods/act-main-3/experiments/"
# output_root="/home/skylar/SurRoL/IROS_SurRoL/rl/act-main-3/experiments/"
# Create an output folder using the datetimed

# output_folder="/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3/experiments20250212_2258_bi_peg_transfer"

# Run the Python script with the newly created output folder
# command="python3 imitate_episodes_copy.py \

command="python3 imitate_episodes_.py \
--task_name \"$task_name\" \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 5 \
--hidden_dim 512 \
--batch_size 8 \
--dim_feedforward 3200 \
--num_epochs 3000 \
--lr 1e-5 \
--seed 0 "

# command="python3 imitate_episodes_.py \
# --task_name \"$task_name\" \
# --policy_class ACT \
# --kl_weight 10 \
# --chunk_size 20 \
# --hidden_dim 512 \
# --batch_size 8 \
# --dim_feedforward 3200 \
# --num_epochs 4000 \
# --lr 1e-5 \
# --seed 50 "


if [[ "$2" == "eval" ]]; then
    command+=" --eval"    
    output_folder="${output_root}20260603_0151_bi_peg_transfer_200_rgb1_top"
else
    output_folder="${output_root}${datetime}_200_rgb1_top"
    mkdir -p "$output_folder"
fi

command+=" --ckpt_dir \"$output_folder\" "

# Check if "eval" is passed as an argument

# Execute the Python command
eval $command
