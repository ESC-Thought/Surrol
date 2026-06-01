#! /bin/bash

# Get the current date and time
task_name=${1:-bi_peg_transfer}
# datetime=$(date +"%Y%m%d_%H%M")
datetime=$(date +"%Y%m%d_%H%M")_${task_name}
# datetime="20250215_2011_bi_peg_transfer"

output_root="/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3/experiments/ours/"
# output_root="/home/skylar/SurRoL/IROS_SurRoL/rl/act-main-3/experiments/"
# Create an output folder using the datetime
output_folder="${output_root}${datetime}_stereo_200_ours"
mkdir -p "$output_folder"

# output_folder="/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3/experiments/whole_policy/20250311_2324_bi_peg_transfer"

# Run the Python script with the newly created output folder
# command="python3 imitate_episodes_copy.py \
command="python3 imitate_episodes_ours.py \
--task_name \"$task_name\" \
--ckpt_dir \"$output_folder\" \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 5 \
--hidden_dim 512 \
--batch_size 8 \
--dim_feedforward 3200 \
--num_epochs 3000 \
--lr 1e-5 \
--seed 0"


# Check if "eval" is passed as an argument
if [[ "$2" == "eval" ]]; then
    command+=" --eval"
fi

# Execute the Python command
eval $command