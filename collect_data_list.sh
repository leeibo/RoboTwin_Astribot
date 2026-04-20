#!/bin/bash


task_config=${1}
gpu_id=0
task_list=(
    place_a2b_right_rotate_view
    place_burger_fries_rotate_view
    place_cans_plasticbox_rotate_view
    place_object_scale_rotate_view
    place_object_stand_rotate_view
    place_shoe_rotate_view
    stack_blocks_three_rotate_view
)
./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

for task_name in "${task_list[@]}"; do
PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
    collect_status=$?
    rm -rf data/${task_name}/${task_config}*/.cache
    exit ${collect_status}
done
