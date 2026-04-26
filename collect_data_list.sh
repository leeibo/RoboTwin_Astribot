#!/bin/bash


task_config=${1}
gpu_id=0
task_list1=(
    place_a2b_right_rotate_view
    place_cans_plasticbox_rotate_view
    place_object_scale_rotate_view
    place_object_stand_rotate_view
    place_shoe_rotate_view
    stack_blocks_three_rotate_view
)
task_list2=(
# beat_block_hammer_rotate_view
# blocks_ranking_rgb_rotate_view
# blocks_ranking_size_rotate_view
# click_alarmclock_rotate_view
# click_bell_rotate_view
# move_pillbottle_pad_rotate_view
# move_stapler_pad_rotate_view
# open_laptop_rotate_view
# place_a2b_left_rotate_view
# place_a2b_right_rotate_view
# place_cans_plasticbox_rotate_view
# place_container_plate_rotate_view
# place_empty_cup_rotate_view
# place_fan_rotate_view
# place_mouse_pad_rotate_view
# place_object_scale_rotate_view
# place_object_stand_rotate_view
# place_shoe_rotate_view
# press_stapler_rotate_view
# shake_bottle_horizontally_rotate_view
# shake_bottle_rotate_view
# stack_blocks_three_rotate_view
# stack_blocks_two_rotate_view
# stamp_seal_rotate_view
# turn_switch_rotate_view

)
./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

for task_name in "${task_list2[@]}"; do
    echo "Collecting data for $task_name"
    PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py $task_name $task_config
    collect_status=$?
    rm -rf data/${task_name}/${task_config}*/.cache
done
