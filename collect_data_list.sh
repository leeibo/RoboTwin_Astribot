#!/bin/bash



gpu_id=0
task_list1=(
blocks_ranking_rgb_fan_double
blocks_ranking_size_fan_double
place_object_basket_fan_double
put_block_on_upper_easy
put_block_on_upper_hard
)
task_list2=(
# beat_block_hammer_rotate_view
# blocks_ranking_rgb_rotate_view
# blocks_ranking_size_rotate_view
# click_bell_rotate_view
# move_pillbottle_pad_rotate_view
# move_stapler_pad_rotate_view
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
# stack_blocks_two_rotate_view
# stamp_seal_rotate_view
# turn_switch_rotate_view

)
task_list3=(
    beat_block_hammer_rotate_view
    blocks_ranking_rgb_rotate_view
    blocks_ranking_size_rotate_view
    click_bell_rotate_view
    move_pillbottle_pad_rotate_view
    move_stapler_pad_rotate_view
    place_a2b_left_rotate_view
    place_a2b_right_rotate_view
    place_cans_plasticbox_rotate_view
    place_container_plate_rotate_view
    place_empty_cup_rotate_view
    place_fan_rotate_view
    place_mouse_pad_rotate_view
    place_object_scale_rotate_view
    place_object_stand_rotate_view
    place_shoe_rotate_view
    press_stapler_rotate_view
    shake_bottle_horizontally_rotate_view
    shake_bottle_rotate_view
    stack_blocks_two_rotate_view
    stamp_seal_rotate_view
    turn_switch_rotate_view
    blocks_ranking_rgb_fan_double
    blocks_ranking_size_fan_double
    place_object_basket_fan_double
    put_block_on_upper_easy
    put_block_on_upper_hard
    count_target_press_button
    count_random_object_press_button
    count_color_kinds_press_button
    check_block_color
    check_cola_date
    check_cola_color
    rank_backside_rgb_blocks
    match_backside_two_blocks
)
task_list4=(
    # Single-layer rotate-view tasks.
    beat_block_hammer_rotate_view
    blocks_ranking_rgb_rotate_view
    blocks_ranking_size_rotate_view
    click_bell_rotate_view
    move_pillbottle_pad_rotate_view
    move_stapler_pad_rotate_view
    place_a2b_left_rotate_view
    place_a2b_right_rotate_view
    place_cans_plasticbox_rotate_view
    place_container_plate_rotate_view
    place_empty_cup_rotate_view
    place_fan_rotate_view
    place_mouse_pad_rotate_view
    place_object_scale_rotate_view
    place_object_stand_rotate_view
    place_shoe_rotate_view
    press_stapler_rotate_view
    shake_bottle_horizontally_rotate_view
    shake_bottle_rotate_view
    stack_blocks_two_rotate_view
    stamp_seal_rotate_view
    turn_switch_rotate_view

    # Double / Fan Double tasks.
    blocks_ranking_rgb_fan_double
    blocks_ranking_size_fan_double
    place_object_basket_fan_double
    put_block_on_upper_easy
    put_block_on_upper_hard

    # Info Gather tasks.
    count_target_press_button
    count_random_object_press_button
    count_color_kinds_press_button
    check_block_color
    check_cola_date
    check_cola_color
    rank_backside_rgb_blocks
    match_backside_two_blocks
)
task_list5=(
    blocks_ranking_size_fan_double
    blocks_ranking_rgb_fan_double

)
task_configs=(
    # info_gathering_randomized
    info_gathering_demo
)
./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

for task_name in "${task_list1[@]}"; do
    for task_config in "${task_configs[@]}"; do
        echo "Collecting data for $task_name"
        PYTHONWARNINGS=ignore::UserWarning \
        python script/collect_data.py $task_name $task_config
        collect_status=$?
        rm -rf data/${task_name}/${task_config}*/.cache
    done
done
