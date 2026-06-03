# Information-Gathering Task Suite Plan

This branch adds experimental tasks where the robot must actively acquire
state before deciding what to do.  The suite is intentionally separate from the
legacy DYJ 34-task whitelist; tracked demo config is `task_config/info_gathering_demo.yml`
and tracked experimental whitelist is `task_config/info_task_whitelist.yml`.

## External asset reference

RMBench implements a pressable button as a small URDF articulation with a
`button_cap` link and a prismatic `button_joint`.  The original task creates it
with:

```python
rand_create_sapien_urdf_obj(
    modelname="005_button",
    modelid=10124,
    fix_root_link=True,
)
```

This repo now has `script/download_rmbench_info_assets.py` to download the
minimal RMBench assets used by this task family:

- `objects/005_button/10124`
- `objects/006_check_button/10124`
- `objects/004_numbercard`

## Task 1: count target objects, then press a button count times

Initial implementation: `envs/count_target_press_button.py`.

Requirements:

- Scene contains 1-4 green target blocks plus distractor blocks.
- Robot must scan the table before acting.
- Robot presses the RMBench red button exactly `x` times, where `x` is the
  number of green target blocks.
- Success is the physical/semantic press counter matching the target count.

Acceptance demo:

```bash
ROBOTWIN_SKIP_RENDER_TEST=1 \
ROBOTWIN_SKIP_ANNOTATED_VIDEO=1 \
ROBOTWIN_SKIP_INSTRUCTIONS=1 \
ROBOTWIN_START_SEED=0 \
ROBOTWIN_MAX_SEED_TRIES=80 \
CUDA_VISIBLE_DEVICES=0 \
PYTHONWARNINGS=ignore::UserWarning \
/home/admin1/yibo/conda/envs/robotwin/bin/python \
  script/collect_data.py count_target_press_button info_gathering_demo
```

## Task 2: count target objects and collect exactly that many

Planned task name: `count_target_collect_container`.

Requirements:

- Scene contains a mixed set of target and distractor objects plus a container.
- The robot first scans/counts target objects.
- It must place exactly the target objects into the container.
- Success checks all targets are in/contacting the container and distractors are
  left outside.

## Task 3: track a moved object out of view

Planned task name: `track_moved_object_direction`.

Requirements:

- A target object starts visible, then is moved out of the current camera view
  along a known left/right direction by scene dynamics or scripted motion.
- The robot must use the observed motion direction to search the correct side
  first and reacquire the object.
- Success checks the object is found and optionally picked/lifted.

## Task 4: open a container and identify contents

Planned task name: `open_container_identify_object`.

Requirements:

- A cabinet/box hides one object sampled from a small object set.
- Initial search should not reveal the object.
- The robot opens the container and focuses on the revealed object.
- Success checks the container is open and the inside object identity is known.

Implementation note: `envs/search_object.py` already contains a robust cabinet
open-then-find flow and should be reused instead of duplicating drawer logic.

## Task 5: action-acquired underside information

Planned task name: `inspect_underside_sort_block`.

Requirements:

- A custom block has an outer color and a different inset color on its bottom.
- The bottom inset is not visible from initial head-only scanning.
- The robot must manipulate the block to expose/inspect the underside, infer the
  inset color, and sort it to the matching region.
- Success checks the block is in the region corresponding to the underside
  color, not the outer color.

Implementation note: this requires either a small custom compound SAPIEN asset
or a controlled block-flip/lift motion with a bottom marker attached.
