# TCP Extrinsics Calibration (Astribot)

This workflow calibrates:
- `gripper_bias`
- `delta_matrix`
- `global_trans_matrix`

for RoboTwin's transform chain:

`R_tcp = R_ee * global_trans_matrix * delta_matrix`

## 1) Set a stable baseline

Edit:
- `assets/embodiments/astribot_descriptions/config.yml`

Suggested start:
- `joint_stiffness: 20000`
- `joint_damping: 200`
- `gripper_bias: 0.19` (your measured value)
- `delta_matrix: I`
- `global_trans_matrix: I`

## 2) Find reachable calibration poses (IK sweep)

Left arm:

```bash
/home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/ik_sweep_curobo.py \
  --curobo-yml assets/embodiments/astribot_descriptions/curobo_left.yml \
  --quat 1,0,0,0 \
  --out script/calibration/ik_left_identity.json
```

Right arm:

```bash
/home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/ik_sweep_curobo.py \
  --curobo-yml assets/embodiments/astribot_descriptions/curobo_right.yml \
  --quat 1,0,0,0 \
  --out script/calibration/ik_right_identity.json
```

Pick 8-12 successful points per arm for data collection.

## 3) Collect debug logs

Run your test and save output:

```bash
PATH=/home/admin1/yibo/conda/envs/robotwin/bin:$PATH \
bash collect_data.sh beat_block_hammer demo_clean 0 | tee script/calibration/calib_run.log
```

## 4) Extract paired samples from logs

```bash
/home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/extract_pose_samples.py \
  --log script/calibration/calib_run.log \
  --source take_dense_action \
  --out script/calibration/samples_all.json
```

Per-arm extraction:

```bash
/home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/extract_pose_samples.py \
  --log script/calibration/calib_run.log \
  --source take_dense_action \
  --arm left \
  --out script/calibration/samples_left.json
```

```bash
/home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/extract_pose_samples.py \
  --log script/calibration/calib_run.log \
  --source take_dense_action \
  --arm right \
  --out script/calibration/samples_right.json
```

## 5) Solve matrices

Left:

```bash
/home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/solve_tcp_extrinsics.py \
  --samples script/calibration/samples_left.json \
  --out script/calibration/fit_left.json
```

Right:

```bash
/home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/solve_tcp_extrinsics.py \
  --samples script/calibration/samples_right.json \
  --out script/calibration/fit_right.json
```

Use the printed `delta_matrix` and `global_trans_matrix` in config.

## 6) Validate

Re-run:

```bash
PATH=/home/admin1/yibo/conda/envs/robotwin/bin:$PATH \
bash collect_data.sh beat_block_hammer demo_clean 0
```

Check:
- `FK-END world_err` near 0
- `ENDLINK_DEBUG` position error near millimeter level
- `TCP_DEBUG` rotation error reduced significantly

If TCP rotation is still biased by a nearly fixed angle:
- keep `delta_matrix`, re-fit `global_trans_matrix` with more samples.

If TCP position still has z/y residual while x is good:
- current model only supports 1D offset (`gripper_bias` along tool x).
- you may need to extend code to a full 3D translation offset.

## 7) Live reference-frame logging (for delta calibration)

Runtime now writes JSONL snapshots after each action:
- log file: `script/calibration/live_frame_records.jsonl`
- each record contains:
  - `live_world_q`, `R_world_live`
  - `reference_world_q`, `R_world_ref`
  - `R_ref_live = R_world_ref.T @ R_world_live`

Reference frame convention used:
- `x_ref = +y_world`
- `y_ref = -x_world`
- `z_ref = x_ref x y_ref`
- visualized at world position `[0, 0, 1]`

Watch records in real time:

```bash
/home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/watch_live_frame_log.py \
  --log script/calibration/live_frame_records.jsonl
```
