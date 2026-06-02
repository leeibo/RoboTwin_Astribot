# Clean-table cuRobo workspace sweep

This note records a clean-table reachability test for the Astribot dual-arm
embodiment with no task objects and no clutter.  The goal is to identify where
cuRobo planning succeeds most reliably before adding task-specific object
geometry, grasps, and physics.

## Setup

- Config: `demo_clean`
- Table: fan table, `fan_angle_deg=150`, `fan_inner_radius=0.3`,
  `fan_outer_radius=0.9`, deterministic clean table.
- Environment: `script/explore_clean_table_curobo_workspace.py` creates a
  minimal task with only table + robot + cameras; `load_actors()` intentionally
  adds no objects.
- Coordinates: robot-centered cylindrical coordinates.
  - `r`: radial distance from robot root.
  - `theta_deg`: angle around robot root.
  - `z_offset`: TCP target height above table top (`table_top_z=0.74` for
    `demo_clean`; e.g. `z_offset=0.12` means TCP target `z≈0.86`).
- Test signal: direct cuRobo `left_plan_path` / `right_plan_path` success from
  homestate.  No action execution or task `check_success()` is involved.

## Command

Coarse sweep:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS=ignore::UserWarning
/home/admin1/yibo/conda/envs/robotwin/bin/python script/explore_clean_table_curobo_workspace.py \
  --config=demo_clean \
  --arms=left,right \
  --orientation-modes=home,topdown_y_radial,topdown_y_inward,topdown_y_tangent,topdown_y_side,horizontal_inward,horizontal_outward,horizontal_side \
  --r-values=0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
  --theta-deg-values=-70,-50,-30,-10,0,10,30,50,70 \
  --z-offsets=0.04,0.08,0.12,0.16,0.22 \
  --progress-every=500 \
  --top-k=80 \
  > logs/clean_table_curobo_workspace_coarse.log 2>&1
```

Refined sweep around the best coarse region:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS=ignore::UserWarning
/home/admin1/yibo/conda/envs/robotwin/bin/python script/explore_clean_table_curobo_workspace.py \
  --config=demo_clean \
  --arms=left,right \
  --orientation-modes=home,horizontal_outward,topdown_y_radial,topdown_y_inward \
  --r-values=0.30,0.33,0.36,0.39,0.42,0.45,0.48 \
  --theta-deg-values=-30,-20,-10,0,10,20,30 \
  --z-offsets=0.08,0.12,0.16,0.20 \
  --progress-every=300 \
  --top-k=100 \
  > logs/clean_table_curobo_workspace_refine.log 2>&1
```

## Orientation modes

The script samples TCP target quaternions.  The most useful modes are:

- `horizontal_outward`: local `+x` points radially outward from the robot and
  local `+z` is approximately world up.
- `topdown_y_radial`: local `+x` points downward; local `+y` follows radial
  direction.
- `topdown_y_inward`: local `+x` points downward; local `+y` points inward.
- `home`: keep each arm's homestate TCP orientation and only move xyz.

The sampled convention matches RoboTwin actions where a positive
`move_by_displacement(..., move_axis="arm", z=...)` lift corresponds to motion
along local `-x` for top-down poses.

## Coarse sweep results

Coarse sweep total: `2274 / 5760 = 39.48%` success.

By orientation:

| Orientation | Success / Total | Rate |
| --- | ---: | ---: |
| `home` | 401 / 720 | 55.69% |
| `horizontal_outward` | 362 / 720 | 50.28% |
| `topdown_y_radial` | 337 / 720 | 46.81% |
| `topdown_y_inward` | 335 / 720 | 46.53% |
| `topdown_y_tangent` | 304 / 720 | 42.22% |
| `horizontal_side` | 283 / 720 | 39.31% |
| `topdown_y_side` | 252 / 720 | 35.00% |
| `horizontal_inward` | 0 / 720 | 0.00% |

By position:

| Axis/bin | Best bins from coarse sweep |
| --- | --- |
| `r` | `0.30–0.45` is strongest; drops notably after `r=0.50`; `r=0.65` only 13.33%. |
| `theta_deg` | center is strongest: `-10°` 46.25%, `0°` 45.47%, `10°` 45.16%; outer angles degrade. |
| `z_offset` | `0.12–0.16` strongest; `0.04` is poor (10.76%). |
| `r,theta` | best coarse bins cluster at `r=0.30–0.45`, `theta=-10°..10°`. |

## Refined sweep results

Refined sweep total: `1430 / 1568 = 91.20%` success.

By orientation:

| Orientation | Success / Total | Rate |
| --- | ---: | ---: |
| `horizontal_outward` | 378 / 392 | 96.43% |
| `topdown_y_radial` | 357 / 392 | 91.07% |
| `topdown_y_inward` | 357 / 392 | 91.07% |
| `home` | 338 / 392 | 86.22% |

By refined position:

| Axis/bin | Result |
| --- | --- |
| `r=0.30` | 218 / 224 = 97.32% |
| `r=0.33` | 216 / 224 = 96.43% |
| `r=0.36` | 216 / 224 = 96.43% |
| `r=0.39` | 212 / 224 = 94.64% |
| `r=0.42` | 202 / 224 = 90.18% |
| `r=0.45` | 192 / 224 = 85.71% |
| `r=0.48` | 174 / 224 = 77.68% |
| `theta=-10°,0°,10°` | each 224 / 224 = 100% |
| `theta=-20°,20°` | each 212 / 224 = 94.64% |
| `theta=-30°,30°` | about 74–75% |
| `z_offset=0.12–0.20` | all above 91%; `z_offset=0.08` is lower at 88.27% |

Important 100% subregions from the refined sweep:

| Region | Success / Total | Notes |
| --- | ---: | --- |
| `horizontal_outward`, `r=0.30–0.39`, `theta=-30°..30°`, `z_offset=0.08–0.20` | 224 / 224 | best orientation-specific robust region |
| `horizontal_outward`, `r=0.33–0.39`, `|theta|<=20°`, `z_offset=0.12–0.20` | 90 / 90 | conservative recommendation avoiding the fan inner edge |
| `topdown_y_radial` + `topdown_y_inward`, `r=0.33–0.39`, `|theta|<=20°`, `z_offset=0.12–0.20` | 180 / 180 | top-down poses also fully reachable in this core region |
| `horizontal_outward/topdown_y_radial/topdown_y_inward`, `r≈0.35–0.42`, `|theta|<=20°`, `z_offset=0.12–0.20` | 270 / 270 | task-friendly range; grid points are `r=0.36,0.39,0.42` |

## Recommended clean-table cuRobo planning range

For high cuRobo success on the clean table, use:

```text
r:          0.35–0.42   (core: 0.33–0.39; avoid going much beyond 0.45)
theta:      -20°..20°   (best: -10°..10°)
z_offset:   0.12–0.20 m above table top
TCP poses:  horizontal_outward, topdown_y_radial, or topdown_y_inward
avoid:      horizontal_inward; very low TCP z_offset (~0.04); r >= 0.50
```

For task object sampling, the earlier empirical `r=0.35–0.45` is reasonable,
but this clean-table sweep suggests that the highest cuRobo reliability is more
central and slightly inward:

```text
preferred object/target radial band: r ≈ 0.35–0.42
preferred angular band:              |theta| <= 20°
preferred TCP target height:          table_top_z + 0.12..0.20
```

Note that this is a pure planner result.  Real task success can still be lower
because grasp choice, carried-object geometry, release height, collisions, and
`check_success()` physics are not included in this clean-table sweep.

## Table-height sweep

A separate branch `measure/clean-table-height-workspace` measured the same
clean-table workspace under different table top heights.  The script option is:

```bash
--table-height-bias=<bias>
```

where the actual fan table top is:

```text
table_top_z = 0.74 + table_height_bias
```

### Height sweep command

All heights used the same grid:

```text
arms:        left,right
orientations: home,horizontal_outward,topdown_y_radial,topdown_y_inward
r:           0.30,0.35,0.40,0.45,0.50,0.55
theta_deg:   -40,-20,0,20,40
z_offset:    0.08,0.12,0.16,0.20
```

That is `960` cuRobo planning queries per height.

Example invocation:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS=ignore::UserWarning
/home/admin1/yibo/conda/envs/robotwin/bin/python script/explore_clean_table_curobo_workspace.py \
  --config=demo_clean \
  --table-height-bias=0.02 \
  --arms=left,right \
  --orientation-modes=home,horizontal_outward,topdown_y_radial,topdown_y_inward \
  --r-values=0.30,0.35,0.40,0.45,0.50,0.55 \
  --theta-deg-values=-40,-20,0,20,40 \
  --z-offsets=0.08,0.12,0.16,0.20 \
  --progress-every=0 \
  --top-k=50
```

Two coverage metrics are reported:

- `success_rate`: raw cuRobo success over all 960 queries.
- `robust_bins`: number of `(r, theta, z_offset)` bins where all tested
  arm/orientation combinations succeed.

### Height sweep results

| table_top_z | bias | Success / 960 | Rate | Robust bins / 120 | Recommended-region rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.62 | -0.12 | 193 | 20.10% | 11 | 35.65% |
| 0.66 | -0.08 | 368 | 38.33% | 24 | 68.52% |
| 0.70 | -0.04 | 549 | 57.19% | 37 | 98.15% |
| 0.72 | -0.02 | 621 | 64.69% | 39 | 100.00% |
| 0.74 | +0.00 | 709 | 73.85% | 50 | 100.00% |
| 0.76 | +0.02 | 718 | 74.79% | 52 | 100.00% |
| 0.78 | +0.04 | 717 | 74.69% | 49 | 99.54% |
| 0.80 | +0.06 | 709 | 73.85% | 45 | 97.69% |
| 0.82 | +0.08 | 701 | 73.02% | 43 | 96.30% |
| 0.84 | +0.10 | 680 | 70.83% | 35 | 91.20% |
| 0.86 | +0.12 | 653 | 68.02% | 31 | 88.43% |

The recommended-region rate above is evaluated on:

```text
r:          0.35–0.45
theta:      |theta| <= 20°
z_offset:   0.12–0.20 m
orientations: home,horizontal_outward,topdown_y_radial,topdown_y_inward
arms:       left,right
```

### Best height

The widest clean-table cuRobo coverage is at:

```text
table_top_z ≈ 0.76 m
table_height_bias ≈ +0.02 m
```

Evidence:

- Highest raw coverage: `718/960 = 74.79%`.
- Highest robust-bin coverage: `52/120` position bins where every tested
  arm/orientation combination succeeds.
- The task-friendly recommended region remains perfect: `216/216 = 100%`.

There is a broad plateau around `0.74–0.80 m`; however, very low tables are much
worse, and very high tables start losing coverage:

- `0.62–0.66 m`: poor coverage; low `z_offset` queries are especially bad.
- `0.74–0.78 m`: best plateau.
- `0.82–0.86 m`: still usable but progressively less broad.

### Height-specific recommended range

For the current DYJ clean-table setup, if table height can be adjusted, prefer:

```text
table_top_z: 0.76 m
sampling r:  0.35–0.45, with core at 0.35–0.40
theta:       -20°..20°
z_offset:    0.12–0.20 m above table top
pose modes:  horizontal_outward first; topdown_y_radial/inward are also good
```

This result is planner-only, so task-level checks should still validate object
release, collision with carried geometry, and success conditions.
