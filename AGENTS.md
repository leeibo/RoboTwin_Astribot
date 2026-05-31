# Repository Guidelines

## Project Structure & Module Organization
`envs/` contains simulation tasks, robot interfaces, and shared environment utilities. Task modules are generally snake_case files with matching class names (for example, `envs/beat_block_hammer.py` -> `class beat_block_hammer`). `script/` contains operational entrypoints (install, asset download, data collection, evaluation, calibration). `policy/` hosts baseline policy implementations (DP, RDT, pi0, pi05, TinyVLA, etc.), each with its own configs and sometimes its own dependency/tooling stack. `description/` holds language/task/object description generators. `test/` contains local sanity scripts (mostly IK/FK checks). Generated artifacts live in `data/`, `eval_result/`, and `assets/` (downloaded content); do not commit these outputs.

## Build, Test, and Development Commands
Use these from repository root unless noted:

```bash
bash script/_install.sh
bash script/_download_assets.sh
bash collect_data.sh beat_block_hammer demo_randomized 0
python script/eval_policy.py --config policy/DP/deploy_policy.yml --overrides --task_name beat_block_hammer --task_config demo_randomized --ckpt_setting debug --expert_data_num 50 --seed 0 --policy_name DP
python script/policy_model_server.py --config policy/DP/deploy_policy.yml --port 9999
python script/eval_policy_client.py --config policy/DP/deploy_policy.yml --port 9999 --overrides --task_name beat_block_hammer --task_config demo_randomized --ckpt_setting debug --expert_data_num 50 --seed 0 --policy_name DP
```

## Coding Style & Naming Conventions
Use Python with 4-space indentation and UTF-8. Prefer clear, small modules and keep runtime settings in YAML configs (`task_config/*.yml`, `policy/*/deploy_policy.yml`) plus CLI `--overrides`, rather than hard-coded values. Follow existing naming patterns: snake_case modules/functions, and task class names aligned with task filenames. For policy subprojects that define lint rules (notably `policy/pi0`, `policy/pi05`, `policy/openvla-oft`), run their local tooling (for example, `ruff check .`) before submitting.

## Testing Guidelines
Run targeted checks for the area you change:
- `python script/test_render.py` for renderer/simulator readiness.
- `python test/test_ik_fk.py` or `python test/find_ik.py` for kinematics sanity checks (CUDA and valid asset paths required).
- `(cd policy/pi0 && pytest)` or `(cd policy/pi05 && pytest)` for OpenPI-style modules.

No repository-wide coverage threshold is enforced; include the exact test commands you ran in your PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages (`fix ...`, `update ...`, `add ...`), sometimes with scoped prefixes (for example, `policy/pi0: fix eval config`). Keep commits focused to one logical change. PRs should include: intent, key files changed, reproducible run/test commands, and evidence (logs, metrics, or screenshots/videos for rendering/eval changes). Link related issues and highlight any required asset/config updates.

## DYJ 34-Task Clean Environment
Use this project as the clean DYJ integration environment:

```bash
cd /home/admin1/Desktop/RoboTwin_main_dyj_clean
git branch --show-current
```

Expected branch:

```text
merge-main-dyj-34tasks
```

Important recent commits:

```text
adeeeed refactor dyj task shared helpers
f147000 move dyj table config to task configs
```

This environment intentionally keeps only the DYJ 34 task modules in `envs/`. The authoritative task list is:

```bash
task_config/rotate_task_whitelist.yml
```

Use the existing RobotWin conda Python explicitly unless the shell has already activated the same environment:

```bash
export ROBOTWIN_PY=/home/admin1/yibo/conda/envs/robotwin/bin/python
```

Default verified config:

```text
demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer
```

The corresponding output directory suffix is:

```text
demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer__easy_fan150_randenv_ep1cam_observer
```

Generated data normally goes under `data_view/` because this config sets `save_path: ./data_view`. Treat `data/`, `data_view/`, `.cache/`, and `logs/` as generated artifacts.

### Single Task Collection
Run from repository root:

```bash
export ROBOTWIN_PY=/home/admin1/yibo/conda/envs/robotwin/bin/python
export CONFIG=demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer

ROBOTWIN_SKIP_RENDER_TEST=1 \
ROBOTWIN_SKIP_ANNOTATED_VIDEO=1 \
ROBOTWIN_SKIP_INSTRUCTIONS=1 \
ROBOTWIN_START_SEED=0 \
ROBOTWIN_MAX_SEED_TRIES=100 \
CUDA_VISIBLE_DEVICES=0 \
PYTHONWARNINGS=ignore::UserWarning \
"$ROBOTWIN_PY" script/collect_data.py beat_block_hammer_rotate_view "$CONFIG"
```

Useful environment variables:

```text
ROBOTWIN_START_SEED       first seed to try
ROBOTWIN_MAX_SEED_TRIES   exclusive upper seed bound; use -1 for unlimited
ROBOTWIN_SKIP_RENDER_TEST skip initial render sanity test
ROBOTWIN_SKIP_ANNOTATED_VIDEO skip annotated video postprocess
ROBOTWIN_SKIP_INSTRUCTIONS skip instruction generation postprocess
CUDA_VISIBLE_DEVICES      GPU id
```

### Batch Collection
For a simple 34-task batch, make sure the RobotWin Python is first on `PATH`, because `collect_dyj_34_tasks.sh` invokes `python`:

```bash
export PATH=/home/admin1/yibo/conda/envs/robotwin/bin:$PATH
CONTINUE_ON_ERROR=1 TASK_TIMEOUT_SECONDS=1800 \
bash collect_dyj_34_tasks.sh demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer 0
```

For deterministic rechecks, use known seed windows. Example pattern:

```bash
export ROBOTWIN_PY=/home/admin1/yibo/conda/envs/robotwin/bin/python
export CONFIG=demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer
mkdir -p logs

while read -r TASK SEED; do
  [ -z "$TASK" ] && continue
  MAX=$((SEED + 100))
  ROBOTWIN_SKIP_RENDER_TEST=1 \
  ROBOTWIN_SKIP_ANNOTATED_VIDEO=1 \
  ROBOTWIN_SKIP_INSTRUCTIONS=1 \
  ROBOTWIN_START_SEED="$SEED" \
  ROBOTWIN_MAX_SEED_TRIES="$MAX" \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONWARNINGS=ignore::UserWarning \
  "$ROBOTWIN_PY" script/collect_data.py "$TASK" "$CONFIG" \
    > "logs/collect_${TASK}.log" 2>&1
done < /tmp/robotwin_known_seed_collect.tsv
```

### Verified Seeds
The last full 34-task verified collection in `data_view/` used these actual collected seeds:

```text
beat_block_hammer_rotate_view 1
blocks_ranking_rgb_fan_double 188
blocks_ranking_rgb_rotate_view 9
blocks_ranking_size_fan_double 29
blocks_ranking_size_rotate_view 12
click_alarmclock_rotate_view 11
click_bell_rotate_view 0
move_pillbottle_pad_rotate_view 1
move_stapler_pad_rotate_view 10
open_laptop_rotate_view 0
place_a2b_left_rotate_view 51
place_a2b_right_rotate_view 30
place_cans_plasticbox_rotate_view 77
place_container_plate_rotate_view 0
place_empty_cup_rotate_view 0
place_fan_rotate_view 1
place_mouse_pad_rotate_view 6
place_object_basket_fan_double 44
place_object_scale_rotate_view 4
place_object_stand_rotate_view 11
place_shoe_rotate_view 34
press_stapler_rotate_view 0
put_block_breadbasket_fan_double 3
put_block_on_upper_easy 0
put_block_on_upper_hard 14
put_block_plasticbox_fan_double 0
put_block_skillet_fan_double 0
search_object 35
shake_bottle_horizontally_rotate_view 3
shake_bottle_rotate_view 8
stack_blocks_three_rotate_view 13
stack_blocks_two_rotate_view 5
stamp_seal_rotate_view 0
turn_switch_rotate_view 0
```

Seed `>= 10` tasks:

```text
blocks_ranking_rgb_fan_double 188
blocks_ranking_size_fan_double 29
blocks_ranking_size_rotate_view 12
click_alarmclock_rotate_view 11
move_stapler_pad_rotate_view 10
place_a2b_left_rotate_view 51
place_a2b_right_rotate_view 30
place_cans_plasticbox_rotate_view 77
place_object_basket_fan_double 44
place_object_stand_rotate_view 11
place_shoe_rotate_view 34
put_block_on_upper_hard 14
search_object 35
stack_blocks_three_rotate_view 13
```

### Validation Commands
Compile all root-level task files and `collect_data.py`:

```bash
export ROBOTWIN_PY=/home/admin1/yibo/conda/envs/robotwin/bin/python
"$ROBOTWIN_PY" -m py_compile $(find envs -maxdepth 1 -type f -name '*.py' | sort) script/collect_data.py
git diff --check
```

Check that `envs/` contains exactly the 34 whitelisted task modules:

```bash
"$ROBOTWIN_PY" - <<'PY'
from pathlib import Path
import yaml

whitelist = set(yaml.safe_load(Path("task_config/rotate_task_whitelist.yml").read_text()))
env_tasks = {
    path.stem
    for path in Path("envs").glob("*.py")
    if not path.name.startswith("_") and path.name != "__init__.py"
}
print("whitelist_count=", len(whitelist))
print("env_task_count=", len(env_tasks))
print("missing_in_env=", sorted(whitelist - env_tasks))
print("extra_in_env=", sorted(env_tasks - whitelist))
raise SystemExit(0 if whitelist == env_tasks else 1)
PY
```

Check 34-task artifacts after collection:

```bash
"$ROBOTWIN_PY" - <<'PY'
from pathlib import Path
import json
import subprocess
import h5py
import yaml

config = "demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer__easy_fan150_randenv_ep1cam_observer"
tasks = yaml.safe_load(Path("task_config/rotate_task_whitelist.yml").read_text())
failed = []
for task in tasks:
    base = Path("data_view") / task / config
    for rel in ["seed.txt", "data/episode0.hdf5", "video/episode0.mp4"]:
        path = base / rel
        if not path.exists() or path.stat().st_size <= 0:
            failed.append((task, rel, "missing_or_empty"))
    if (base / "collection_failure.json").exists():
        failed.append((task, "collection_failure.json", "exists"))
    hdf5_path = base / "data" / "episode0.hdf5"
    if hdf5_path.exists():
        with h5py.File(hdf5_path, "r") as file:
            if len(file.keys()) == 0:
                failed.append((task, "hdf5", "empty_keys"))
    video_path = base / "video" / "episode0.mp4"
    if video_path.exists():
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames", "-of", "json",
                str(video_path),
            ],
            text=True,
            capture_output=True,
        )
        if proc.returncode:
            failed.append((task, "ffprobe", proc.stderr.strip()))
        else:
            streams = json.loads(proc.stdout or "{}").get("streams") or []
            frames = streams[0].get("nb_frames") if streams else None
            if frames in (None, "N/A") or int(frames) <= 0:
                failed.append((task, "video_frames", str(frames)))

print("artifact_checked=", len(tasks))
print("artifact_failed=", len(failed))
for item in failed[:50]:
    print(item)
raise SystemExit(1 if failed else 0)
PY
```

### Code Organization Notes
Table geometry is configured in YAML under `task_table_configs`; task files should only select `ROTATE_TABLE_SHAPE` and, for named double-table profiles, `ROTATE_TABLE_CONFIG_KEY`.

Common rotate-view helpers now live in `Base_Task`, including `_get_robot_root_xy_yaw()` and `_scan_scene_two_views()`. Task-specific scan differences should be expressed with class attributes such as:

```python
ROTATE_SCAN_SCENE_R = 0.64
ROTATE_SCAN_SCENE_Z_BIAS = 0.90
ROTATE_SCAN_SCENE_FALLBACK_THETAS = (1.00, -1.00)
```

The 5 block-to-plate fan-double tasks use `PutBlockFanDoubleMixin` in `envs/_base_task.py`; keep shared private block/plate helper logic there instead of duplicating it across task files.
