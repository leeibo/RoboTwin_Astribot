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
