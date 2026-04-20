import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.rotate_vlm import export_task_vlm_dataset  # noqa: E402


def _load_whitelist(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("tasks", "include", "task_list", "whitelist_tasks", "selected_tasks"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise SystemExit(f"unsupported whitelist format: {path}")
    tasks = []
    seen = set()
    for item in data:
        task_name = str(item).strip()
        if not task_name or task_name in seen:
            continue
        seen.add(task_name)
        tasks.append(task_name)
    return tasks


def _discover_task_dir(data_root: Path, task_name: str, task_config: str | None) -> Path | None:
    task_root = data_root / task_name
    if not task_root.exists():
        return None
    candidates = sorted(path for path in task_root.iterdir() if path.is_dir())
    if task_config:
        prefix = f"{task_config}__"
        candidates = [path for path in candidates if path.name.startswith(prefix)]
    if not candidates:
        return None
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rotate VLM datasets for all whitelist tasks under data/.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--whitelist-file", type=str, default="task_config/rotate_task_whitelist.yml")
    parser.add_argument("--task-config", type=str, default=None, help="Optional config prefix, e.g. demo_randomized_easy_ep2")
    parser.add_argument("--summary-path", type=str, default=None, help="Optional JSON summary output path")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    whitelist = _load_whitelist(Path(args.whitelist_file))
    summary: dict[str, dict] = {}

    for task_name in whitelist:
        task_dir = _discover_task_dir(data_root=data_root, task_name=task_name, task_config=args.task_config)
        if task_dir is None:
            summary[task_name] = {"status": "missing_task_dir"}
            print(f"[skip] {task_name}: task dir not found")
            continue
        export_summary = export_task_vlm_dataset(save_dir=str(task_dir), overwrite=True)
        summary[task_name] = {
            "status": "ok",
            "task_dir": str(task_dir),
            "task_type_counts": export_summary.get("task_type_counts", {}),
            "sample_count": int(export_summary.get("sample_count", 0)),
        }
        print(f"[ok] {task_name}: {summary[task_name]['task_type_counts']}")

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
