import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.rotate_vlm import export_task_vlm_dataset  # noqa: E402


DEFAULT_TASK_WORKERS = max(1, min(4, int(os.cpu_count() or 1)))
DEFAULT_EPISODE_WORKERS = max(1, min(8, int(os.cpu_count() or 1)))


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


def _resolve_worker_count(value: int | None, fallback: int, upper_bound: int | None = None) -> int:
    try:
        workers = int(value) if value is not None else int(fallback)
    except (TypeError, ValueError):
        workers = int(fallback)
    workers = max(1, workers)
    if upper_bound is not None:
        workers = min(workers, int(upper_bound))
    return workers


def _default_episode_workers(task_workers: int) -> int:
    total_cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(DEFAULT_EPISODE_WORKERS, total_cpu // max(1, int(task_workers))))


def _export_single_task(job: tuple[str, str, int]) -> tuple[str, dict]:
    task_name, task_dir, episode_workers = job
    started_at = time.time()
    export_summary = export_task_vlm_dataset(
        save_dir=str(task_dir),
        overwrite=True,
        num_workers=int(episode_workers),
    )
    return (
        task_name,
        {
            "status": "ok",
            "task_dir": str(task_dir),
            "task_type_counts": export_summary.get("task_type_counts", {}),
            "sample_count": int(export_summary.get("sample_count", 0)),
            "worker_count": int(export_summary.get("worker_count", episode_workers)),
            "elapsed_seconds": round(float(time.time() - started_at), 3),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rotate VLM datasets for all whitelist tasks under data/.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--whitelist-file", type=str, default="task_config/rotate_task_whitelist.yml")
    parser.add_argument("--task-config", type=str, default=None, help="Optional config prefix, e.g. demo_randomized_easy_ep2")
    parser.add_argument("--summary-path", type=str, default=None, help="Optional JSON summary output path")
    parser.add_argument(
        "--task-workers",
        type=int,
        default=None,
        help="Number of whitelist tasks exported concurrently. Defaults to min(4, cpu_count).",
    )
    parser.add_argument(
        "--episode-workers",
        type=int,
        default=None,
        help=(
            "Episode workers per task. Defaults to a CPU-budgeted split based on --task-workers. "
            "Passed through to export_task_vlm_dataset()."
        ),
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    whitelist = _load_whitelist(Path(args.whitelist_file))
    summary: dict[str, dict] = {}
    pending_jobs: list[tuple[str, str, int]] = []

    for task_name in whitelist:
        task_dir = _discover_task_dir(data_root=data_root, task_name=task_name, task_config=args.task_config)
        if task_dir is None:
            summary[task_name] = {"status": "missing_task_dir"}
            print(f"[skip] {task_name}: task dir not found")
            continue

        pending_jobs.append((task_name, str(task_dir), 1))

    task_workers = _resolve_worker_count(args.task_workers, DEFAULT_TASK_WORKERS, upper_bound=len(pending_jobs) or 1)
    episode_workers = _resolve_worker_count(
        args.episode_workers,
        _default_episode_workers(task_workers),
    )
    pending_jobs = [(task_name, task_dir, episode_workers) for task_name, task_dir, _ in pending_jobs]

    if pending_jobs:
        print(
            f"[start] exporting {len(pending_jobs)} task(s) "
            f"with task_workers={task_workers}, episode_workers={episode_workers}"
        )

    if task_workers <= 1 or len(pending_jobs) <= 1:
        for job in pending_jobs:
            task_name = job[0]
            try:
                _, task_summary = _export_single_task(job)
                summary[task_name] = task_summary
                print(
                    f"[ok] {task_name}: {task_summary['task_type_counts']} "
                    f"(samples={task_summary['sample_count']}, {task_summary['elapsed_seconds']}s)"
                )
            except Exception as exc:
                summary[task_name] = {
                    "status": "failed",
                    "task_dir": job[1],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                print(f"[failed] {task_name}: {type(exc).__name__}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=task_workers) as executor:
            future_map = {executor.submit(_export_single_task, job): job for job in pending_jobs}
            for future in as_completed(future_map):
                task_name, task_dir, _ = future_map[future]
                try:
                    finished_task_name, task_summary = future.result()
                    summary[finished_task_name] = task_summary
                    print(
                        f"[ok] {finished_task_name}: {task_summary['task_type_counts']} "
                        f"(samples={task_summary['sample_count']}, {task_summary['elapsed_seconds']}s)"
                    )
                except Exception as exc:
                    summary[task_name] = {
                        "status": "failed",
                        "task_dir": task_dir,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                    print(f"[failed] {task_name}: {type(exc).__name__}: {exc}")

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
