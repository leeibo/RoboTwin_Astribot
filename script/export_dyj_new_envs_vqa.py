from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.export_rotate_object_search_visibility_memory_v2 import (  # noqa: E402
    DEFAULT_TASK_WORKERS,
    _default_episode_workers,
    _discover_task_dir,
    _load_whitelist,
    _resolve_task_config_name,
    _resolve_worker_count,
)
from script.export_rotate_object_search_visibility_memory_v3 import (  # noqa: E402
    DEFAULT_OUTPUT_DIR_NAME as DEFAULT_V3_OUTPUT_DIR_NAME,
    export_single_task as export_v3_single_task,
)
from script.rotate_vlm import (  # noqa: E402
    DEFAULT_ACTION_CHUNK_SIZE,
    DEFAULT_MAX_CONTEXT_FRAMES,
    REGISTERED_TASK_TYPES,
    export_task_vlm_dataset,
)


DEFAULT_DYJ_WHITELIST = "task_config/dyj_new_envs_whitelist.yml"


def _write_summary(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _task_name_from_save_dir(save_dir: Path) -> str:
    if save_dir.parent.name == "data":
        return save_dir.name
    return save_dir.parent.name or save_dir.name


def _export_one_task(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    task_name = str(job["task_name"])
    task_dir = Path(job["task_dir"])
    started_at = time.time()
    task_config_name = _resolve_task_config_name(task_dir, job.get("task_config"))
    result: dict[str, Any] = {
        "status": "ok",
        "task_dir": str(task_dir),
        "task_config_name": task_config_name,
    }

    if bool(job.get("export_legacy", True)):
        legacy_summary = export_task_vlm_dataset(
            save_dir=str(task_dir),
            overwrite=True,
            max_context_frames=int(job["max_context_frames"]),
            action_chunk_size=int(job["action_chunk_size"]),
            task_types=list(job["legacy_task_types"]),
            num_workers=int(job["episode_workers"]),
            episode_indices=job.get("episode_indices"),
            max_episodes=job.get("max_episodes"),
        )
        result["legacy_vlm"] = {
            "output_dir": str(task_dir / "vlm"),
            "task_type_counts": legacy_summary.get("task_type_counts", {}),
            "sample_count": int(legacy_summary.get("sample_count", 0)),
            "worker_count": int(legacy_summary.get("worker_count", job["episode_workers"])),
            "samples_paths": legacy_summary.get("samples_paths", {}),
        }

    if bool(job.get("export_v3", True)):
        v3_summary = export_v3_single_task(
            save_dir=str(task_dir),
            task_config_name=task_config_name,
            output_dir_name=str(job["v3_output_dir_name"]),
            max_context_frames=int(job["max_context_frames"]),
            action_chunk_size=int(job["action_chunk_size"]),
            num_workers=int(job["episode_workers"]),
            overwrite=True,
            episode_indices=job.get("episode_indices"),
            max_episodes=job.get("max_episodes"),
        )
        result["object_search_visibility_memory_v3"] = {
            "output_dir": v3_summary.get("output_dir"),
            "output_path": v3_summary.get("output_path"),
            "task_type_counts": v3_summary.get("task_type_counts", {}),
            "sample_count": int(v3_summary.get("sample_count", 0)),
            "search_count": int(v3_summary.get("search_count", 0)),
            "direct_count": int(v3_summary.get("direct_count", 0)),
            "action_count": int(v3_summary.get("action_count", 0)),
            "worker_count": int(v3_summary.get("worker_count", job["episode_workers"])),
        }

    result["elapsed_seconds"] = round(float(time.time() - started_at), 3)
    return task_name, result


def _print_task_ok(task_name: str, task_summary: dict[str, Any]) -> None:
    chunks: list[str] = []
    legacy = task_summary.get("legacy_vlm", {}) or {}
    if legacy:
        chunks.append(f"legacy={legacy.get('task_type_counts', {})}")
    v3 = task_summary.get("object_search_visibility_memory_v3", {}) or {}
    if v3:
        chunks.append(
            "v3="
            f"{v3.get('task_type_counts', {})} "
            f"(search={v3.get('search_count', 0)}, direct={v3.get('direct_count', 0)}, action={v3.get('action_count', 0)})"
        )
    elapsed = task_summary.get("elapsed_seconds", 0)
    print(f"[ok] {task_name}: {', '.join(chunks)} ({elapsed}s)")


def _load_dyj_whitelist(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"whitelist file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("tasks", "include", "task_list", "whitelist_tasks", "selected_tasks"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise SystemExit(f"unsupported whitelist format: {path}")
    return _load_whitelist(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export VQA files for dyj new rotate/fan-double tasks. "
            "This writes legacy vlm/{object_search,angle_delta,memory_compression_vqa}.json "
            "and the v3 action-expanded object_search_visibility_memory dataset."
        )
    )
    parser.add_argument("--save-dir", type=str, default=None, help="Optional single collected task directory.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--whitelist-file", type=str, default=DEFAULT_DYJ_WHITELIST)
    parser.add_argument("--task-config", type=str, default=None, help="Config prefix, e.g. demo_randomized_easy_ep200")
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--v3-output-dir-name", type=str, default=DEFAULT_V3_OUTPUT_DIR_NAME)
    parser.add_argument("--max-context-frames", type=int, default=DEFAULT_MAX_CONTEXT_FRAMES)
    parser.add_argument("--action-chunk-size", type=int, default=DEFAULT_ACTION_CHUNK_SIZE)
    parser.add_argument("--episode-indices", nargs="*", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--task-workers", type=int, default=None)
    parser.add_argument("--episode-workers", type=int, default=None)
    parser.add_argument("--legacy-task-types", nargs="*", default=list(REGISTERED_TASK_TYPES))
    parser.add_argument("--skip-legacy", action="store_true", help="Do not write vlm/*.json legacy VQA files.")
    parser.add_argument("--skip-v3", action="store_true", help="Do not write the v3 action-expanded VQA file.")
    args = parser.parse_args()

    if args.skip_legacy and args.skip_v3:
        raise SystemExit("nothing to export: both --skip-legacy and --skip-v3 were set")

    episode_indices = None if args.episode_indices is None else [int(idx) for idx in args.episode_indices]
    pending_jobs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    if args.save_dir:
        task_dir = Path(args.save_dir)
        pending_jobs.append(
            {
                "task_name": _task_name_from_save_dir(task_dir),
                "task_dir": str(task_dir),
                "task_config": args.task_config,
            }
        )
    else:
        data_root = Path(args.data_root)
        for task_name in _load_dyj_whitelist(Path(args.whitelist_file)):
            task_dir = _discover_task_dir(data_root=data_root, task_name=task_name, task_config=args.task_config)
            if task_dir is None:
                summary[task_name] = {"status": "missing_task_dir"}
                print(f"[skip] {task_name}: task dir not found")
                continue
            pending_jobs.append(
                {
                    "task_name": task_name,
                    "task_dir": str(task_dir),
                    "task_config": args.task_config,
                }
            )

    task_workers = _resolve_worker_count(args.task_workers, DEFAULT_TASK_WORKERS, upper_bound=len(pending_jobs) or 1)
    episode_workers = _resolve_worker_count(args.episode_workers, _default_episode_workers(task_workers))
    legacy_task_types = [str(task_type) for task_type in args.legacy_task_types]
    invalid_task_types = [task_type for task_type in legacy_task_types if task_type not in set(REGISTERED_TASK_TYPES)]
    if invalid_task_types:
        raise SystemExit(f"unsupported legacy task type(s): {invalid_task_types}")

    for job in pending_jobs:
        job.update(
            {
                "episode_workers": int(episode_workers),
                "max_context_frames": int(args.max_context_frames),
                "action_chunk_size": max(int(args.action_chunk_size), 1),
                "legacy_task_types": list(legacy_task_types),
                "v3_output_dir_name": str(args.v3_output_dir_name),
                "episode_indices": episode_indices,
                "max_episodes": args.max_episodes,
                "export_legacy": not bool(args.skip_legacy),
                "export_v3": not bool(args.skip_v3),
            }
        )

    if pending_jobs:
        print(
            f"[start] exporting {len(pending_jobs)} dyj task(s) "
            f"with task_workers={task_workers}, episode_workers={episode_workers}"
        )

    if task_workers <= 1 or len(pending_jobs) <= 1:
        for job in pending_jobs:
            task_name = str(job["task_name"])
            try:
                finished_task_name, task_summary = _export_one_task(job)
                summary[finished_task_name] = task_summary
                _print_task_ok(finished_task_name, task_summary)
            except Exception as exc:
                summary[task_name] = {
                    "status": "failed",
                    "task_dir": job["task_dir"],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                print(f"[failed] {task_name}: {type(exc).__name__}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=int(task_workers)) as executor:
            future_map = {executor.submit(_export_one_task, job): job for job in pending_jobs}
            for future in as_completed(future_map):
                job = future_map[future]
                task_name = str(job["task_name"])
                try:
                    finished_task_name, task_summary = future.result()
                    summary[finished_task_name] = task_summary
                    _print_task_ok(finished_task_name, task_summary)
                except Exception as exc:
                    summary[task_name] = {
                        "status": "failed",
                        "task_dir": job["task_dir"],
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                    print(f"[failed] {task_name}: {type(exc).__name__}: {exc}")

    _write_summary(args.summary_path, summary)


if __name__ == "__main__":
    main()
