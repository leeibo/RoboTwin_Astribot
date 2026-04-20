from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.rotate_vlm import export_task_vlm_dataset  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else dict(payload)


def _load_whitelist(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("tasks", "include", "task_list", "whitelist_tasks", "selected_tasks"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise SystemExit(f"unsupported whitelist format: {path}")
    tasks: list[str] = []
    seen = set()
    for item in data:
        task_name = str(item).strip()
        if not task_name or task_name in seen:
            continue
        seen.add(task_name)
        tasks.append(task_name)
    return tasks


def _sanitize_tag(text: str) -> str:
    return str(text).strip().replace(" ", "_")


def _infer_difficulty_tag(config: dict[str, Any]) -> str:
    custom_tag = config.get("difficulty_tag", None)
    if custom_tag is not None and str(custom_tag).strip():
        return _sanitize_tag(str(custom_tag))

    fan_angle_deg = config.get("fan_angle_deg", None)
    if fan_angle_deg is None:
        return "unknown"

    fan_angle = float(fan_angle_deg)
    fan_angle_int = int(round(fan_angle))
    if fan_angle <= 170.0:
        level = "easy"
    elif fan_angle <= 220.0:
        level = "medium"
    else:
        level = "hard"
    return f"{level}_fan{fan_angle_int}"


def _storage_setting(task_config: str, config: dict[str, Any]) -> str:
    return f"{task_config}__{_infer_difficulty_tag(config)}"


def _requested_episode_num(task_config: str) -> int:
    config = _load_yaml(REPO_ROOT / "task_config" / f"{task_config}.yml")
    return int(config.get("episode_num", 0))


def _requested_language_num(task_config: str) -> int:
    config = _load_yaml(REPO_ROOT / "task_config" / f"{task_config}.yml")
    return int(config.get("language_num", 50))


def _read_seed_list(task_dir: Path) -> list[int]:
    seed_path = task_dir / "seed.txt"
    if not seed_path.exists():
        return []
    raw = seed_path.read_text(encoding="utf-8").split()
    return [int(value) for value in raw if str(value).strip()]


def _count_hdf5_episodes(task_dir: Path) -> int:
    return len(sorted((task_dir / "data").glob("episode*.hdf5")))


def _count_instruction_files(task_dir: Path) -> int:
    return len(sorted((task_dir / "instructions").glob("episode*.json")))


def _instructions_complete(task_dir: Path, requested_episode_num: int) -> bool:
    return _count_instruction_files(task_dir) >= int(requested_episode_num)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _attempted_seed_count(seed_list: list[int], failure_report: dict[str, Any] | None) -> int | None:
    if failure_report is not None:
        if failure_report.get("next_seed_to_try") is not None:
            return int(failure_report["next_seed_to_try"])
        if failure_report.get("last_attempted_seed") is not None:
            return int(failure_report["last_attempted_seed"]) + 1
    if seed_list:
        return max(seed_list) + 1
    return None


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _task_dir(data_root: Path, task_name: str, storage_setting: str) -> Path:
    return data_root / task_name / storage_setting


def _task_report_path(task_dir: Path) -> Path:
    return task_dir / "collection_task_report.json"


def _per_task_report_copy_path(data_root: Path, task_name: str, task_config: str) -> Path:
    return data_root / "collection_reports" / "per_task" / f"{task_name}__{task_config}.json"


def _collect_log_path(log_dir: Path, task_config: str, task_name: str) -> Path:
    return log_dir / f"collect_rotate_randomized_whitelist__{task_config}__{task_name}.log"


def _export_log_path(log_dir: Path, task_config: str, task_name: str) -> Path:
    return log_dir / f"export_rotate_randomized_whitelist__{task_config}__{task_name}.log"


def _vlm_export_complete(task_dir: Path, requested_episode_num: int) -> bool:
    vlm_dir = task_dir / "vlm"
    required = (
        vlm_dir / "object_search.json",
        vlm_dir / "angle_delta.json",
        vlm_dir / "memory_compression_vqa.json",
        vlm_dir / "manifest.json",
    )
    if not all(path.exists() for path in required):
        return False
    manifest = _read_json_if_exists(vlm_dir / "manifest.json")
    if manifest is None:
        return False
    if int(manifest.get("episode_count", 0) or 0) < int(requested_episode_num):
        return False
    if int(manifest.get("annotated_video_count", 0) or 0) < int(requested_episode_num):
        return False
    return True


def _combined_task_status(payload: dict[str, Any], export_vqa: bool) -> str:
    if str(payload.get("collect_status", "pending")) == "failed":
        return "collect_failed"
    if str(payload.get("collect_status", "pending")) != "ok":
        return "collect_pending"
    if str(payload.get("instruction_status", "pending")) == "failed":
        return "instruction_failed"
    if str(payload.get("instruction_status", "pending")) != "ok":
        return "collect_ok_postprocess_pending"
    if not export_vqa:
        return "ok"
    if str(payload.get("export_status", "pending")) == "failed":
        return "export_failed"
    if str(payload.get("export_status", "pending")) != "ok":
        return "collect_ok_postprocess_pending"
    return "ok"


def _build_task_payload(
    *,
    task_name: str,
    task_config: str,
    data_root: Path,
    storage_setting: str,
    log_dir: Path,
    export_vqa: bool,
    existing_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_dir = _task_dir(data_root, task_name, storage_setting)
    collect_log_path = _collect_log_path(log_dir, task_config, task_name)
    export_log_path = _export_log_path(log_dir, task_config, task_name)
    requested_episode_num = _requested_episode_num(task_config)
    seed_list = _read_seed_list(task_dir)
    failure_report_path = task_dir / "collection_failure.json"
    failure_report = _read_json_if_exists(failure_report_path)
    attempted_seed_count = _attempted_seed_count(seed_list, failure_report)
    collected_episode_num = _count_hdf5_episodes(task_dir)
    instruction_file_count = _count_instruction_files(task_dir)

    report = {} if existing_report is None else dict(existing_report)
    if collected_episode_num >= requested_episode_num:
        collect_status = "ok"
    elif report.get("collect_exit_code") not in (None, 0) or failure_report is not None:
        collect_status = "failed"
    else:
        collect_status = "pending"

    if report.get("instruction_status") == "failed":
        instruction_status = "failed"
    elif _instructions_complete(task_dir, requested_episode_num):
        instruction_status = "ok"
    else:
        instruction_status = "pending"

    if not export_vqa:
        export_status = "skipped"
    elif report.get("export_status") == "failed":
        export_status = "failed"
    elif _vlm_export_complete(task_dir, requested_episode_num):
        export_status = "ok"
    else:
        export_status = "pending"

    success_rate = None
    if attempted_seed_count not in (None, 0):
        success_rate = float(len(seed_list)) / float(attempted_seed_count)

    payload = {
        "task_name": task_name,
        "task_config": task_config,
        "storage_setting": storage_setting,
        "task_dir": str(task_dir),
        "log_path": str(collect_log_path),
        "collect_log_path": str(collect_log_path),
        "export_log_path": str(export_log_path),
        "requested_episode_num": int(requested_episode_num),
        "seed_success_episode_num": int(len(seed_list)),
        "collected_episode_num": int(collected_episode_num),
        "attempted_seed_count": attempted_seed_count,
        "seed_success_rate": success_rate,
        "failure_report_path": (str(failure_report_path) if failure_report_path.exists() else None),
        "failure_report": failure_report,
        "collect_status": collect_status,
        "collect_exit_code": report.get("collect_exit_code"),
        "collect_started_at": report.get("collect_started_at"),
        "collect_finished_at": report.get("collect_finished_at"),
        "collect_elapsed_seconds": report.get("collect_elapsed_seconds", 0.0),
        "instruction_status": instruction_status,
        "instruction_file_count": int(instruction_file_count),
        "instruction_started_at": report.get("instruction_started_at"),
        "instruction_finished_at": report.get("instruction_finished_at"),
        "instruction_elapsed_seconds": report.get("instruction_elapsed_seconds", 0.0),
        "instruction_exit_code": report.get("instruction_exit_code"),
        "instruction_error": report.get("instruction_error"),
        "export_status": export_status,
        "export_started_at": report.get("export_started_at"),
        "export_finished_at": report.get("export_finished_at"),
        "export_elapsed_seconds": report.get("export_elapsed_seconds", 0.0),
        "export_summary": report.get("export_summary"),
        "export_error": report.get("export_error"),
    }
    payload["status"] = _combined_task_status(payload, export_vqa=export_vqa)
    return payload


def _write_task_report(data_root: Path, payload: dict[str, Any]) -> None:
    task_dir = Path(payload["task_dir"])
    report_path = _task_report_path(task_dir)
    per_task_copy = _per_task_report_copy_path(data_root, str(payload["task_name"]), str(payload["task_config"]))
    _write_json(report_path, payload)
    _write_json(per_task_copy, payload)


def _is_collect_complete(payload: dict[str, Any]) -> bool:
    return str(payload.get("collect_status", "")) == "ok"


def _is_export_complete(payload: dict[str, Any], export_vqa: bool) -> bool:
    if str(payload.get("instruction_status", "")) != "ok":
        return False
    if not export_vqa:
        return True
    return str(payload.get("export_status", "")) == "ok"


def _run_collect_task(
    *,
    task_name: str,
    task_config: str,
    data_root: Path,
    storage_setting: str,
    gpu_id: str,
    max_seed_tries: int | None,
    log_dir: Path,
    export_vqa: bool,
    inline_annotated_video: bool,
    inline_instructions: bool,
) -> dict[str, Any]:
    task_dir = _task_dir(data_root, task_name, storage_setting)
    collect_log_path = _collect_log_path(log_dir, task_config, task_name)
    existing_report = _read_json_if_exists(_task_report_path(task_dir))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env["ROBOTWIN_SKIP_ANNOTATED_VIDEO"] = ("0" if inline_annotated_video else "1")
    env["ROBOTWIN_SKIP_INSTRUCTIONS"] = ("0" if inline_instructions else "1")
    if max_seed_tries is not None:
        env["ROBOTWIN_MAX_SEED_TRIES"] = str(int(max_seed_tries))

    command = [sys.executable, "script/collect_data.py", task_name, task_config]
    collect_started_at = time.time()
    collect_started_at_text = _timestamp()
    collect_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(collect_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{collect_started_at_text}] START collect {task_name} {task_config} "
            f"(inline_annotated_video={inline_annotated_video}, inline_instructions={inline_instructions})\n"
        )
        log_file.flush()
        collect_result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.write(f"[{_timestamp()}] END collect {task_name} exit={collect_result.returncode}\n")
        log_file.flush()

    merged_report = {
        **({} if existing_report is None else existing_report),
        "collect_exit_code": int(collect_result.returncode),
        "collect_started_at": collect_started_at_text,
        "collect_finished_at": _timestamp(),
        "collect_elapsed_seconds": round(float(time.time() - collect_started_at), 3),
    }
    if not inline_instructions:
        merged_report["instruction_status"] = "pending"
        merged_report["instruction_started_at"] = None
        merged_report["instruction_finished_at"] = None
        merged_report["instruction_elapsed_seconds"] = 0.0
        merged_report["instruction_exit_code"] = None
        merged_report["instruction_error"] = None
    payload = _build_task_payload(
        task_name=task_name,
        task_config=task_config,
        data_root=data_root,
        storage_setting=storage_setting,
        log_dir=log_dir,
        export_vqa=export_vqa,
        existing_report=merged_report,
    )
    _write_task_report(data_root, payload)
    return payload


def _run_postprocess_task(
    *,
    task_name: str,
    task_config: str,
    data_root: Path,
    storage_setting: str,
    log_dir: Path,
    export_vqa: bool,
) -> dict[str, Any]:
    task_dir = _task_dir(data_root, task_name, storage_setting)
    export_log_path = _export_log_path(log_dir, task_config, task_name)
    existing_report = _read_json_if_exists(_task_report_path(task_dir))
    payload = _build_task_payload(
        task_name=task_name,
        task_config=task_config,
        data_root=data_root,
        storage_setting=storage_setting,
        log_dir=log_dir,
        export_vqa=export_vqa,
        existing_report=existing_report,
    )

    if not _is_collect_complete(payload):
        payload["instruction_status"] = "failed"
        payload["instruction_error"] = {
            "type": "MissingCollection",
            "message": "cannot run postprocess before collection completes",
        }
        if export_vqa:
            payload["export_status"] = "failed"
            payload["export_error"] = {
                "type": "MissingCollection",
                "message": "cannot export VQA before collection completes",
            }
        payload["status"] = _combined_task_status(payload, export_vqa=export_vqa)
        _write_task_report(data_root, payload)
        return payload

    requested_episode_num = int(payload["requested_episode_num"])
    language_num = _requested_language_num(task_config)
    export_log_path.parent.mkdir(parents=True, exist_ok=True)

    instruction_started_at = None
    instruction_elapsed_seconds = float(payload.get("instruction_elapsed_seconds") or 0.0)
    instruction_exit_code = payload.get("instruction_exit_code")
    instruction_error = payload.get("instruction_error")

    export_started_at = None
    export_elapsed_seconds = float(payload.get("export_elapsed_seconds") or 0.0)
    export_summary = payload.get("export_summary")
    export_error = payload.get("export_error")

    with open(export_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{_timestamp()}] START postprocess {task_name} {task_config}\n")
        log_file.flush()

        if _instructions_complete(task_dir, requested_episode_num):
            payload["instruction_status"] = "ok"
            payload["instruction_error"] = None
            if payload.get("instruction_started_at") is None:
                payload["instruction_started_at"] = None
                payload["instruction_finished_at"] = _timestamp()
                payload["instruction_exit_code"] = 0
            log_file.write(f"[{_timestamp()}] SKIP instructions {task_name}: already complete\n")
        else:
            instruction_started_at = time.time()
            payload["instruction_started_at"] = _timestamp()
            instruction_command = [
                "bash",
                "gen_episode_instructions.sh",
                task_name,
                storage_setting,
                str(language_num),
                task_config,
            ]
            result = subprocess.run(
                instruction_command,
                cwd=str(REPO_ROOT / "description"),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            instruction_elapsed_seconds = time.time() - instruction_started_at
            instruction_exit_code = int(result.returncode)
            payload["instruction_finished_at"] = _timestamp()
            payload["instruction_elapsed_seconds"] = round(float(instruction_elapsed_seconds), 3)
            payload["instruction_exit_code"] = instruction_exit_code
            if result.returncode == 0 and _instructions_complete(task_dir, requested_episode_num):
                payload["instruction_status"] = "ok"
                payload["instruction_error"] = None
            else:
                payload["instruction_status"] = "failed"
                instruction_error = {
                    "type": "InstructionExportFailed",
                    "message": f"instruction export exit={result.returncode}",
                }
                payload["instruction_error"] = instruction_error
                log_file.write(f"[{_timestamp()}] END postprocess {task_name} with instruction failure\n")
                log_file.flush()
                payload["status"] = _combined_task_status(payload, export_vqa=export_vqa)
                _write_task_report(data_root, payload)
                return payload

        if not export_vqa:
            payload["export_status"] = "skipped"
            payload["export_error"] = None
            log_file.write(f"[{_timestamp()}] SKIP VQA export {task_name}: --export-vqa not set\n")
            log_file.write(f"[{_timestamp()}] END postprocess {task_name} status=ok\n")
            log_file.flush()
            payload["status"] = _combined_task_status(payload, export_vqa=export_vqa)
            _write_task_report(data_root, payload)
            return payload

        if _vlm_export_complete(task_dir, requested_episode_num):
            payload["export_status"] = "ok"
            payload["export_error"] = None
            if payload.get("export_started_at") is None:
                payload["export_started_at"] = None
                payload["export_finished_at"] = _timestamp()
            log_file.write(f"[{_timestamp()}] SKIP VQA export {task_name}: already complete\n")
        else:
            export_started_at = time.time()
            payload["export_started_at"] = _timestamp()
            try:
                export_summary = export_task_vlm_dataset(save_dir=str(task_dir), overwrite=True)
                payload["export_status"] = "ok"
                payload["export_summary"] = export_summary
                payload["export_error"] = None
            except Exception as exc:  # pragma: no cover - failure is reported in JSON
                payload["export_status"] = "failed"
                export_error = {"type": type(exc).__name__, "message": str(exc)}
                payload["export_error"] = export_error
            export_elapsed_seconds = time.time() - export_started_at
            payload["export_finished_at"] = _timestamp()
            payload["export_elapsed_seconds"] = round(float(export_elapsed_seconds), 3)

        log_file.write(f"[{_timestamp()}] END postprocess {task_name} status={payload['export_status']}\n")
        log_file.flush()

    payload = _build_task_payload(
        task_name=task_name,
        task_config=task_config,
        data_root=data_root,
        storage_setting=storage_setting,
        log_dir=log_dir,
        export_vqa=export_vqa,
        existing_report=payload,
    )
    _write_task_report(data_root, payload)
    return payload


def _upsert_summary_task(run_payload: dict[str, Any], task_payload: dict[str, Any]) -> None:
    tasks = list(run_payload.get("tasks", []))
    for idx, existing in enumerate(tasks):
        if str(existing.get("task_name", "")) == str(task_payload.get("task_name", "")):
            tasks[idx] = task_payload
            run_payload["tasks"] = tasks
            return
    tasks.append(task_payload)
    run_payload["tasks"] = tasks


def _update_summary_counters(run_payload: dict[str, Any], export_vqa: bool) -> None:
    tasks = list(run_payload.get("tasks", []))
    collect_success = sum(1 for item in tasks if str(item.get("collect_status", "")) == "ok")
    collect_failed = sum(1 for item in tasks if str(item.get("collect_status", "")) == "failed")
    instruction_success = sum(1 for item in tasks if str(item.get("instruction_status", "")) == "ok")
    instruction_failed = sum(1 for item in tasks if str(item.get("instruction_status", "")) == "failed")
    export_success = sum(1 for item in tasks if str(item.get("export_status", "")) == "ok")
    export_failed = sum(1 for item in tasks if str(item.get("export_status", "")) == "failed")
    complete_task_num = sum(1 for item in tasks if str(item.get("status", "")) == "ok")
    failed_task_num = sum(
        1 for item in tasks if str(item.get("status", "")) in {"collect_failed", "instruction_failed", "export_failed"}
    )
    run_payload["collect_success_task_num"] = int(collect_success)
    run_payload["collect_failed_task_num"] = int(collect_failed)
    run_payload["instruction_success_task_num"] = int(instruction_success)
    run_payload["instruction_failed_task_num"] = int(instruction_failed)
    run_payload["export_success_task_num"] = int(export_success if export_vqa else 0)
    run_payload["export_failed_task_num"] = int(export_failed if export_vqa else 0)
    run_payload["complete_task_num"] = int(complete_task_num)
    run_payload["success_task_num"] = int(complete_task_num)
    run_payload["failed_task_num"] = int(failed_task_num)


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _format_seconds(seconds: Any) -> str:
    total_seconds = int(round(_to_float(seconds)))
    hours, remain = divmod(total_seconds, 3600)
    minutes, secs = divmod(remain, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _task_stats_rows(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(run_payload.get("tasks", [])):
        export_summary = item.get("export_summary") or {}
        task_type_counts = export_summary.get("task_type_counts", {}) or {}
        collect_elapsed_seconds = _to_float(item.get("collect_elapsed_seconds"))
        instruction_elapsed_seconds = _to_float(item.get("instruction_elapsed_seconds"))
        export_elapsed_seconds = _to_float(item.get("export_elapsed_seconds"))
        total_elapsed_seconds = collect_elapsed_seconds + instruction_elapsed_seconds + export_elapsed_seconds
        success_rate = item.get("seed_success_rate")
        row = {
            "task_name": str(item.get("task_name", "")),
            "status": str(item.get("status", "")),
            "collect_status": str(item.get("collect_status", "")),
            "instruction_status": str(item.get("instruction_status", "")),
            "export_status": str(item.get("export_status", "")),
            "requested_episode_num": int(item.get("requested_episode_num", 0) or 0),
            "collected_episode_num": int(item.get("collected_episode_num", 0) or 0),
            "seed_success_episode_num": int(item.get("seed_success_episode_num", 0) or 0),
            "attempted_seed_count": item.get("attempted_seed_count"),
            "seed_success_rate": success_rate,
            "seed_success_rate_percent": (None if success_rate is None else round(float(success_rate) * 100.0, 2)),
            "instruction_file_count": int(item.get("instruction_file_count", 0) or 0),
            "collect_elapsed_seconds": round(collect_elapsed_seconds, 3),
            "instruction_elapsed_seconds": round(instruction_elapsed_seconds, 3),
            "export_elapsed_seconds": round(export_elapsed_seconds, 3),
            "total_elapsed_seconds": round(total_elapsed_seconds, 3),
            "collect_elapsed_hms": _format_seconds(collect_elapsed_seconds),
            "instruction_elapsed_hms": _format_seconds(instruction_elapsed_seconds),
            "export_elapsed_hms": _format_seconds(export_elapsed_seconds),
            "total_elapsed_hms": _format_seconds(total_elapsed_seconds),
            "sample_count": int(export_summary.get("sample_count", 0) or 0),
            "object_search_count": int(task_type_counts.get("object_search", 0) or 0),
            "angle_delta_count": int(task_type_counts.get("angle_delta", 0) or 0),
            "memory_compression_vqa_count": int(task_type_counts.get("memory_compression_vqa", 0) or 0),
            "task_dir": str(item.get("task_dir", "")),
            "collect_started_at": item.get("collect_started_at"),
            "collect_finished_at": item.get("collect_finished_at"),
            "instruction_started_at": item.get("instruction_started_at"),
            "instruction_finished_at": item.get("instruction_finished_at"),
            "export_started_at": item.get("export_started_at"),
            "export_finished_at": item.get("export_finished_at"),
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_task_stats_reports(summary_path: Path, run_payload: dict[str, Any]) -> None:
    rows = _task_stats_rows(run_payload)
    stem = summary_path.stem
    base_name = (stem[:-9] if stem.endswith("__summary") else stem)
    stats_json_path = summary_path.with_name(f"{base_name}__task_stats.json")
    stats_csv_path = summary_path.with_name(f"{base_name}__task_stats.csv")
    stats_md_path = summary_path.with_name(f"{base_name}__task_stats.md")

    total_collect_seconds = round(sum(_to_float(row.get("collect_elapsed_seconds")) for row in rows), 3)
    total_instruction_seconds = round(sum(_to_float(row.get("instruction_elapsed_seconds")) for row in rows), 3)
    total_export_seconds = round(sum(_to_float(row.get("export_elapsed_seconds")) for row in rows), 3)
    total_pipeline_seconds = round(sum(_to_float(row.get("total_elapsed_seconds")) for row in rows), 3)
    total_samples = int(sum(int(row.get("sample_count", 0) or 0) for row in rows))
    attempted_seed_total = sum(
        int(row.get("attempted_seed_count")) for row in rows if row.get("attempted_seed_count") not in (None, "")
    )
    attempted_seed_available = any(row.get("attempted_seed_count") not in (None, "") for row in rows)
    success_episode_total = int(sum(int(row.get("seed_success_episode_num", 0) or 0) for row in rows))
    overall_seed_success_rate = None
    if attempted_seed_available and attempted_seed_total > 0:
        overall_seed_success_rate = round(float(success_episode_total) / float(attempted_seed_total), 6)

    overview = {
        "task_config": run_payload.get("task_config"),
        "storage_setting": run_payload.get("storage_setting"),
        "phase": run_payload.get("phase"),
        "started_at": run_payload.get("started_at"),
        "finished_at": run_payload.get("finished_at"),
        "elapsed_seconds": run_payload.get("elapsed_seconds"),
        "task_num": len(rows),
        "collect_success_task_num": run_payload.get("collect_success_task_num", 0),
        "collect_failed_task_num": run_payload.get("collect_failed_task_num", 0),
        "instruction_success_task_num": run_payload.get("instruction_success_task_num", 0),
        "instruction_failed_task_num": run_payload.get("instruction_failed_task_num", 0),
        "export_success_task_num": run_payload.get("export_success_task_num", 0),
        "export_failed_task_num": run_payload.get("export_failed_task_num", 0),
        "complete_task_num": run_payload.get("complete_task_num", 0),
        "total_collect_seconds": total_collect_seconds,
        "total_instruction_seconds": total_instruction_seconds,
        "total_export_seconds": total_export_seconds,
        "total_pipeline_seconds": total_pipeline_seconds,
        "total_collect_hms": _format_seconds(total_collect_seconds),
        "total_instruction_hms": _format_seconds(total_instruction_seconds),
        "total_export_hms": _format_seconds(total_export_seconds),
        "total_pipeline_hms": _format_seconds(total_pipeline_seconds),
        "total_samples": total_samples,
        "success_episode_total": success_episode_total,
        "attempted_seed_total": (attempted_seed_total if attempted_seed_available else None),
        "overall_seed_success_rate": overall_seed_success_rate,
        "overall_seed_success_rate_percent": (
            None if overall_seed_success_rate is None else round(float(overall_seed_success_rate) * 100.0, 2)
        ),
    }

    _write_json(
        stats_json_path,
        {
            "overview": overview,
            "tasks": rows,
        },
    )

    csv_fields = [
        "task_name",
        "status",
        "collect_status",
        "instruction_status",
        "export_status",
        "requested_episode_num",
        "collected_episode_num",
        "seed_success_episode_num",
        "attempted_seed_count",
        "seed_success_rate_percent",
        "instruction_file_count",
        "collect_elapsed_seconds",
        "instruction_elapsed_seconds",
        "export_elapsed_seconds",
        "total_elapsed_seconds",
        "collect_elapsed_hms",
        "instruction_elapsed_hms",
        "export_elapsed_hms",
        "total_elapsed_hms",
        "sample_count",
        "object_search_count",
        "angle_delta_count",
        "memory_compression_vqa_count",
        "task_dir",
    ]
    _write_csv(stats_csv_path, rows, csv_fields)

    md_lines = [
        "# Rotate Collection Task Stats",
        "",
        "## Overview",
        "",
        f"- task_config: `{overview['task_config']}`",
        f"- storage_setting: `{overview['storage_setting']}`",
        f"- phase: `{overview['phase']}`",
        f"- started_at: `{overview['started_at']}`",
        f"- finished_at: `{overview['finished_at']}`",
        f"- task_num: `{overview['task_num']}`",
        f"- complete_task_num: `{overview['complete_task_num']}`",
        f"- collect_success_task_num: `{overview['collect_success_task_num']}`",
        f"- instruction_success_task_num: `{overview['instruction_success_task_num']}`",
        f"- export_success_task_num: `{overview['export_success_task_num']}`",
        f"- total_collect_time: `{overview['total_collect_hms']}`",
        f"- total_instruction_time: `{overview['total_instruction_hms']}`",
        f"- total_export_time: `{overview['total_export_hms']}`",
        f"- total_pipeline_time: `{overview['total_pipeline_hms']}`",
        f"- total_samples: `{overview['total_samples']}`",
        f"- overall_seed_success_rate: `{'-' if overview['overall_seed_success_rate_percent'] is None else overview['overall_seed_success_rate_percent']}`",
        "",
        "## Per Task",
        "",
        "| task | status | collect | instruction | export | episodes | success rate % | collect time | postprocess time | total time | samples |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        postprocess_seconds = _to_float(row.get("instruction_elapsed_seconds")) + _to_float(row.get("export_elapsed_seconds"))
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task_name", "")),
                    str(row.get("status", "")),
                    str(row.get("collect_status", "")),
                    str(row.get("instruction_status", "")),
                    str(row.get("export_status", "")),
                    f"{int(row.get('collected_episode_num', 0) or 0)}/{int(row.get('requested_episode_num', 0) or 0)}",
                    ("-" if row.get("seed_success_rate_percent") is None else f"{float(row['seed_success_rate_percent']):.2f}"),
                    str(row.get("collect_elapsed_hms", "")),
                    _format_seconds(postprocess_seconds),
                    str(row.get("total_elapsed_hms", "")),
                    str(int(row.get("sample_count", 0) or 0)),
                ]
            )
            + " |"
        )
    _write_text(stats_md_path, "\n".join(md_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase rotate whitelist pipeline: collect all simulation data first, then export VQA/annotated videos."
    )
    parser.add_argument("--task-config", type=str, default="demo_randomized_easy_ep200")
    parser.add_argument("--whitelist-file", type=str, default="task_config/rotate_task_whitelist.yml")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--max-seed-tries", type=int, default=1000)
    parser.add_argument("--clear-data", action="store_true")
    parser.add_argument("--export-vqa", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase", type=str, choices=("all", "collect", "export"), default="all")
    parser.add_argument("--inline-annotated-video", action="store_true")
    parser.add_argument("--inline-instructions", action="store_true")
    parser.add_argument("--summary-tag", type=str, default=None)
    args = parser.parse_args()

    task_config_path = REPO_ROOT / "task_config" / f"{args.task_config}.yml"
    config = _load_yaml(task_config_path)
    storage_setting = _storage_setting(args.task_config, config)
    whitelist = _load_whitelist(REPO_ROOT / args.whitelist_file)
    if args.task_limit is not None:
        whitelist = whitelist[: max(int(args.task_limit), 0)]

    data_root = REPO_ROOT / args.data_root
    if args.clear_data and data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    report_dir = data_root / "collection_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_name = f"collect_rotate_randomized_whitelist__{args.task_config}"
    if args.summary_tag is not None and str(args.summary_tag).strip():
        summary_name += f"__{_sanitize_tag(str(args.summary_tag))}"
    summary_path = report_dir / f"{summary_name}__summary.json"
    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    run_started_at = time.time()
    run_payload: dict[str, Any] = {
        "task_config": args.task_config,
        "whitelist_file": args.whitelist_file,
        "storage_setting": storage_setting,
        "gpu_id": str(args.gpu_id),
        "max_seed_tries": int(args.max_seed_tries) if args.max_seed_tries is not None else None,
        "clear_data": bool(args.clear_data),
        "export_vqa": bool(args.export_vqa),
        "continue_on_error": bool(args.continue_on_error),
        "resume": bool(args.resume),
        "phase": str(args.phase),
        "inline_annotated_video": bool(args.inline_annotated_video),
        "inline_instructions": bool(args.inline_instructions),
        "summary_tag": (None if args.summary_tag is None else str(args.summary_tag)),
        "started_at": _timestamp(),
        "tasks": [],
    }
    _write_json(summary_path, run_payload)

    def _record(task_payload: dict[str, Any]) -> None:
        _upsert_summary_task(run_payload, task_payload)
        run_payload["finished_at"] = _timestamp()
        run_payload["elapsed_seconds"] = round(float(time.time() - run_started_at), 3)
        _update_summary_counters(run_payload, export_vqa=bool(args.export_vqa))
        _write_json(summary_path, run_payload)
        _write_task_stats_reports(summary_path, run_payload)

    if args.phase in {"collect", "all"}:
        for task_name in whitelist:
            existing_report = _read_json_if_exists(
                _task_report_path(_task_dir(data_root, task_name, storage_setting))
            )
            task_payload = _build_task_payload(
                task_name=task_name,
                task_config=args.task_config,
                data_root=data_root,
                storage_setting=storage_setting,
                log_dir=log_dir,
                export_vqa=bool(args.export_vqa),
                existing_report=existing_report,
            )
            if args.resume and _is_collect_complete(task_payload):
                _write_task_report(data_root, task_payload)
                _record(task_payload)
                continue
            _record(task_payload)
            task_payload = _run_collect_task(
                task_name=task_name,
                task_config=args.task_config,
                data_root=data_root,
                storage_setting=storage_setting,
                gpu_id=str(args.gpu_id),
                max_seed_tries=args.max_seed_tries,
                log_dir=log_dir,
                export_vqa=bool(args.export_vqa),
                inline_annotated_video=bool(args.inline_annotated_video),
                inline_instructions=bool(args.inline_instructions),
            )
            _record(task_payload)
            if str(task_payload.get("collect_status", "")) == "failed" and not args.continue_on_error:
                return

    if args.phase in {"export", "all"}:
        for task_name in whitelist:
            existing_report = _read_json_if_exists(
                _task_report_path(_task_dir(data_root, task_name, storage_setting))
            )
            task_payload = _build_task_payload(
                task_name=task_name,
                task_config=args.task_config,
                data_root=data_root,
                storage_setting=storage_setting,
                log_dir=log_dir,
                export_vqa=bool(args.export_vqa),
                existing_report=existing_report,
            )
            if args.resume and _is_export_complete(task_payload, export_vqa=bool(args.export_vqa)):
                _write_task_report(data_root, task_payload)
                _record(task_payload)
                continue
            _record(task_payload)
            task_payload = _run_postprocess_task(
                task_name=task_name,
                task_config=args.task_config,
                data_root=data_root,
                storage_setting=storage_setting,
                log_dir=log_dir,
                export_vqa=bool(args.export_vqa),
            )
            _record(task_payload)
            if str(task_payload.get("status", "")) in {"instruction_failed", "export_failed"} and not args.continue_on_error:
                return


if __name__ == "__main__":
    main()
