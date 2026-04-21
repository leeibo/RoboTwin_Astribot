from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.utils.camera_visibility import image_u_to_yaw_error_rad  # noqa: E402
from envs.utils.rotate_theta import DEFAULT_STAGE1_THETA_UNIT_DEG  # noqa: E402
from script.rotate_vlm import (  # noqa: E402
    _JsonArrayWriter,
    _build_messages,
    _format_frame_field,
    _format_frame_summary,
    _format_object_name,
    _format_rotate_text,
    _format_uv_1000,
    _infer_primary_arm,
    _read_json,
    _resolve_search_phrase,
    _task_instruction,
    _task_subtask_think_clause,
    _to_vqa_rotation_pair,
    _wrap_to_180,
)
from script.rotate_vlm.snapshots import load_hdf5_episode_data  # noqa: E402


TASK_TYPE_NAME = "object_search_visibility_memory_v2"
DEFAULT_OUTPUT_DIR_NAME = "vlm_object_search_visibility_memory_v2"
DEFAULT_MAX_CONTEXT_FRAMES = 16
DEFAULT_TASK_WORKERS = max(1, min(4, int(os.cpu_count() or 1)))
DEFAULT_EPISODE_WORKERS = max(1, min(8, int(os.cpu_count() or 1)))
SEARCH_UNIT_DEG = int(DEFAULT_STAGE1_THETA_UNIT_DEG)
SEARCH_SIGN_EPS_DEG = 1.0
DIRECT_ALIGN_EPS_DEG = 1.0
SEARCH_CURRENT_HEADING_GAP_DEG = 20.0
SEARCH_HISTORY_HEADING_GAP_DEG = 15.0
DIRECT_HISTORY_HEADING_GAP_DEG = 12.0
DIRECT_ALIGN_BUCKET_EPS_DEG = 3.0
DIRECT_NEAR_BUCKET_EPS_DEG = 20.0
MAX_SEARCH_HISTORY_FRAMES = 3
MAX_DIRECT_HISTORY_FRAMES = 4


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
    seen: set[str] = set()
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


def _resolve_task_config_name(task_dir: Path, task_config: str | None) -> str | None:
    if task_config:
        return str(task_config)
    name = str(task_dir.name)
    if "__" in name:
        return name.split("__", 1)[0]
    return name or None


def _load_head_camera_fovy_deg(task_config_name: str | None) -> float:
    if not task_config_name:
        return 60.0
    config_path = REPO_ROOT / "task_config" / f"{task_config_name}.yml"
    camera_cfg_path = REPO_ROOT / "task_config" / "_camera_config.yml"
    if not config_path.exists() or not camera_cfg_path.exists():
        return 60.0
    task_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    camera_type = str((((task_payload.get("camera", {}) or {}).get("head_camera_type", "Large")))).strip() or "Large"
    camera_payload = yaml.safe_load(camera_cfg_path.read_text(encoding="utf-8")) or {}
    camera_entry = camera_payload.get(camera_type, {}) or {}
    try:
        return float(camera_entry.get("fovy", 60.0))
    except (TypeError, ValueError):
        return 60.0


def _collect_episode_pairs(save_dir: Path) -> list[tuple[int, Path, Path]]:
    metadata_dir = save_dir / "subtask_metadata"
    data_dir = save_dir / "data"
    pairs: list[tuple[int, Path, Path]] = []
    if not metadata_dir.exists() or not data_dir.exists():
        return pairs
    for metadata_path in sorted(metadata_dir.glob("episode*.json")):
        stem = metadata_path.stem
        suffix = stem.removeprefix("episode")
        if not suffix.isdigit():
            continue
        episode_idx = int(suffix)
        hdf5_path = data_dir / f"episode{episode_idx}.hdf5"
        if hdf5_path.exists():
            pairs.append((episode_idx, metadata_path, hdf5_path))
    return pairs


def _valid_uv(uv: list[float] | tuple[float, float] | None) -> list[float] | None:
    if not isinstance(uv, (list, tuple)) or len(uv) < 2:
        return None
    try:
        u = float(uv[0])
        v = float(uv[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    if u < 0.0 or v < 0.0:
        return None
    return [u, v]


def _annotation_target_key(annotation: dict[str, Any]) -> str | None:
    for key_name in ("search_target_keys", "action_target_keys"):
        keys = annotation.get(key_name, None) or []
        if keys:
            return str(keys[0])
    focus_key = annotation.get("focus_object_key", None)
    return None if focus_key is None else str(focus_key)


def _target_uv_for_annotation(annotation: dict[str, Any], target_key: str | None) -> list[float] | None:
    visible_keys = {str(key) for key in (annotation.get("visible_object_keys", []) or [])}
    if target_key is None:
        return _valid_uv(annotation.get("target_uv_norm", None))
    visible_map = annotation.get("visible_object_uv_map", {}) or {}
    if isinstance(visible_map, dict) and str(target_key) in visible_map:
        uv = _valid_uv(visible_map.get(str(target_key), None))
        if uv is not None:
            return uv
    target_uv = _valid_uv(annotation.get("target_uv_norm", None))
    if target_uv is not None and str(target_key) in visible_keys:
        return target_uv
    return None


def _annotation_has_visible_target(annotation: dict[str, Any], target_key: str | None) -> bool:
    if target_key is None:
        return False
    if _target_uv_for_annotation(annotation, target_key) is not None:
        return True
    visible_keys = {str(key) for key in (annotation.get("visible_object_keys", []) or [])}
    return str(target_key) in visible_keys


def _annotation_planned_delta_deg(annotation: dict[str, Any]) -> float:
    target_theta = annotation.get("camera_target_theta", None)
    if target_theta is None:
        return 0.0
    try:
        current_heading = float(annotation.get("waist_heading_deg", 0.0) or 0.0)
        return _wrap_to_180(math.degrees(float(target_theta)) - current_heading)
    except (TypeError, ValueError):
        return 0.0


def _frame_heading_deg(annotation: dict[str, Any]) -> float:
    try:
        return float(annotation.get("waist_heading_deg", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _iter_subtask_spans(annotations: list[dict[str, Any]]) -> list[tuple[int, int]]:
    if not annotations:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    prev_subtask = int(annotations[0].get("subtask", 0) or 0)
    for idx in range(1, len(annotations)):
        subtask = int(annotations[idx].get("subtask", 0) or 0)
        if subtask != prev_subtask:
            spans.append((start, idx))
            start = idx
            prev_subtask = subtask
    spans.append((start, len(annotations)))
    return spans


def _first_target_visible_rel_idx(subtask_annotations: list[dict[str, Any]], target_key: str | None) -> int | None:
    for rel_idx, annotation in enumerate(subtask_annotations):
        raw_stage = int(annotation.get("stage", 0) or 0)
        if raw_stage >= 3:
            continue
        if _annotation_has_visible_target(annotation, target_key):
            return int(rel_idx)
    return None


def _latest_target_history_rel_idx(subtask_annotations: list[dict[str, Any]], current_rel_idx: int, target_key: str | None) -> int | None:
    for rel_idx in range(current_rel_idx - 1, -1, -1):
        if _annotation_has_visible_target(subtask_annotations[rel_idx], target_key):
            return rel_idx
    return None


def _align_bucket_key(camera_delta_env_deg: int | float) -> str:
    abs_deg = abs(float(camera_delta_env_deg))
    if abs_deg <= float(DIRECT_ALIGN_BUCKET_EPS_DEG):
        return "aligned"
    if abs_deg <= float(DIRECT_NEAR_BUCKET_EPS_DEG):
        return "near"
    return "far"


def _select_sparse_history_indices(
    subtask_annotations: list[dict[str, Any]],
    candidate_indices: list[int],
    force_indices: list[int] | tuple[int, ...],
    max_items: int,
    min_heading_gap_deg: float,
) -> list[int]:
    if max_items <= 0 or not candidate_indices:
        return []
    selected: list[int] = []
    selected_set: set[int] = set()
    candidate_set = {int(idx) for idx in candidate_indices}

    for rel_idx in sorted({int(idx) for idx in force_indices if int(idx) in candidate_set}):
        if len(selected) >= int(max_items):
            break
        selected.append(int(rel_idx))
        selected_set.add(int(rel_idx))

    for rel_idx in reversed(sorted(candidate_set)):
        if len(selected) >= int(max_items):
            break
        if int(rel_idx) in selected_set:
            continue
        heading_deg = _frame_heading_deg(subtask_annotations[int(rel_idx)])
        is_novel = True
        for selected_idx in selected:
            selected_heading_deg = _frame_heading_deg(subtask_annotations[int(selected_idx)])
            if abs(_wrap_to_180(heading_deg - selected_heading_deg)) < float(min_heading_gap_deg):
                is_novel = False
                break
        if not is_novel:
            continue
        selected.append(int(rel_idx))
        selected_set.add(int(rel_idx))

    for rel_idx in reversed(sorted(candidate_set)):
        if len(selected) >= int(max_items):
            break
        if int(rel_idx) in selected_set:
            continue
        selected.append(int(rel_idx))
        selected_set.add(int(rel_idx))

    return sorted(selected[-int(max_items):])


def _search_direction_sign(subtask_annotations: list[dict[str, Any]], current_rel_idx: int) -> int:
    current = subtask_annotations[current_rel_idx]
    planned_delta = _annotation_planned_delta_deg(current)
    if abs(planned_delta) >= SEARCH_SIGN_EPS_DEG:
        return 1 if planned_delta > 0.0 else -1

    current_heading = _frame_heading_deg(current)
    for next_rel_idx in range(current_rel_idx + 1, len(subtask_annotations)):
        next_heading = _frame_heading_deg(subtask_annotations[next_rel_idx])
        diff = _wrap_to_180(next_heading - current_heading)
        if abs(diff) >= SEARCH_SIGN_EPS_DEG:
            return 1 if diff > 0.0 else -1

    for prev_rel_idx in range(current_rel_idx - 1, -1, -1):
        prev_heading = _frame_heading_deg(subtask_annotations[prev_rel_idx])
        diff = _wrap_to_180(current_heading - prev_heading)
        if abs(diff) >= SEARCH_SIGN_EPS_DEG:
            return 1 if diff > 0.0 else -1

    return 1


def _current_anchor_indices(
    subtask_annotations: list[dict[str, Any]],
    target_key: str | None,
    image_w: int,
    image_h: int,
    fovy_deg: float,
) -> tuple[list[int], int | None]:
    valid_indices = [
        int(rel_idx)
        for rel_idx, annotation in enumerate(subtask_annotations)
        if int(annotation.get("stage", 0) or 0) < 3
    ]
    if not valid_indices:
        return [], None

    first_target_rel_idx = _first_target_visible_rel_idx(subtask_annotations, target_key)
    anchors: list[int] = []
    anchor_set: set[int] = set()

    search_last_heading_deg: float | None = None
    search_last_sign: int | None = None
    direct_last_heading_deg: float | None = None
    direct_last_bucket: str | None = None
    direct_last_visible: bool | None = None

    search_end_rel_idx = valid_indices[-1] if first_target_rel_idx is None else max(int(first_target_rel_idx) - 1, valid_indices[0])

    def _keep(rel_idx: int) -> None:
        value = int(rel_idx)
        if value in anchor_set:
            return
        anchor_set.add(value)
        anchors.append(value)

    for rel_idx in valid_indices:
        annotation = subtask_annotations[int(rel_idx)]
        heading_deg = _frame_heading_deg(annotation)

        if first_target_rel_idx is None or int(rel_idx) < int(first_target_rel_idx):
            direction_sign = _search_direction_sign(subtask_annotations, int(rel_idx))
            keep = False
            if search_last_heading_deg is None:
                keep = True
            elif abs(_wrap_to_180(float(heading_deg) - float(search_last_heading_deg))) >= float(SEARCH_CURRENT_HEADING_GAP_DEG):
                keep = True
            elif search_last_sign is not None and int(direction_sign) != int(search_last_sign):
                keep = True
            elif int(rel_idx) == int(search_end_rel_idx):
                keep = True
            if keep:
                _keep(int(rel_idx))
                search_last_heading_deg = float(heading_deg)
                search_last_sign = int(direction_sign)
            continue

        current_visible = _annotation_has_visible_target(annotation, target_key)
        latest_target_rel_idx = None if current_visible else _latest_target_history_rel_idx(
            subtask_annotations,
            int(rel_idx),
            target_key,
        )
        evidence_annotation = annotation if current_visible else (
            None if latest_target_rel_idx is None else subtask_annotations[int(latest_target_rel_idx)]
        )
        camera_delta_env_deg = _direct_camera_delta_env_deg(
            current_annotation=annotation,
            evidence_annotation=evidence_annotation,
            target_key=target_key,
            image_w=int(image_w),
            image_h=int(image_h),
            fovy_deg=float(fovy_deg),
        )
        delta_bucket = _align_bucket_key(camera_delta_env_deg)
        keep = False
        if direct_last_heading_deg is None:
            keep = True
        elif int(rel_idx) == int(first_target_rel_idx):
            keep = True
        elif delta_bucket != direct_last_bucket:
            keep = True
        elif direct_last_visible is not None and bool(current_visible) != bool(direct_last_visible):
            keep = True
        elif int(rel_idx) == int(valid_indices[-1]):
            keep = True
        if keep:
            _keep(int(rel_idx))
            direct_last_heading_deg = float(heading_deg)
            direct_last_bucket = str(delta_bucket)
            direct_last_visible = bool(current_visible)

    _keep(int(valid_indices[-1]))
    return sorted(anchor_set), first_target_rel_idx


def _history_selection_indices_v2(
    subtask_annotations: list[dict[str, Any]],
    current_rel_idx: int,
    history_limit: int,
    target_key: str | None,
    anchor_indices: list[int],
    first_target_rel_idx: int | None,
) -> list[int]:
    if history_limit <= 0 or current_rel_idx <= 0:
        return []

    current_rel_idx = int(current_rel_idx)
    if first_target_rel_idx is None or int(current_rel_idx) < int(first_target_rel_idx):
        candidate_indices = [
            int(rel_idx)
            for rel_idx in anchor_indices
            if int(rel_idx) < int(current_rel_idx) and (first_target_rel_idx is None or int(rel_idx) < int(first_target_rel_idx))
        ]
        force_indices: list[int] = []
        if candidate_indices:
            force_indices.append(int(candidate_indices[0]))
            force_indices.append(int(candidate_indices[-1]))
        return _select_sparse_history_indices(
            subtask_annotations=subtask_annotations,
            candidate_indices=sorted(set(candidate_indices)),
            force_indices=force_indices,
            max_items=min(int(history_limit), int(MAX_SEARCH_HISTORY_FRAMES)),
            min_heading_gap_deg=float(SEARCH_HISTORY_HEADING_GAP_DEG),
        )

    candidate_set = {int(rel_idx) for rel_idx in anchor_indices if int(rel_idx) < int(current_rel_idx)}
    latest_target_rel_idx = _latest_target_history_rel_idx(subtask_annotations, int(current_rel_idx), target_key)
    if latest_target_rel_idx is not None:
        candidate_set.add(int(latest_target_rel_idx))
    if first_target_rel_idx is not None and int(first_target_rel_idx) < int(current_rel_idx):
        candidate_set.add(int(first_target_rel_idx))
    candidate_indices = sorted(candidate_set)
    force_indices: list[int] = []
    if candidate_indices:
        force_indices.append(int(candidate_indices[-1]))
    if latest_target_rel_idx is not None:
        force_indices.append(int(latest_target_rel_idx))
    if first_target_rel_idx is not None and int(first_target_rel_idx) < int(current_rel_idx):
        force_indices.append(int(first_target_rel_idx))
    return _select_sparse_history_indices(
        subtask_annotations=subtask_annotations,
        candidate_indices=candidate_indices,
        force_indices=force_indices,
        max_items=min(int(history_limit), int(MAX_DIRECT_HISTORY_FRAMES)),
        min_heading_gap_deg=float(DIRECT_HISTORY_HEADING_GAP_DEG),
    )


def _prompt_evidence(
    prompt_annotations: list[dict[str, Any]],
    target_key: str | None,
) -> tuple[int | None, int | None, list[float] | None, bool]:
    for prompt_idx in range(len(prompt_annotations) - 1, -1, -1):
        annotation = prompt_annotations[prompt_idx]
        uv = _target_uv_for_annotation(annotation, target_key)
        if uv is None and not _annotation_has_visible_target(annotation, target_key):
            continue
        frame_idx = int(annotation.get("frame_idx", prompt_idx))
        return prompt_idx + 1, frame_idx, uv, prompt_idx < len(prompt_annotations) - 1
    return None, None, None, False


def _direct_camera_delta_env_deg(
    current_annotation: dict[str, Any],
    evidence_annotation: dict[str, Any] | None,
    target_key: str | None,
    image_w: int,
    image_h: int,
    fovy_deg: float,
) -> int:
    current_uv = _target_uv_for_annotation(current_annotation, target_key)
    if current_uv is not None:
        yaw_error_rad = image_u_to_yaw_error_rad(
            current_uv[0],
            image_w=int(image_w),
            image_h=int(image_h),
            fovy_rad=math.radians(float(fovy_deg)),
        )
        yaw_error_deg = _wrap_to_180(math.degrees(float(yaw_error_rad)))
        if abs(yaw_error_deg) < DIRECT_ALIGN_EPS_DEG:
            return 0
        return int(round(yaw_error_deg))

    if evidence_annotation is not None:
        diff = _wrap_to_180(_frame_heading_deg(evidence_annotation) - _frame_heading_deg(current_annotation))
        return int(round(diff))

    return 0


def _build_images(
    output_dir: Path,
    episode_idx: int,
    frames: list[Any],
    frame_indices: list[int],
) -> list[str]:
    from PIL import Image

    image_dir = output_dir / "images" / f"episode{int(episode_idx)}"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for frame_idx in frame_indices:
        path = image_dir / f"frame_{int(frame_idx):04d}.png"
        if not path.exists():
            Image.fromarray(frames[int(frame_idx)]).save(path)
        paths.append(str(path))
    return paths


def _build_user_prompt(metadata: dict[str, Any], subtask_id: int, prompt_image_count: int) -> str:
    instruction = _task_instruction(metadata, subtask_id)
    image_tokens = "".join("<image>" for _ in range(int(prompt_image_count)))
    return (
        f'{image_tokens}Your task is: "{instruction}" '
        "The input images are ordered from earliest to latest, and the last image is the current view. "
        "Please think about the next action and output it. "
        "Your response should be in the format of: "
        "<think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>."
    )


def _build_assistant_response(
    metadata: dict[str, Any],
    current_annotation: dict[str, Any],
    prompt_annotations: list[dict[str, Any]],
    evidence_prompt_index: int | None,
    evidence_uv: list[float] | None,
    evidence_from_history: bool,
    camera_delta_env_deg: int,
) -> str:
    subtask_id = int(current_annotation.get("subtask", 0) or 0)
    phrase = _resolve_search_phrase(
        metadata,
        type(
            "_TmpSlot",
            (),
            {
                "subtask_id": subtask_id,
                "target_key": lambda self=None: _annotation_target_key(current_annotation),
                "target_keys": lambda self=None: [
                    str(key)
                    for key in (current_annotation.get("search_target_keys", []) or [])
                ]
                or [str(key) for key in (current_annotation.get("action_target_keys", []) or [])],
            },
        )(),
    )
    task_clause = _task_subtask_think_clause(metadata, subtask_id)
    memory_summary = _format_frame_summary(len(prompt_annotations))
    camera_field = _format_rotate_text((camera_delta_env_deg, 0))
    has_target = evidence_prompt_index is not None
    primary_arm = _infer_primary_arm(metadata)

    if has_target and evidence_from_history and evidence_uv is not None:
        evidence_text = (
            f"The {phrase} was found in frame {int(evidence_prompt_index)} "
            f"at {_format_uv_1000(evidence_uv)}."
        )
    elif has_target and evidence_uv is not None:
        evidence_text = f"The {phrase} is visible in the current view at {_format_uv_1000(evidence_uv)}."
    elif has_target:
        evidence_text = f"The {phrase} is supported by the sampled memory."
    else:
        direction_text = "left" if int(camera_delta_env_deg) > 0 else "right"
        evidence_text = (
            f"The {phrase} does not appear in the sampled frames. "
            f"Continue searching by rotating {direction_text} by {SEARCH_UNIT_DEG} degrees."
        )

    carry_text = ""
    carried_keys = [str(key) for key in (current_annotation.get("carried_object_keys", []) or [])]
    if carried_keys and primary_arm is not None:
        carry_text = f" The {_format_object_name(metadata, carried_keys[-1])} is currently held by the {primary_arm} hand."

    info_text = "Info sufficient." if has_target else "Info incomplete."
    think = (
        f"{memory_summary} "
        f"{task_clause} "
        f"The target object is the {phrase}. "
        f"{evidence_text} "
        f"{info_text}"
        f"{carry_text} "
        f"Next: {camera_field}."
    ).strip()
    info_field = "1" if has_target else "0"
    frame_field = _format_frame_field([int(evidence_prompt_index)] if evidence_prompt_index is not None else [])
    return (
        f"<think>{think}</think>"
        f"<info>{info_field}</info>"
        f"<frame>{frame_field}</frame>"
        f"<camera>{camera_field}</camera>"
        f"<action></action>"
    )


def _sample_dedup_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    metadata = sample.get("metadata", {}) or {}
    return (
        int(metadata.get("subtask_id", 0) or 0),
        str(metadata.get("mode", "")),
        None if metadata.get("target_key", None) is None else str(metadata.get("target_key")),
        int(metadata.get("camera_delta_deg", 0) or 0),
        None if metadata.get("evidence_frame_idx", None) is None else int(metadata.get("evidence_frame_idx")),
        bool(metadata.get("current_target_visible", False)),
    )


def _dedup_episode_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(samples) <= 1:
        return list(samples)

    latest_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for sample in samples:
        latest_by_key[_sample_dedup_key(sample)] = sample

    return sorted(
        latest_by_key.values(),
        key=lambda item: (
            int(((item.get("metadata", {}) or {}).get("subtask_id", 0) or 0)),
            int(((item.get("metadata", {}) or {}).get("current_frame_idx", 0) or 0)),
            (
                -1
                if ((item.get("metadata", {}) or {}).get("evidence_frame_idx", None)) is None
                else int((item.get("metadata", {}) or {}).get("evidence_frame_idx"))
            ),
            str(((item.get("metadata", {}) or {}).get("mode", ""))),
        ),
    )


def _episode_samples_payload(
    args: tuple[str, str, int, str, str, int, float],
) -> dict[str, Any]:
    save_dir, output_dir_name, episode_idx, metadata_path, hdf5_path, max_context_frames, fovy_deg = args
    save_path = Path(save_dir)
    output_dir = save_path / output_dir_name
    metadata = _read_json(Path(metadata_path))
    frames, _, _, _, _ = load_hdf5_episode_data(str(hdf5_path))
    annotations = [dict(item) for item in (metadata.get("frame_annotations", []) or [])]

    if not frames or not annotations:
        return {
            "episode_idx": int(episode_idx),
            "sample_count": 0,
            "search_count": 0,
            "direct_count": 0,
            "samples": [],
        }

    image_h = int(frames[0].shape[0])
    image_w = int(frames[0].shape[1])
    history_limit = max(int(max_context_frames) - 1, 0)
    samples: list[dict[str, Any]] = []
    search_count = 0
    direct_count = 0

    for span_start, span_end in _iter_subtask_spans(annotations):
        subtask_annotations = annotations[span_start:span_end]
        subtask_target_key = next(
            (
                _annotation_target_key(annotation)
                for annotation in subtask_annotations
                if int(annotation.get("stage", 0) or 0) < 3
            ),
            None,
        )
        anchor_indices, first_target_rel_idx = _current_anchor_indices(
            subtask_annotations=subtask_annotations,
            target_key=subtask_target_key,
            image_w=int(image_w),
            image_h=int(image_h),
            fovy_deg=float(fovy_deg),
        )
        for current_rel_idx in anchor_indices:
            current_annotation = subtask_annotations[int(current_rel_idx)]
            raw_stage = int(current_annotation.get("stage", 0) or 0)
            target_key = _annotation_target_key(current_annotation) or subtask_target_key
            history_rel_indices = _history_selection_indices_v2(
                subtask_annotations=subtask_annotations,
                current_rel_idx=int(current_rel_idx),
                history_limit=int(history_limit),
                target_key=target_key,
                anchor_indices=anchor_indices,
                first_target_rel_idx=first_target_rel_idx,
            )
            prompt_annotations = [subtask_annotations[idx] for idx in history_rel_indices] + [current_annotation]
            prompt_frame_indices = [int(item.get("frame_idx", 0)) for item in prompt_annotations]
            evidence_prompt_index, evidence_frame_idx, evidence_uv, evidence_from_history = _prompt_evidence(
                prompt_annotations=prompt_annotations,
                target_key=target_key,
            )
            evidence_annotation = None
            if evidence_prompt_index is not None:
                evidence_annotation = prompt_annotations[int(evidence_prompt_index) - 1]

            if evidence_prompt_index is not None:
                direct_count += 1
                camera_delta_env_deg = _direct_camera_delta_env_deg(
                    current_annotation=current_annotation,
                    evidence_annotation=evidence_annotation,
                    target_key=target_key,
                    image_w=image_w,
                    image_h=image_h,
                    fovy_deg=fovy_deg,
                )
                mode = "direct_locate"
            else:
                search_count += 1
                direction_sign = _search_direction_sign(subtask_annotations, current_rel_idx)
                camera_delta_env_deg = int(direction_sign * SEARCH_UNIT_DEG)
                mode = "search_unit"

            images = _build_images(
                output_dir=output_dir,
                episode_idx=episode_idx,
                frames=frames,
                frame_indices=prompt_frame_indices,
            )
            user_prompt = _build_user_prompt(
                metadata=metadata,
                subtask_id=int(current_annotation.get("subtask", 0) or 0),
                prompt_image_count=len(prompt_frame_indices),
            )
            assistant_prompt = _build_assistant_response(
                metadata=metadata,
                current_annotation=current_annotation,
                prompt_annotations=prompt_annotations,
                evidence_prompt_index=evidence_prompt_index,
                evidence_uv=evidence_uv,
                evidence_from_history=bool(evidence_from_history),
                camera_delta_env_deg=int(camera_delta_env_deg),
            )
            samples.append(
                {
                    "images": images,
                    "messages": _build_messages(user_content=user_prompt, assistant_content=assistant_prompt),
                    "metadata": {
                        "episode_idx": int(episode_idx),
                        "task_name": str(metadata.get("task_name", "")),
                        "task_type": TASK_TYPE_NAME,
                        "subtask_id": int(current_annotation.get("subtask", 0) or 0),
                        "raw_stage": int(raw_stage),
                        "mode": str(mode),
                        "current_frame_idx": int(current_annotation.get("frame_idx", 0) or 0),
                        "prompt_frame_indices": list(prompt_frame_indices),
                        "prompt_image_count": int(len(prompt_frame_indices)),
                        "target_key": None if target_key is None else str(target_key),
                        "current_target_visible": bool(_annotation_has_visible_target(current_annotation, target_key)),
                        "prompt_has_target": bool(evidence_prompt_index is not None),
                        "evidence_from_history": bool(evidence_from_history),
                        "evidence_frame_idx": evidence_frame_idx,
                        "evidence_prompt_index": evidence_prompt_index,
                        "evidence_uv_norm": evidence_uv,
                        "camera_delta_deg": int(_to_vqa_rotation_pair((camera_delta_env_deg, 0))[0]),
                        "search_unit_deg": int(SEARCH_UNIT_DEG),
                        "sparse_version": "v2",
                        "history_frame_budget": (
                            int(MAX_SEARCH_HISTORY_FRAMES)
                            if str(mode) == "search_unit"
                            else int(MAX_DIRECT_HISTORY_FRAMES)
                        ),
                    },
                }
            )

    samples = _dedup_episode_samples(samples)
    search_count = sum(1 for sample in samples if str(((sample.get("metadata", {}) or {}).get("mode", ""))) == "search_unit")
    direct_count = sum(1 for sample in samples if str(((sample.get("metadata", {}) or {}).get("mode", ""))) == "direct_locate")

    return {
        "episode_idx": int(episode_idx),
        "sample_count": int(len(samples)),
        "search_count": int(search_count),
        "direct_count": int(direct_count),
        "samples": samples,
    }


def export_single_task(
    save_dir: str,
    task_config_name: str | None = None,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    max_context_frames: int = DEFAULT_MAX_CONTEXT_FRAMES,
    num_workers: int | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    save_path = Path(save_dir)
    output_dir = save_path / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{TASK_TYPE_NAME}.json"
    writer = _JsonArrayWriter(output_path, overwrite=overwrite)

    episode_pairs = _collect_episode_pairs(save_path)
    worker_count = _resolve_worker_count(num_workers, DEFAULT_EPISODE_WORKERS, upper_bound=len(episode_pairs) or 1)
    fovy_deg = _load_head_camera_fovy_deg(task_config_name)
    job_args = [
        (
            str(save_path),
            str(output_dir_name),
            int(episode_idx),
            str(metadata_path),
            str(hdf5_path),
            int(max_context_frames),
            float(fovy_deg),
        )
        for episode_idx, metadata_path, hdf5_path in episode_pairs
    ]

    total_samples = 0
    total_search = 0
    total_direct = 0
    episode_count = 0
    try:
        if worker_count <= 1 or len(job_args) <= 1:
            payload_iter = (_episode_samples_payload(args) for args in job_args)
        else:
            executor = ProcessPoolExecutor(max_workers=int(worker_count))
            payload_iter = executor.map(_episode_samples_payload, job_args, chunksize=1)
        try:
            for payload in payload_iter:
                episode_count += 1
                total_samples += int(payload.get("sample_count", 0) or 0)
                total_search += int(payload.get("search_count", 0) or 0)
                total_direct += int(payload.get("direct_count", 0) or 0)
                for sample in payload.get("samples", []) or []:
                    writer.append(sample)
        finally:
            if worker_count > 1 and len(job_args) > 1:
                executor.shutdown(wait=True, cancel_futures=False)
        writer.commit()
    except Exception:
        writer.abort()
        raise

    return {
        "status": "ok",
        "task_dir": str(save_path),
        "output_dir": str(output_dir),
        "output_path": str(output_path),
        "task_type_counts": {TASK_TYPE_NAME: int(total_samples)},
        "sample_count": int(total_samples),
        "search_count": int(total_search),
        "direct_count": int(total_direct),
        "episode_count": int(episode_count),
        "worker_count": int(worker_count),
        "task_config_name": task_config_name,
        "head_camera_fovy_deg": float(fovy_deg),
    }


def _export_single_task_job(job: tuple[str, str, str | None, str, int]) -> tuple[str, dict[str, Any]]:
    task_name, task_dir, task_config_name, output_dir_name, episode_workers = job
    started_at = time.time()
    summary = export_single_task(
        save_dir=task_dir,
        task_config_name=task_config_name,
        output_dir_name=output_dir_name,
        num_workers=episode_workers,
        overwrite=True,
    )
    summary["elapsed_seconds"] = round(float(time.time() - started_at), 3)
    return task_name, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export framewise rotate object-search expansion samples. "
            "Samples are derived from raw frame annotations and stored in a dedicated output directory."
        )
    )
    parser.add_argument("--save-dir", type=str, default=None, help="Optional single task directory to export.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--whitelist-file", type=str, default="task_config/rotate_task_whitelist.yml")
    parser.add_argument("--task-config", type=str, default=None, help="Task config prefix, e.g. demo_randomized_easy_ep200")
    parser.add_argument("--output-dir-name", type=str, default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--max-context-frames", type=int, default=DEFAULT_MAX_CONTEXT_FRAMES)
    parser.add_argument("--task-workers", type=int, default=None)
    parser.add_argument("--episode-workers", type=int, default=None)
    args = parser.parse_args()

    if args.save_dir:
        task_dir = Path(args.save_dir)
        task_config_name = _resolve_task_config_name(task_dir, args.task_config)
        summary = export_single_task(
            save_dir=str(task_dir),
            task_config_name=task_config_name,
            output_dir_name=str(args.output_dir_name),
            max_context_frames=int(args.max_context_frames),
            num_workers=args.episode_workers,
            overwrite=True,
        )
        if args.summary_path:
            summary_path = Path(args.summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[ok] {task_dir.name}: {summary['task_type_counts']} "
            f"(search={summary['search_count']}, direct={summary['direct_count']}, workers={summary['worker_count']})"
        )
        return

    data_root = Path(args.data_root)
    whitelist = _load_whitelist(Path(args.whitelist_file))
    pending_jobs: list[tuple[str, str, str | None, str, int]] = []
    summary: dict[str, Any] = {}
    for task_name in whitelist:
        task_dir = _discover_task_dir(data_root=data_root, task_name=task_name, task_config=args.task_config)
        if task_dir is None:
            summary[task_name] = {"status": "missing_task_dir"}
            print(f"[skip] {task_name}: task dir not found")
            continue
        task_config_name = _resolve_task_config_name(task_dir, args.task_config)
        pending_jobs.append((task_name, str(task_dir), task_config_name, str(args.output_dir_name), 1))

    task_workers = _resolve_worker_count(args.task_workers, DEFAULT_TASK_WORKERS, upper_bound=len(pending_jobs) or 1)
    episode_workers = _resolve_worker_count(args.episode_workers, _default_episode_workers(task_workers))
    pending_jobs = [
        (task_name, task_dir, task_config_name, output_dir_name, episode_workers)
        for task_name, task_dir, task_config_name, output_dir_name, _ in pending_jobs
    ]

    if pending_jobs:
        print(
            f"[start] exporting {len(pending_jobs)} task(s) "
            f"with task_workers={task_workers}, episode_workers={episode_workers}, output_dir={args.output_dir_name}"
        )

    if task_workers <= 1 or len(pending_jobs) <= 1:
        for job in pending_jobs:
            task_name = job[0]
            try:
                _, task_summary = _export_single_task_job(job)
                summary[task_name] = task_summary
                print(
                    f"[ok] {task_name}: {task_summary['task_type_counts']} "
                    f"(search={task_summary['search_count']}, direct={task_summary['direct_count']}, {task_summary['elapsed_seconds']}s)"
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
            future_map = {executor.submit(_export_single_task_job, job): job for job in pending_jobs}
            for future in as_completed(future_map):
                task_name, task_dir, _, _, _ = future_map[future]
                try:
                    finished_task_name, task_summary = future.result()
                    summary[finished_task_name] = task_summary
                    print(
                        f"[ok] {finished_task_name}: {task_summary['task_type_counts']} "
                        f"(search={task_summary['search_count']}, direct={task_summary['direct_count']}, {task_summary['elapsed_seconds']}s)"
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
