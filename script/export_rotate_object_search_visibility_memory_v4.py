from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.export_rotate_object_search_visibility_memory_v2 import (  # noqa: E402
    DEFAULT_EPISODE_WORKERS,
    DEFAULT_MAX_CONTEXT_FRAMES,
    DEFAULT_TASK_WORKERS,
    MAX_DIRECT_HISTORY_FRAMES,
    MAX_SEARCH_HISTORY_FRAMES,
    SEARCH_UNIT_DEG,
    _annotation_has_visible_target,
    _annotation_target_key,
    _build_assistant_response,
    _build_images,
    _build_user_prompt,
    _current_anchor_indices,
    _default_episode_workers,
    _direct_camera_delta_env_deg,
    _discover_task_dir,
    _history_selection_indices_v2,
    _iter_subtask_spans,
    _load_head_camera_fovy_deg,
    _load_whitelist,
    _prompt_evidence,
    _resolve_task_config_name,
    _resolve_worker_count,
    _search_direction_sign,
)
from script.rotate_vlm import (  # noqa: E402
    DEFAULT_ACTION_CHUNK_SIZE,
    _JsonArrayWriter,
    _build_messages,
    _build_object_search_user_prompt,
    _build_stage_action_chunk,
    _build_stage_action_chunk_text,
    _metadata_action_pairs,
    _read_json,
    _render_object_search_response,
    _search_camera_delta,
    _to_vqa_rotation_pair,
)
from script.rotate_vlm.models import EpisodeContext, MemorySlot  # noqa: E402
from script.rotate_vlm.snapshots import (  # noqa: E402
    _build_memory_slots,
    _collect_annotation_segments,
    _make_memory_slot,
    _make_snapshot,
    compress_memory_slots,
    load_hdf5_episode_data,
)


TASK_TYPE_NAME = "object_search_visibility_memory_v4"
DEFAULT_OUTPUT_DIR_NAME = "vlm_object_search_visibility_memory_v4"


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


def _filter_episode_pairs(
    episode_pairs: list[tuple[int, Path, Path]],
    episode_indices: list[int] | None = None,
    max_episodes: int | None = None,
) -> list[tuple[int, Path, Path]]:
    result = list(episode_pairs)
    if episode_indices:
        allowed = {int(idx) for idx in episode_indices}
        result = [pair for pair in result if int(pair[0]) in allowed]
    if max_episodes is not None:
        result = result[: max(int(max_episodes), 0)]
    return result


def _sample_uniform_action_frame_indices(
    frame_indices: list[int],
    action_chunk_size: int,
) -> tuple[list[int], int, int]:
    chunk_size = max(int(action_chunk_size), 1)
    clean_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not clean_indices:
        return [], 0, chunk_size

    segment_len = len(clean_indices)
    if segment_len >= chunk_size:
        sampled = [
            clean_indices[min(int(i * segment_len // chunk_size), segment_len - 1)]
            for i in range(chunk_size)
        ]
        return sampled, chunk_size, 0

    return list(clean_indices), segment_len, chunk_size - segment_len


def _stage12_action_chunk_metadata_by_frame(
    annotations: list[dict[str, Any]],
    action_chunk_size: int,
) -> dict[int, dict[str, Any]]:
    chunk_size = max(int(action_chunk_size), 1)
    metadata_by_frame: dict[int, dict[str, Any]] = {}
    for seg_start, seg_end in _collect_annotation_segments(annotations):
        segment = annotations[seg_start:seg_end]
        if not segment:
            continue
        stage = int(segment[0].get("stage", 0) or 0)
        if stage not in (1, 2):
            continue
        segment_frame_indices = [int(item.get("frame_idx", 0)) for item in segment]
        sampled_frame_indices, actual_size, pad_count = _sample_uniform_action_frame_indices(
            segment_frame_indices,
            action_chunk_size=chunk_size,
        )
        segment_start_frame_idx = int(segment_frame_indices[0])
        segment_end_frame_idx = int(segment_frame_indices[-1])
        payload = {
            "action_chunk_frame_indices": list(sampled_frame_indices),
            "action_chunk_actual_size": int(actual_size),
            "action_chunk_pad_count": int(pad_count),
            "stage12_segment_start_frame_idx": int(segment_start_frame_idx),
            "stage12_segment_end_frame_idx": int(segment_end_frame_idx),
            "stage12_segment_start_annotation_index": int(seg_start),
            "stage12_segment_end_annotation_index": int(seg_end),
        }
        for frame_idx in segment_frame_indices:
            metadata_by_frame[int(frame_idx)] = dict(payload)
    return metadata_by_frame


def _build_stage12_action_chunk(
    context: EpisodeContext,
    current_annotation: dict[str, Any],
    action_metadata: dict[str, Any] | None,
) -> list[list[float]] | None:
    if not action_metadata:
        return None
    action_frame_indices = [int(idx) for idx in (action_metadata.get("action_chunk_frame_indices", []) or [])]
    if not action_frame_indices:
        return None
    current_slot = _make_memory_slot(
        slot_idx=0,
        annotation=dict(current_annotation),
        roles=[f"stage{int(current_annotation.get('stage', 0) or 0)}_action_chunk"],
        action_chunk_frame_indices=action_frame_indices,
        action_chunk_actual_size=int(action_metadata.get("action_chunk_actual_size", len(action_frame_indices)) or 0),
        action_chunk_pad_count=int(action_metadata.get("action_chunk_pad_count", 0) or 0),
    )
    return _build_stage_action_chunk(context, current_slot)


def _assistant_response_with_action_chunk(
    assistant_content: str,
    action_chunk: list[list[float]] | None,
) -> str:
    if action_chunk is None:
        return str(assistant_content)
    action_text = _build_stage_action_chunk_text(action_chunk)
    content = str(assistant_content)
    if "<action></action>" in content:
        return content.replace("<action></action>", f"<action>{action_text}</action>", 1)
    return content


def _stage12_samples(
    output_dir: Path,
    episode_idx: int,
    metadata: dict[str, Any],
    context: EpisodeContext,
    annotations: list[dict[str, Any]],
    max_context_frames: int,
    action_chunk_size: int,
    fovy_deg: float,
) -> tuple[list[dict[str, Any]], int, int]:
    frames = context.frames
    if not frames or not annotations:
        return [], 0, 0

    image_h = int(frames[0].shape[0])
    image_w = int(frames[0].shape[1])
    history_limit = max(int(max_context_frames) - 1, 0)
    action_metadata_by_frame = _stage12_action_chunk_metadata_by_frame(
        annotations=annotations,
        action_chunk_size=int(action_chunk_size),
    )
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

            action_metadata = action_metadata_by_frame.get(int(current_annotation.get("frame_idx", 0) or 0), None)
            stage_action_chunk = _build_stage12_action_chunk(
                context=context,
                current_annotation=current_annotation,
                action_metadata=action_metadata,
            )
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
            assistant_prompt = _assistant_response_with_action_chunk(
                assistant_prompt,
                action_chunk=stage_action_chunk,
            )
            sample = {
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
                    "sparse_version": "v4",
                    "history_frame_budget": (
                        int(MAX_SEARCH_HISTORY_FRAMES)
                        if str(mode) == "search_unit"
                        else int(MAX_DIRECT_HISTORY_FRAMES)
                    ),
                },
            }
            if stage_action_chunk is not None and action_metadata is not None:
                sample["action"] = stage_action_chunk
                sample["metadata"].update(
                    {
                        "action_chunk_size": int(max(int(action_chunk_size), 1)),
                        "action_chunk_frame_indices": list(action_metadata.get("action_chunk_frame_indices", []) or []),
                        "action_chunk_actual_size": int(action_metadata.get("action_chunk_actual_size", 0) or 0),
                        "action_chunk_pad_count": int(action_metadata.get("action_chunk_pad_count", 0) or 0),
                        "stage12_segment_start_frame_idx": int(action_metadata.get("stage12_segment_start_frame_idx", 0) or 0),
                        "stage12_segment_end_frame_idx": int(action_metadata.get("stage12_segment_end_frame_idx", 0) or 0),
                        "stage12_segment_start_annotation_index": int(
                            action_metadata.get("stage12_segment_start_annotation_index", 0) or 0
                        ),
                        "stage12_segment_end_annotation_index": int(
                            action_metadata.get("stage12_segment_end_annotation_index", 0) or 0
                        ),
                    }
                )
            samples.append(sample)

    search_count = sum(1 for sample in samples if str(((sample.get("metadata", {}) or {}).get("mode", ""))) == "search_unit")
    direct_count = sum(1 for sample in samples if str(((sample.get("metadata", {}) or {}).get("mode", ""))) == "direct_locate")
    return samples, int(search_count), int(direct_count)


def _pre_history_by_stage3_segment(
    annotations: list[dict[str, Any]],
    action_chunk_size: int,
    max_context_frames: int,
) -> tuple[dict[tuple[int, int], list[MemorySlot]], int]:
    memory_slots = _build_memory_slots(annotations=annotations, action_chunk_size=int(action_chunk_size))
    history_limit = max(int(max_context_frames) - 1, 0)
    history_slots: list[MemorySlot] = []
    pre_history: dict[tuple[int, int], list[MemorySlot]] = {}

    for slot in memory_slots:
        if history_slots and int(slot.subtask_id) != int(history_slots[-1].subtask_id):
            history_slots = compress_memory_slots(history_slots)
        if "stage3_chunk" in slot.roles:
            key = (int(slot.subtask_id), int(slot.frame_idx))
            pre_history.setdefault(key, list(history_slots[-history_limit:]) if history_limit > 0 else [])
        history_slots.append(slot)
        if int(max_context_frames) > 0 and len(history_slots) >= int(max_context_frames):
            history_slots = compress_memory_slots(history_slots)

    next_slot_idx = 0 if not memory_slots else max(int(slot.slot_idx) for slot in memory_slots) + 1
    return pre_history, int(next_slot_idx)


def _make_shifted_stage3_slot(
    slot_idx: int,
    annotation: dict[str, Any],
    chunk_frame_indices: list[int],
    action_chunk_size: int,
) -> MemorySlot:
    return _make_memory_slot(
        slot_idx=int(slot_idx),
        annotation=dict(annotation),
        roles=["stage3_chunk", "stage3_chunk_shifted"],
        action_chunk_frame_indices=list(chunk_frame_indices),
        action_chunk_actual_size=len(chunk_frame_indices),
        action_chunk_pad_count=max(int(action_chunk_size) - len(chunk_frame_indices), 0),
    )


def _build_stage3_shifted_action_samples(
    save_dir: Path,
    output_dir_name: str,
    episode_idx: int,
    metadata: dict[str, Any],
    context: EpisodeContext,
    annotations: list[dict[str, Any]],
    max_context_frames: int,
    action_chunk_size: int,
) -> list[dict[str, Any]]:
    output_dir = Path(save_dir) / output_dir_name
    chunk_size = max(int(action_chunk_size), 1)
    history_limit = max(int(max_context_frames) - 1, 0)
    pre_history_map, next_slot_idx = _pre_history_by_stage3_segment(
        annotations=annotations,
        action_chunk_size=chunk_size,
        max_context_frames=int(max_context_frames),
    )
    samples: list[dict[str, Any]] = []

    for seg_start, seg_end in _collect_annotation_segments(annotations):
        segment = annotations[seg_start:seg_end]
        if not segment:
            continue
        if int(segment[0].get("stage", 0) or 0) != 3:
            continue

        segment_len = int(seg_end - seg_start)
        segment_start_frame_idx = int(segment[0].get("frame_idx", seg_start))
        segment_end_frame_idx = int(segment[-1].get("frame_idx", seg_end - 1))
        segment_subtask_id = int(segment[0].get("subtask", 0) or 0)
        base_history = list(pre_history_map.get((segment_subtask_id, segment_start_frame_idx), []))

        for offset in range(min(chunk_size, segment_len)):
            lattice_history: list[MemorySlot] = list(base_history)
            lattice_index = 0
            for annotation_idx in range(seg_start + offset, seg_end, chunk_size):
                current_annotation = dict(annotations[annotation_idx])
                chunk_annotations = annotations[annotation_idx : min(annotation_idx + chunk_size, seg_end)]
                chunk_frame_indices = [int(item.get("frame_idx", 0)) for item in chunk_annotations]
                current_slot = _make_shifted_stage3_slot(
                    slot_idx=next_slot_idx,
                    annotation=current_annotation,
                    chunk_frame_indices=chunk_frame_indices,
                    action_chunk_size=chunk_size,
                )
                next_slot_idx += 1

                prompt_history = list(lattice_history)
                prompt_history = list(prompt_history[-history_limit:]) if history_limit > 0 else []
                snapshot = _make_snapshot(current_slot=current_slot, prompt_slots=prompt_history)
                stage_action_chunk = _build_stage_action_chunk(context, snapshot.current_slot)
                snapshot.current_slot.current_annotation["action_chunk"] = stage_action_chunk
                prompt_frame_indices = list(snapshot.prompt_frame_indices)
                images = _build_images(
                    output_dir=output_dir,
                    episode_idx=episode_idx,
                    frames=context.frames,
                    frame_indices=prompt_frame_indices,
                )
                target_key = snapshot.current_slot.target_key()
                sample = {
                    "images": images,
                    "messages": _build_messages(
                        _build_object_search_user_prompt(metadata, snapshot),
                        _render_object_search_response(metadata, snapshot),
                    ),
                    "metadata": {
                        "episode_idx": int(episode_idx),
                        "task_name": str(metadata.get("task_name", "")),
                        "task_type": TASK_TYPE_NAME,
                        "subtask_id": int(snapshot.subtask_id),
                        "raw_stage": int(snapshot.stage),
                        "stage": int(snapshot.stage),
                        "roles": list(snapshot.roles),
                        "mode": "action_chunk_shifted",
                        "current_frame_idx": int(snapshot.current_frame_idx),
                        "prompt_frame_indices": prompt_frame_indices,
                        "prompt_image_count": int(len(images)),
                        "prompt_planned_actions": _metadata_action_pairs(snapshot.prompt_planned_actions),
                        "prompt_view_deltas": _metadata_action_pairs(snapshot.prompt_planned_actions),
                        "target_key": None if target_key is None else str(target_key),
                        "current_target_visible": bool(_annotation_has_visible_target(snapshot.current_annotation, target_key)),
                        "prompt_has_target": bool(snapshot.evidence_prompt_index is not None),
                        "memory_support_ready": bool(snapshot.memory_support_ready),
                        "evidence_from_history": bool(snapshot.evidence_from_history),
                        "evidence_frame_idx": snapshot.evidence_frame_idx,
                        "evidence_prompt_index": snapshot.evidence_prompt_index,
                        "evidence_uv_norm": snapshot.evidence_uv_norm,
                        "camera_delta_deg": int(_to_vqa_rotation_pair((_search_camera_delta(snapshot), 0))[0]),
                        "search_unit_deg": int(SEARCH_UNIT_DEG),
                        "action_chunk_size": int(chunk_size),
                        "action_chunk_frame_indices": list(chunk_frame_indices),
                        "action_chunk_actual_size": int(snapshot.current_slot.action_chunk_actual_size),
                        "action_chunk_pad_count": int(snapshot.current_slot.action_chunk_pad_count),
                        "action_lattice_offset": int(offset),
                        "action_lattice_index": int(lattice_index),
                        "stage3_segment_start_frame_idx": int(segment_start_frame_idx),
                        "stage3_segment_end_frame_idx": int(segment_end_frame_idx),
                        "stage3_segment_start_annotation_index": int(seg_start),
                        "stage3_segment_end_annotation_index": int(seg_end),
                        "sparse_version": "v4",
                    },
                    "action": stage_action_chunk,
                }
                samples.append(sample)
                lattice_history.append(current_slot)
                if int(max_context_frames) > 0 and len(lattice_history) >= int(max_context_frames):
                    lattice_history = compress_memory_slots(lattice_history)
                lattice_index += 1

    return sorted(
        samples,
        key=lambda item: (
            int(((item.get("metadata", {}) or {}).get("subtask_id", 0) or 0)),
            int(((item.get("metadata", {}) or {}).get("stage3_segment_start_frame_idx", 0) or 0)),
            int(((item.get("metadata", {}) or {}).get("current_frame_idx", 0) or 0)),
            int(((item.get("metadata", {}) or {}).get("action_lattice_offset", 0) or 0)),
        ),
    )


def _sample_sort_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    metadata = sample.get("metadata", {}) or {}
    mode = str(metadata.get("mode", ""))
    mode_priority = {"search_unit": 0, "direct_locate": 1, "action_chunk_shifted": 2}.get(mode, 9)
    return (
        int(metadata.get("subtask_id", 0) or 0),
        int(metadata.get("current_frame_idx", 0) or 0),
        int(mode_priority),
        int(metadata.get("action_lattice_offset", -1) or -1),
    )


def _episode_samples_payload(
    args: tuple[str, str, int, str, str, int, int, float],
) -> dict[str, Any]:
    save_dir, output_dir_name, episode_idx, metadata_path, hdf5_path, max_context_frames, action_chunk_size, fovy_deg = args
    save_path = Path(save_dir)
    output_dir = save_path / output_dir_name
    metadata = _read_json(Path(metadata_path))
    frames, left_arm_actions, left_gripper_actions, right_arm_actions, right_gripper_actions = load_hdf5_episode_data(str(hdf5_path))
    annotations = [dict(item) for item in (metadata.get("frame_annotations", []) or [])]

    if not frames or not annotations:
        return {
            "episode_idx": int(episode_idx),
            "sample_count": 0,
            "search_count": 0,
            "direct_count": 0,
            "action_count": 0,
            "samples": [],
        }

    context = EpisodeContext(
        metadata=dict(metadata),
        hdf5_path=str(hdf5_path),
        action_chunk_size=max(int(action_chunk_size), 1),
        max_context_frames=int(max_context_frames),
        frames=frames,
        left_arm_actions=left_arm_actions,
        left_gripper_actions=left_gripper_actions,
        right_arm_actions=right_arm_actions,
        right_gripper_actions=right_gripper_actions,
    )
    stage12_samples, search_count, direct_count = _stage12_samples(
        output_dir=output_dir,
        episode_idx=int(episode_idx),
        metadata=metadata,
        context=context,
        annotations=annotations,
        max_context_frames=int(max_context_frames),
        action_chunk_size=max(int(action_chunk_size), 1),
        fovy_deg=float(fovy_deg),
    )
    stage3_samples = _build_stage3_shifted_action_samples(
        save_dir=save_path,
        output_dir_name=str(output_dir_name),
        episode_idx=int(episode_idx),
        metadata=metadata,
        context=context,
        annotations=annotations,
        max_context_frames=int(max_context_frames),
        action_chunk_size=max(int(action_chunk_size), 1),
    )
    samples = sorted(stage12_samples + stage3_samples, key=_sample_sort_key)
    stage12_action_count = sum(1 for sample in stage12_samples if "action" in sample)
    stage3_action_count = len(stage3_samples)

    return {
        "episode_idx": int(episode_idx),
        "sample_count": int(len(samples)),
        "search_count": int(search_count),
        "direct_count": int(direct_count),
        "action_count": int(stage12_action_count + stage3_action_count),
        "stage12_action_count": int(stage12_action_count),
        "stage3_action_count": int(stage3_action_count),
        "samples": samples,
    }


def export_single_task(
    save_dir: str,
    task_config_name: str | None = None,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    max_context_frames: int = DEFAULT_MAX_CONTEXT_FRAMES,
    action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
    num_workers: int | None = None,
    overwrite: bool = True,
    episode_indices: list[int] | None = None,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    save_path = Path(save_dir)
    output_dir = save_path / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{TASK_TYPE_NAME}.json"
    writer = _JsonArrayWriter(output_path, overwrite=overwrite)

    episode_pairs = _filter_episode_pairs(
        _collect_episode_pairs(save_path),
        episode_indices=episode_indices,
        max_episodes=max_episodes,
    )
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
            max(int(action_chunk_size), 1),
            float(fovy_deg),
        )
        for episode_idx, metadata_path, hdf5_path in episode_pairs
    ]

    total_samples = 0
    total_search = 0
    total_direct = 0
    total_action = 0
    total_stage12_action = 0
    total_stage3_action = 0
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
                total_action += int(payload.get("action_count", 0) or 0)
                total_stage12_action += int(payload.get("stage12_action_count", 0) or 0)
                total_stage3_action += int(payload.get("stage3_action_count", 0) or 0)
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
        "action_count": int(total_action),
        "stage12_action_count": int(total_stage12_action),
        "stage3_action_count": int(total_stage3_action),
        "episode_count": int(episode_count),
        "worker_count": int(worker_count),
        "task_config_name": task_config_name,
        "head_camera_fovy_deg": float(fovy_deg),
        "action_chunk_size": max(int(action_chunk_size), 1),
        "episode_indices": None if episode_indices is None else [int(idx) for idx in episode_indices],
        "max_episodes": max_episodes,
    }


def _export_single_task_job(job: tuple[str, str, str | None, str, int, int, int, list[int] | None, int | None]) -> tuple[str, dict[str, Any]]:
    task_name, task_dir, task_config_name, output_dir_name, episode_workers, max_context_frames, action_chunk_size, episode_indices, max_episodes = job
    started_at = time.time()
    summary = export_single_task(
        save_dir=task_dir,
        task_config_name=task_config_name,
        output_dir_name=output_dir_name,
        num_workers=episode_workers,
        max_context_frames=max_context_frames,
        action_chunk_size=action_chunk_size,
        overwrite=True,
        episode_indices=episode_indices,
        max_episodes=max_episodes,
    )
    summary["elapsed_seconds"] = round(float(time.time() - started_at), 3)
    return task_name, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export rotate object-search visibility-memory v4 samples. "
            "v4 keeps v3 shifted stage3 actions and adds sampled action chunks to stage1/stage2 turns."
        )
    )
    parser.add_argument("--save-dir", type=str, default=None, help="Optional single task directory to export.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--whitelist-file", type=str, default="task_config/rotate_task_whitelist.yml")
    parser.add_argument("--task-config", type=str, default=None, help="Task config prefix, e.g. demo_randomized_easy_ep200")
    parser.add_argument("--output-dir-name", type=str, default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--max-context-frames", type=int, default=DEFAULT_MAX_CONTEXT_FRAMES)
    parser.add_argument("--action-chunk-size", type=int, default=DEFAULT_ACTION_CHUNK_SIZE)
    parser.add_argument("--episode-indices", nargs="*", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--task-workers", type=int, default=None)
    parser.add_argument("--episode-workers", type=int, default=None)
    args = parser.parse_args()

    episode_indices = None if args.episode_indices is None else [int(idx) for idx in args.episode_indices]

    if args.save_dir:
        task_dir = Path(args.save_dir)
        task_config_name = _resolve_task_config_name(task_dir, args.task_config)
        summary = export_single_task(
            save_dir=str(task_dir),
            task_config_name=task_config_name,
            output_dir_name=str(args.output_dir_name),
            max_context_frames=int(args.max_context_frames),
            action_chunk_size=max(int(args.action_chunk_size), 1),
            num_workers=args.episode_workers,
            overwrite=True,
            episode_indices=episode_indices,
            max_episodes=args.max_episodes,
        )
        if args.summary_path:
            summary_path = Path(args.summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[ok] {task_dir.name}: {summary['task_type_counts']} "
            f"(search={summary['search_count']}, direct={summary['direct_count']}, "
            f"action={summary['action_count']}, workers={summary['worker_count']})"
        )
        return

    data_root = Path(args.data_root)
    whitelist = _load_whitelist(Path(args.whitelist_file))
    pending_jobs: list[tuple[str, str, str | None, str, int, int, int, list[int] | None, int | None]] = []
    summary: dict[str, Any] = {}
    for task_name in whitelist:
        task_dir = _discover_task_dir(data_root=data_root, task_name=task_name, task_config=args.task_config)
        if task_dir is None:
            summary[task_name] = {"status": "missing_task_dir"}
            print(f"[skip] {task_name}: task dir not found")
            continue
        task_config_name = _resolve_task_config_name(task_dir, args.task_config)
        pending_jobs.append(
            (
                task_name,
                str(task_dir),
                task_config_name,
                str(args.output_dir_name),
                1,
                int(args.max_context_frames),
                max(int(args.action_chunk_size), 1),
                episode_indices,
                args.max_episodes,
            )
        )

    task_workers = _resolve_worker_count(args.task_workers, DEFAULT_TASK_WORKERS, upper_bound=len(pending_jobs) or 1)
    episode_workers = _resolve_worker_count(args.episode_workers, _default_episode_workers(task_workers))
    pending_jobs = [
        (task_name, task_dir, task_config_name, output_dir_name, episode_workers, max_context_frames, action_chunk_size, job_episode_indices, max_episodes)
        for task_name, task_dir, task_config_name, output_dir_name, _, max_context_frames, action_chunk_size, job_episode_indices, max_episodes in pending_jobs
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
                    f"(search={task_summary['search_count']}, direct={task_summary['direct_count']}, "
                    f"action={task_summary['action_count']}, {task_summary['elapsed_seconds']}s)"
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
                task_name, task_dir, _, _, _, _, _, _, _ = future_map[future]
                try:
                    finished_task_name, task_summary = future.result()
                    summary[finished_task_name] = task_summary
                    print(
                        f"[ok] {finished_task_name}: {task_summary['task_type_counts']} "
                        f"(search={task_summary['search_count']}, direct={task_summary['direct_count']}, "
                        f"action={task_summary['action_count']}, {task_summary['elapsed_seconds']}s)"
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
