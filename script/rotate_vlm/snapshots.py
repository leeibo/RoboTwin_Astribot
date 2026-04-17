from __future__ import annotations

import math
from typing import Any

import cv2
import h5py
import numpy as np

from .models import CompressionEvent, EpisodeContext, EpisodeSnapshot, MemorySlot


TARGET_THETA_KEY_DECIMALS = 6
DEFAULT_MEMORY_FOV_HALF_DEG = 35.0
DEFAULT_COVERAGE_STEP_DEG = 2.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _wrap_to_180(angle_deg: float) -> float:
    angle_deg = float(angle_deg)
    while angle_deg <= -180.0:
        angle_deg += 360.0
    while angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def _normalize_uv(uv: Any) -> list[float] | None:
    if not isinstance(uv, (list, tuple)) or len(uv) < 2:
        return None
    return [float(uv[0]), float(uv[1])]


def _annotation_target_keys(annotation: dict[str, Any]) -> list[str]:
    keys = annotation.get("search_target_keys", None) or []
    if keys:
        return [str(key) for key in keys]
    keys = annotation.get("action_target_keys", None) or []
    if keys:
        return [str(key) for key in keys]
    focus_key = annotation.get("focus_object_key", None)
    if focus_key is not None:
        return [str(focus_key)]
    return []


def _annotation_planned_delta_deg(annotation: dict[str, Any]) -> float:
    target_theta = annotation.get("camera_target_theta", None)
    current_heading = _safe_float(annotation.get("waist_heading_deg", 0.0))
    if target_theta is None:
        return 0.0
    return _wrap_to_180(math.degrees(float(target_theta)) - current_heading)


def _annotation_planned_heading_deg(annotation: dict[str, Any]) -> float:
    return _wrap_to_180(
        _safe_float(annotation.get("waist_heading_deg", 0.0)) + _annotation_planned_delta_deg(annotation)
    )


def _decode_jpeg_array(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to decode JPEG frame from HDF5")
    return frame


def load_hdf5_episode_data(hdf5_path: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    left_arm = np.zeros((0, 0), dtype=np.float64)
    right_arm = np.zeros((0, 0), dtype=np.float64)
    with h5py.File(hdf5_path, "r") as hdf5_file:
        rgb_dataset = None
        for key in [
            "observation/camera_head/rgb",
            "observation/head_camera/rgb",
            "camera_head/rgb",
            "head_camera/rgb",
        ]:
            if key in hdf5_file:
                rgb_dataset = hdf5_file[key]
                break
        if rgb_dataset is None:
            candidate_paths: list[str] = []

            def _visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    candidate_paths.append(name)

            hdf5_file.visititems(_visit)
            for suffix in ("camera_head/rgb", "head_camera/rgb", "/rgb", "rgb"):
                for path in candidate_paths:
                    if path.endswith(suffix):
                        rgb_dataset = hdf5_file[path]
                        break
                if rgb_dataset is not None:
                    break
        if rgb_dataset is None:
            frame_count = 0
            if "joint_action/left_arm" in hdf5_file:
                frame_count = max(frame_count, int(hdf5_file["joint_action/left_arm"].shape[0]))
            if "joint_action/right_arm" in hdf5_file:
                frame_count = max(frame_count, int(hdf5_file["joint_action/right_arm"].shape[0]))
            if frame_count <= 0:
                raise KeyError(f"missing camera_head/head_camera RGB dataset in {hdf5_path}")
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(frame_count)]
        else:
            frames = [_decode_jpeg_array(bytes(payload)) for payload in rgb_dataset[:]]
        if "joint_action/left_arm" in hdf5_file:
            left_arm = np.array(hdf5_file["joint_action/left_arm"][:], dtype=np.float64)
        if "joint_action/right_arm" in hdf5_file:
            right_arm = np.array(hdf5_file["joint_action/right_arm"][:], dtype=np.float64)
    return frames, left_arm, right_arm


def _segment_key(annotation: dict[str, Any]) -> tuple[int, int, float | None]:
    subtask_id = int(annotation.get("subtask", 0) or 0)
    stage = int(annotation.get("stage", 0) or 0)
    if stage == 3:
        return (subtask_id, stage, None)
    target_theta = annotation.get("camera_target_theta", None)
    if target_theta is None:
        return (subtask_id, stage, None)
    return (subtask_id, stage, round(float(target_theta), TARGET_THETA_KEY_DECIMALS))


def _collect_annotation_segments(annotations: list[dict[str, Any]]) -> list[tuple[int, int]]:
    if not annotations:
        return []
    segments: list[tuple[int, int]] = []
    seg_start = 0
    prev_key = _segment_key(annotations[0])
    for idx in range(1, len(annotations)):
        key = _segment_key(annotations[idx])
        if key != prev_key:
            segments.append((seg_start, idx))
            seg_start = idx
            prev_key = key
    segments.append((seg_start, len(annotations)))
    return segments


def _make_memory_slot(
    slot_idx: int,
    annotation: dict[str, Any],
    roles: list[str],
    action_chunk_frame_indices: list[int] | None = None,
    action_chunk_actual_size: int = 0,
    action_chunk_pad_count: int = 0,
) -> MemorySlot:
    planned_delta_deg = 0.0 if "stage3_chunk" in roles else _annotation_planned_delta_deg(annotation)
    return MemorySlot(
        slot_idx=int(slot_idx),
        frame_idx=int(annotation.get("frame_idx", slot_idx)),
        subtask_id=int(annotation.get("subtask", 0) or 0),
        stage=int(annotation.get("stage", 0) or 0),
        current_annotation=dict(annotation),
        current_heading_deg=_safe_float(annotation.get("waist_heading_deg", 0.0)),
        planned_delta_deg=float(planned_delta_deg),
        planned_heading_deg=(
            _safe_float(annotation.get("waist_heading_deg", 0.0))
            if "stage3_chunk" in roles
            else _annotation_planned_heading_deg(annotation)
        ),
        roles=list(roles),
        action_chunk_frame_indices=[] if action_chunk_frame_indices is None else [int(idx) for idx in action_chunk_frame_indices],
        action_chunk_actual_size=int(action_chunk_actual_size),
        action_chunk_pad_count=int(action_chunk_pad_count),
    )


def _build_memory_slots(
    annotations: list[dict[str, Any]],
    action_chunk_size: int,
) -> list[MemorySlot]:
    slots: list[MemorySlot] = []
    slot_idx = 0
    chunk_size = max(int(action_chunk_size), 1)
    for seg_start, seg_end in _collect_annotation_segments(annotations):
        segment = annotations[seg_start:seg_end]
        if not segment:
            continue
        stage = int(segment[0].get("stage", 0) or 0)
        if stage == 3:
            for chunk_start in range(seg_start, seg_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seg_end)
                chunk_annotations = annotations[chunk_start:chunk_end]
                current_annotation = dict(chunk_annotations[0])
                chunk_frame_indices = [int(item.get("frame_idx", 0)) for item in chunk_annotations]
                slots.append(
                    _make_memory_slot(
                        slot_idx=slot_idx,
                        annotation=current_annotation,
                        roles=["stage3_chunk"],
                        action_chunk_frame_indices=chunk_frame_indices,
                        action_chunk_actual_size=len(chunk_frame_indices),
                        action_chunk_pad_count=max(chunk_size - len(chunk_frame_indices), 0),
                    )
                )
                slot_idx += 1
            continue

        current_annotation = dict(segment[0])
        roles = [f"stage{int(stage)}_segment"]
        slots.append(
            _make_memory_slot(
                slot_idx=slot_idx,
                annotation=current_annotation,
                roles=roles,
            )
        )
        slot_idx += 1
    return slots


def _slot_target_uv(slot: MemorySlot) -> list[float] | None:
    uv = slot.target_uv_norm()
    if uv is None:
        return None
    if float(uv[0]) < 0.0 or float(uv[1]) < 0.0:
        return None
    return uv


def _choose_evidence_slot(current_slot: MemorySlot, prompt_slots: list[MemorySlot]) -> tuple[MemorySlot | None, list[float] | None]:
    current_uv = _slot_target_uv(current_slot)
    if current_uv is not None:
        return current_slot, current_uv

    current_target_key = current_slot.target_key()
    current_target_keys = set(current_slot.target_keys())
    for slot in reversed(prompt_slots):
        slot_uv = _slot_target_uv(slot)
        if slot_uv is None:
            continue
        slot_target_keys = set(slot.target_keys())
        if current_target_key is not None and current_target_key in slot_target_keys:
            return slot, slot_uv
        if current_target_keys and slot_target_keys & current_target_keys:
            return slot, slot_uv
    return None, None


def _make_snapshot(current_slot: MemorySlot, prompt_slots: list[MemorySlot]) -> EpisodeSnapshot:
    prompt_frame_indices = [int(slot.frame_idx) for slot in prompt_slots] + [int(current_slot.frame_idx)]
    prompt_planned_actions = [(int(round(float(slot.planned_delta_deg))), 0) for slot in prompt_slots]
    evidence_slot, evidence_uv = _choose_evidence_slot(current_slot, prompt_slots)
    evidence_prompt_index = None
    evidence_from_history = False
    if evidence_slot is not None:
        for idx, frame_idx in enumerate(prompt_frame_indices, start=1):
            if int(frame_idx) == int(evidence_slot.frame_idx):
                evidence_prompt_index = int(idx)
        evidence_from_history = int(evidence_slot.frame_idx) != int(current_slot.frame_idx)

    return EpisodeSnapshot(
        current_slot=current_slot,
        prompt_slots=list(prompt_slots),
        prompt_frame_indices=prompt_frame_indices,
        prompt_planned_actions=prompt_planned_actions,
        evidence_frame_idx=(None if evidence_slot is None else int(evidence_slot.frame_idx)),
        evidence_prompt_index=evidence_prompt_index,
        evidence_uv_norm=evidence_uv,
        evidence_from_history=bool(evidence_from_history),
        memory_support_ready=bool(evidence_slot is not None and evidence_prompt_index is not None),
    )


def _coverage_grid(step_deg: float = DEFAULT_COVERAGE_STEP_DEG) -> np.ndarray:
    return np.arange(-180.0, 180.0 + float(step_deg) * 0.5, float(step_deg), dtype=np.float64)


def _slot_coverage_indices(
    slot: MemorySlot,
    grid: np.ndarray,
    half_fov_deg: float = DEFAULT_MEMORY_FOV_HALF_DEG,
) -> set[int]:
    diffs = np.abs(np.vectorize(_wrap_to_180)(grid - float(slot.current_heading_deg)))
    return {int(idx) for idx, diff in enumerate(diffs.tolist()) if float(diff) <= float(half_fov_deg) + 1e-6}


def _compress_memory_slots(
    slots: list[MemorySlot],
    half_fov_deg: float = DEFAULT_MEMORY_FOV_HALF_DEG,
) -> list[MemorySlot]:
    if len(slots) <= 1:
        return list(slots)

    grid = _coverage_grid()
    covered: set[int] = set()
    kept_reversed: list[MemorySlot] = []
    for slot in reversed(slots):
        coverage = _slot_coverage_indices(slot, grid=grid, half_fov_deg=half_fov_deg)
        if coverage - covered:
            kept_reversed.append(slot)
            covered.update(coverage)

    if not kept_reversed:
        kept_reversed.append(slots[-1])

    kept = list(reversed(kept_reversed))
    kept.sort(key=lambda slot: (int(slot.frame_idx), int(slot.slot_idx)))
    return kept


def _apply_history_compression(
    history_slots: list[MemorySlot],
    compression_events: list[CompressionEvent],
    trigger: str,
    trigger_frame_idx: int,
) -> list[MemorySlot]:
    before_slots = list(history_slots)
    if len(before_slots) <= 1:
        return before_slots
    after_slots = _compress_memory_slots(before_slots)
    if len(after_slots) < len(before_slots):
        compression_events.append(
            CompressionEvent(
                trigger=str(trigger),
                trigger_frame_idx=int(trigger_frame_idx),
                before_slots=list(before_slots),
                after_slots=list(after_slots),
            )
        )
    return list(after_slots)


def build_episode_context(
    metadata: dict[str, Any],
    hdf5_path: str,
    action_chunk_size: int = 10,
    max_context_frames: int = 16,
) -> EpisodeContext:
    frames, left_arm_actions, right_arm_actions = load_hdf5_episode_data(hdf5_path)
    annotations = [dict(item) for item in list(metadata.get("frame_annotations", []) or [])]
    memory_slots = _build_memory_slots(annotations=annotations, action_chunk_size=int(action_chunk_size))

    context = EpisodeContext(
        metadata=dict(metadata),
        hdf5_path=str(hdf5_path),
        action_chunk_size=int(action_chunk_size),
        max_context_frames=int(max_context_frames),
        frames=frames,
        left_arm_actions=left_arm_actions,
        right_arm_actions=right_arm_actions,
        memory_slots=list(memory_slots),
    )

    history_slots: list[MemorySlot] = []
    history_limit = max(int(max_context_frames) - 1, 0)

    for slot in memory_slots:
        if history_slots and int(slot.subtask_id) != int(history_slots[-1].subtask_id):
            history_slots = _apply_history_compression(
                history_slots=history_slots,
                compression_events=context.compression_events,
                trigger="subtask_switch",
                trigger_frame_idx=int(slot.frame_idx),
            )

        prompt_history = list(history_slots[-history_limit:]) if history_limit > 0 else []
        context.snapshots.append(_make_snapshot(current_slot=slot, prompt_slots=prompt_history))

        history_slots.append(slot)
        if int(max_context_frames) > 0 and len(history_slots) >= int(max_context_frames):
            history_slots = _apply_history_compression(
                history_slots=history_slots,
                compression_events=context.compression_events,
                trigger="max_context",
                trigger_frame_idx=int(slot.frame_idx),
            )

    return context
