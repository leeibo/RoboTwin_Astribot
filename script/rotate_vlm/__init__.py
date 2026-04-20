from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .annotated_video import default_annotated_video_path, export_annotated_video
from .models import CompressionEvent, EpisodeContext, EpisodeSnapshot, MemorySlot
from .snapshots import (
    DEFAULT_MEMORY_FOV_HALF_DEG,
    _coverage_grid,
    _slot_coverage_indices,
    build_episode_context,
    compress_memory_slots,
)


DEFAULT_MAX_CONTEXT_FRAMES = 16
DEFAULT_ACTION_CHUNK_SIZE = 10
REGISTERED_TASK_TYPES = ("object_search", "angle_delta", "memory_compression_vqa")
MAX_COMPRESSION_VARIANTS_PER_SIZE = 64


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return str(path)


class _JsonArrayWriter:
    def __init__(self, path: Path, overwrite: bool = True):
        self.path = Path(path)
        self.tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing_items: list[Any] = []
        if not overwrite and self.path.exists():
            with open(self.path, "r", encoding="utf-8") as file:
                existing_items = list(json.load(file))
        self.file = open(self.tmp_path, "w", encoding="utf-8")
        self.file.write("[\n")
        self.is_first = True
        self.count = 0
        self.committed = False
        for item in existing_items:
            self.append(item)

    def append(self, item: Any) -> None:
        if not self.is_first:
            self.file.write(",\n")
        json.dump(item, self.file, ensure_ascii=False)
        self.is_first = False
        self.count += 1

    def commit(self) -> None:
        if self.committed:
            return
        self.file.write("\n]\n")
        self.file.close()
        self.tmp_path.replace(self.path)
        self.committed = True

    def abort(self) -> None:
        if not self.file.closed:
            self.file.close()
        if self.tmp_path.exists():
            self.tmp_path.unlink()


def _parse_search_phrase(instruction: str, fallback_name: str | None = None) -> str:
    text = str(instruction or "").strip()
    lower_text = text.lower()
    patterns = [
        r"find the (.+?) and ",
        r"find the (.+?)\.",
        r"find the (.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower_text)
        if match:
            return match.group(1).strip()
    if fallback_name:
        return str(fallback_name).strip().lower()
    return "target object"


def _pretty_object_name(raw_name: str) -> str:
    text = str(raw_name or "").strip()
    if not text:
        return "object"
    text = text.split("/", 1)[0]
    text = re.sub(r"^\d+[_-]?", "", text)
    text = text.replace("_", " ").replace("-", " ").strip()
    return text or "object"


def _infer_object_name_map(metadata: dict[str, Any]) -> dict[str, str]:
    object_name_map = {
        str(key): _pretty_object_name(str(value).strip())
        for key, value in (metadata.get("object_key_to_name", {}) or {}).items()
        if str(value).strip()
    }
    subtask_defs = metadata.get("subtask_defs", []) or []
    instruction_map = metadata.get("subtask_instruction_map", {}) or {}
    for subtask in subtask_defs:
        if not isinstance(subtask, dict):
            continue
        search_keys = [str(key) for key in (subtask.get("search_target_keys", []) or [])]
        if len(search_keys) != 1:
            continue
        key = search_keys[0]
        if key in object_name_map:
            continue
        instruction = instruction_map.get(str(subtask.get("id")), "")
        phrase = _parse_search_phrase(str(instruction), fallback_name=key)
        if phrase and phrase != "target object":
            object_name_map[key] = phrase
    return object_name_map


def _format_object_name(metadata: dict[str, Any], object_key: str | None) -> str:
    if object_key is None:
        return "object"
    object_name_map = _infer_object_name_map(metadata)
    name = object_name_map.get(str(object_key), str(object_key))
    return _pretty_object_name(name)


def _subtask_instruction(metadata: dict[str, Any], subtask_id: int) -> str:
    instruction_map = metadata.get("subtask_instruction_map", {}) or {}
    return str(instruction_map.get(str(subtask_id), metadata.get("task_instruction", ""))).strip()


def _resolve_search_phrase(metadata: dict[str, Any], slot: MemorySlot) -> str:
    instruction = _subtask_instruction(metadata, slot.subtask_id)
    target_key = slot.target_key()
    object_name_map = _infer_object_name_map(metadata)
    fallback_name = None if target_key is None else object_name_map.get(target_key, None)
    return _parse_search_phrase(str(instruction), fallback_name=fallback_name)


def _join_phrases(phrases: list[str]) -> str:
    items = [str(item).strip() for item in phrases if str(item).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _resolve_target_phrase(metadata: dict[str, Any], slot: MemorySlot, with_articles: bool = False) -> str:
    names: list[str] = []
    seen = set()
    for object_key in slot.target_keys():
        name = _format_object_name(metadata, object_key)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    if not names:
        phrase = _resolve_search_phrase(metadata, slot)
        return f"the {phrase}" if with_articles else phrase
    if with_articles:
        return _join_phrases([f"the {name}" for name in names])
    return _join_phrases(names)


def _format_uv_1000(uv_norm: list[float] | None) -> str:
    if not isinstance(uv_norm, (list, tuple)) or len(uv_norm) < 2:
        return "(-1,-1)"
    return f"({int(round(float(uv_norm[0]) * 1000.0))},{int(round(float(uv_norm[1]) * 1000.0))})"


def _wrap_to_180(angle_deg: float) -> float:
    angle_deg = float(angle_deg)
    while angle_deg <= -180.0:
        angle_deg += 360.0
    while angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def _format_frame_summary(frame_count: int) -> str:
    frame_count = int(frame_count)
    if frame_count <= 1:
        return "Frames: current only."
    return f"Frames: {frame_count} total ({frame_count - 1} history + current)."


def _format_frame_field(indices: list[int] | tuple[int, ...] | int | None) -> str:
    if indices is None:
        return "[]"
    if isinstance(indices, int):
        values = [int(indices)]
    else:
        values = [int(value) for value in indices]
    if not values:
        return "[]"
    return "[" + ", ".join(str(value) for value in values) + "]"


def _rotation_difference_pair(from_heading_deg: float, to_heading_deg: float) -> tuple[int, int]:
    return (int(round(_wrap_to_180(float(to_heading_deg) - float(from_heading_deg)))), 0)


def _slot_view_delta(previous_slot: MemorySlot, current_slot: MemorySlot) -> tuple[int, int]:
    return _rotation_difference_pair(previous_slot.current_heading_deg, current_slot.current_heading_deg)


def _slot_sequence_view_deltas(slots: list[MemorySlot]) -> list[tuple[int, int]]:
    if len(slots) <= 1:
        return []
    return [_slot_view_delta(previous_slot, current_slot) for previous_slot, current_slot in zip(slots[:-1], slots[1:])]


def _format_rotation_pairs(actions: list[tuple[int, int]]) -> str:
    if not actions:
        return "none"
    return "[" + ", ".join(f"({int(dx)}, {int(dy)})" for dx, dy in actions) + "]"


def _format_rotate_text(rotation: tuple[int, int]) -> str:
    return f"Rotate({int(rotation[0])}, {int(rotation[1])})"


def _history_or_current_images(
    save_dir: Path,
    episode_idx: int,
    context: EpisodeContext,
    frame_indices: list[int],
) -> list[str]:
    image_dir = save_dir / "vlm" / "images" / f"episode{int(episode_idx)}"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    for frame_idx in frame_indices:
        frame = context.frames[int(frame_idx)]
        path = image_dir / f"frame_{int(frame_idx):04d}.png"
        if not path.exists():
            Image.fromarray(frame).save(path)
        image_paths.append(str(path))
    return image_paths


def _metadata_action_pairs(actions: list[tuple[int, int]]) -> list[list[str]]:
    return [[str(int(a)), str(int(b))] for a, b in actions]


def _infer_primary_arm(metadata: dict[str, Any]) -> str | None:
    placeholders = metadata.get("scene_info_placeholders", {}) or {}
    arm = str(placeholders.get("{a}", "")).strip().lower()
    if arm in {"left", "right"}:
        return arm
    task_name = str(metadata.get("task_name", "")).lower()
    if "_left_" in task_name or task_name.endswith("_left") or "_left_rotate_view" in task_name:
        return "left"
    if "_right_" in task_name or task_name.endswith("_right") or "_right_rotate_view" in task_name:
        return "right"
    return None


def _build_messages(user_content: str, assistant_content: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(user_content)},
        {"role": "assistant", "content": str(assistant_content)},
    ]


def _memory_slot_by_frame(slots: list[MemorySlot], frame_idx: int | None) -> MemorySlot | None:
    if frame_idx is None:
        return None
    for slot in slots:
        if int(slot.frame_idx) == int(frame_idx):
            return slot
    return None


def _search_camera_delta(snapshot: EpisodeSnapshot) -> int:
    current_slot = snapshot.current_slot
    if "stage3_chunk" in current_slot.roles:
        return 0
    if snapshot.evidence_from_history and snapshot.evidence_frame_idx is not None:
        evidence_slot = _memory_slot_by_frame(snapshot.prompt_slots, snapshot.evidence_frame_idx)
        if evidence_slot is not None:
            return int(round(_wrap_to_180(evidence_slot.current_heading_deg - current_slot.current_heading_deg)))
    return int(round(float(current_slot.planned_delta_deg)))


def _search_info_complete(snapshot: EpisodeSnapshot) -> bool:
    if int(snapshot.stage) >= 2:
        return True
    return bool(snapshot.memory_support_ready and snapshot.evidence_prompt_index is not None and snapshot.evidence_uv_norm is not None)


def _object_search_memory_summary(snapshot: EpisodeSnapshot) -> str:
    return _format_frame_summary(len(snapshot.prompt_frame_indices))


def _object_search_frame_field(snapshot: EpisodeSnapshot, info_complete: bool) -> str:
    if "stage3_chunk" in snapshot.roles:
        return _format_frame_field([int(len(snapshot.prompt_frame_indices))])
    if info_complete and snapshot.evidence_prompt_index is not None:
        return _format_frame_field([int(snapshot.evidence_prompt_index)])
    return _format_frame_field([])


def _build_object_search_user_prompt(metadata: dict[str, Any], snapshot: EpisodeSnapshot) -> str:
    instruction = _subtask_instruction(metadata, snapshot.subtask_id)
    image_tokens = "".join("<image>" for _ in snapshot.prompt_frame_indices)
    return (
        f'{image_tokens}Your task is: "{instruction}" '
        "The input images are ordered from earliest to latest, and the last image is the current view. "
        "Please think about the next action and output it. "
        "Your response should be in the format of: "
        "<think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>."
    )


def _render_object_search_response(metadata: dict[str, Any], snapshot: EpisodeSnapshot) -> str:
    current_slot = snapshot.current_slot
    instruction = _subtask_instruction(metadata, current_slot.subtask_id)
    phrase = _resolve_search_phrase(metadata, current_slot)
    camera_rotation = (_search_camera_delta(snapshot), 0)
    info_complete = _search_info_complete(snapshot)
    has_evidence_frame = snapshot.evidence_prompt_index is not None and snapshot.evidence_uv_norm is not None
    primary_arm = _infer_primary_arm(metadata)
    memory_summary = _object_search_memory_summary(snapshot)
    past_actions = _format_rotation_pairs(snapshot.prompt_planned_actions)
    camera_field = _format_rotate_text(camera_rotation)

    if info_complete and has_evidence_frame and snapshot.evidence_from_history:
        evidence_text = f"The {phrase} was found in frame {int(snapshot.evidence_prompt_index)} at {_format_uv_1000(snapshot.evidence_uv_norm)}."
    elif info_complete and has_evidence_frame:
        evidence_text = f"The {phrase} is visible in the current view at {_format_uv_1000(snapshot.evidence_uv_norm)}."
    elif info_complete:
        evidence_text = f"The {phrase} has already been localized earlier."
    else:
        evidence_text = f"The {phrase} is not visible in the current memory."

    carry_text = ""
    carried_keys = current_slot.carried_keys()
    if carried_keys and primary_arm is not None:
        carried_name = _format_object_name(metadata, carried_keys[-1])
        carry_text = f" The {carried_name} is currently held by the {primary_arm} hand."

    action_text = ""
    if "stage3_chunk" in current_slot.roles:
        action_text = " The robot is now executing the task."

    info_text = "Info sufficient." if info_complete else "Info incomplete."
    think = (
        f"{memory_summary} "
        f"Past actions: {past_actions}. "
        f'The current task is "{instruction}". '
        f"The target object is the {phrase}. "
        f"{evidence_text} "
        f"{info_text}"
        f"{carry_text}"
        f"{action_text} "
        f"Next: {camera_field}."
    ).strip()
    info_field = "1" if info_complete else "0"
    frame_field = _object_search_frame_field(snapshot, info_complete=info_complete)
    action_field = _build_stage_action_chunk_text(_build_stage_action_chunk_from_slot(snapshot)) if "stage3_chunk" in current_slot.roles else ""
    return (
        f"<think>{think}</think>"
        f"<info>{info_field}</info>"
        f"<frame>{frame_field}</frame>"
        f"<camera>{camera_field}</camera>"
        f"<action>{action_field}</action>"
    )


def _build_stage_action_chunk(context: EpisodeContext, slot: MemorySlot) -> list[list[float]]:
    row_dim = int(context.left_arm_actions.shape[1] if context.left_arm_actions.ndim == 2 else 0) + int(
        context.right_arm_actions.shape[1] if context.right_arm_actions.ndim == 2 else 0
    )
    if not slot.action_chunk_frame_indices:
        return [[0.0] * row_dim for _ in range(int(context.action_chunk_size))]

    action_chunk: list[list[float]] = []
    for frame_idx in slot.action_chunk_frame_indices:
        left = []
        right = []
        if int(frame_idx) < context.left_arm_actions.shape[0]:
            left = np.array(context.left_arm_actions[int(frame_idx)], dtype=np.float64).tolist()
        if int(frame_idx) < context.right_arm_actions.shape[0]:
            right = np.array(context.right_arm_actions[int(frame_idx)], dtype=np.float64).tolist()
        row = list(left) + list(right)
        if not row:
            row = [0.0] * row_dim
        action_chunk.append(row)
    while len(action_chunk) < int(context.action_chunk_size):
        action_chunk.append(list(action_chunk[-1] if action_chunk else ([0.0] * row_dim)))
    return action_chunk


def _build_stage_action_chunk_text(action_chunk: list[list[float]]) -> str:
    return json.dumps(action_chunk, ensure_ascii=False, separators=(",", ":"))


def _build_stage_action_chunk_from_slot(snapshot: EpisodeSnapshot) -> list[list[float]]:
    return snapshot.current_annotation.get("action_chunk", []) or []


def _build_object_search_sample(
    save_dir: Path,
    episode_idx: int,
    metadata: dict[str, Any],
    context: EpisodeContext,
    snapshot: EpisodeSnapshot,
) -> dict[str, Any]:
    stage_action_chunk = None
    if "stage3_chunk" in snapshot.roles:
        stage_action_chunk = _build_stage_action_chunk(context, snapshot.current_slot)
        snapshot.current_slot.current_annotation["action_chunk"] = stage_action_chunk

    images = _history_or_current_images(save_dir, episode_idx, context, snapshot.prompt_frame_indices)
    sample = {
        "images": images,
        "messages": _build_messages(
            _build_object_search_user_prompt(metadata, snapshot),
            _render_object_search_response(metadata, snapshot),
        ),
        "metadata": {
            "episode_idx": int(episode_idx),
            "task_name": str(metadata.get("task_name", "")),
            "task_type": "object_search",
            "subtask_id": int(snapshot.subtask_id),
            "stage": int(snapshot.stage),
            "roles": list(snapshot.roles),
            "current_frame_idx": int(snapshot.current_frame_idx),
            "prompt_frame_indices": list(snapshot.prompt_frame_indices),
            "prompt_image_count": int(len(images)),
            "prompt_planned_actions": _metadata_action_pairs(snapshot.prompt_planned_actions),
            "prompt_view_deltas": _metadata_action_pairs(snapshot.prompt_planned_actions),
            "memory_support_ready": bool(snapshot.memory_support_ready),
            "evidence_from_history": bool(snapshot.evidence_from_history),
            "evidence_frame_idx": snapshot.evidence_frame_idx,
            "evidence_prompt_index": snapshot.evidence_prompt_index,
            "evidence_uv_norm": snapshot.evidence_uv_norm,
            "camera_delta_deg": int(_search_camera_delta(snapshot)),
            "action_chunk_size": int(context.action_chunk_size),
            "action_chunk_actual_size": int(snapshot.current_slot.action_chunk_actual_size),
            "action_chunk_pad_count": int(snapshot.current_slot.action_chunk_pad_count),
        },
    }
    if stage_action_chunk is not None:
        sample["action"] = stage_action_chunk
    return sample


def _build_angle_delta_user_prompt(metadata: dict[str, Any], previous_slot: MemorySlot, current_slot: MemorySlot) -> str:
    return (
        '<image><image>'
        "The first image is a history frame and the second image is the current frame. "
        "The two images are ordered from earlier to later. "
        "Please estimate the rotation difference from the history frame to the current frame. "
        "Your response should be in the format of: <think>...</think><camera>...</camera>."
    )


def _render_angle_delta_response(rotation_difference: tuple[int, int]) -> str:
    think = (
        "Frames: 2 total (1 history + current). "
        f"From frame 1 to frame 2, the rotation difference is ({int(rotation_difference[0])}, {int(rotation_difference[1])})."
    )
    return f"<think>{think}</think><camera>{_format_rotate_text(rotation_difference)}</camera>"


def _dedup_slots_by_frame(slots: list[MemorySlot]) -> list[MemorySlot]:
    deduped: list[MemorySlot] = []
    for slot in sorted(slots, key=lambda item: (int(item.frame_idx), int(item.slot_idx))):
        if deduped and int(deduped[-1].frame_idx) == int(slot.frame_idx):
            deduped[-1] = slot
            continue
        deduped.append(slot)
    return deduped


def _collect_angle_delta_pairs(slots: list[MemorySlot]) -> list[tuple[MemorySlot, MemorySlot, tuple[int, int]]]:
    grouped: dict[int, list[MemorySlot]] = {}
    for slot in slots:
        if "stage3_chunk" in slot.roles or int(slot.stage) > 2:
            continue
        grouped.setdefault(int(slot.subtask_id), []).append(slot)

    pairs: list[tuple[MemorySlot, MemorySlot, tuple[int, int]]] = []
    for subtask_slots in grouped.values():
        ordered_slots = _dedup_slots_by_frame(subtask_slots)
        if len(ordered_slots) < 2:
            continue
        anchor_slot = ordered_slots[0]
        for current_slot in ordered_slots[1:]:
            if int(current_slot.frame_idx) == int(anchor_slot.frame_idx):
                continue
            pairs.append((anchor_slot, current_slot, _slot_view_delta(anchor_slot, current_slot)))
    return pairs


def _build_angle_delta_sample(
    save_dir: Path,
    episode_idx: int,
    metadata: dict[str, Any],
    context: EpisodeContext,
    previous_slot: MemorySlot,
    current_slot: MemorySlot,
    angle_delta_pair: tuple[int, int],
) -> dict[str, Any]:
    frame_indices = [int(previous_slot.frame_idx), int(current_slot.frame_idx)]
    images = _history_or_current_images(save_dir, episode_idx, context, frame_indices)
    return {
        "images": images,
        "messages": _build_messages(
            _build_angle_delta_user_prompt(metadata, previous_slot, current_slot),
            _render_angle_delta_response(angle_delta_pair),
        ),
        "metadata": {
            "episode_idx": int(episode_idx),
            "task_name": str(metadata.get("task_name", "")),
            "task_type": "angle_delta",
            "subtask_id": int(current_slot.subtask_id),
            "frame_indices": frame_indices,
            "source_slot_frame_indices": [int(previous_slot.frame_idx), int(current_slot.frame_idx)],
            "angle_delta_deg": int(angle_delta_pair[0]),
            "camera_delta_pair": _metadata_action_pairs([angle_delta_pair])[0],
            "view_rotation_difference": _metadata_action_pairs([angle_delta_pair])[0],
        },
    }


def _select_spread_slots(slots: list[MemorySlot], count: int) -> list[MemorySlot]:
    if count <= 0:
        return []
    if count >= len(slots):
        return list(slots)
    target_positions = np.linspace(0, len(slots) - 1, num=count)
    used: set[int] = set()
    selected: list[MemorySlot] = []
    for target in target_positions.tolist():
        candidate_order = sorted(range(len(slots)), key=lambda idx: (abs(idx - target), idx))
        for candidate_idx in candidate_order:
            if candidate_idx in used:
                continue
            used.add(candidate_idx)
            selected.append(slots[candidate_idx])
            break
    selected.sort(key=lambda slot: int(slot.frame_idx))
    return selected


def _slot_identity(slot: MemorySlot) -> tuple[int, int]:
    return int(slot.frame_idx), int(slot.slot_idx)


def _ordered_slot_subset(before_slots: list[MemorySlot], slot_set: set[tuple[int, int]]) -> list[MemorySlot]:
    return [slot for slot in before_slots if _slot_identity(slot) in slot_set]


def _slot_positions(sample_slots: list[MemorySlot], kept_slots: list[MemorySlot]) -> list[int]:
    frame_to_position = {_slot_identity(slot): idx + 1 for idx, slot in enumerate(sample_slots)}
    positions: list[int] = []
    for slot in kept_slots:
        position = frame_to_position.get(_slot_identity(slot), None)
        if position is not None:
            positions.append(int(position))
    return positions


def _slot_has_object_evidence(slot: MemorySlot, object_key: str) -> bool:
    uv = slot.uv_for_object_key(object_key)
    if isinstance(uv, (list, tuple)) and len(uv) >= 2 and float(uv[0]) >= 0.0 and float(uv[1]) >= 0.0:
        return True
    discovered_map = slot.discovered_last_uv_map()
    if str(object_key) in discovered_map:
        discovered_uv = discovered_map[str(object_key)]
        if isinstance(discovered_uv, (list, tuple)) and len(discovered_uv) >= 2:
            return True
    return str(object_key) in set(slot.visible_keys()) or str(object_key) in set(slot.discovered_keys())


def _latest_valid_evidence_positions(sample_slots: list[MemorySlot]) -> list[int]:
    if not sample_slots:
        return []
    latest_slot = sample_slots[-1]
    target_keys = [str(key) for key in latest_slot.target_keys() if str(key).strip()]
    if not target_keys:
        return [len(sample_slots)]
    evidence_positions: set[int] = set()
    for object_key in target_keys:
        latest_position = None
        for position, slot in enumerate(sample_slots, start=1):
            if _slot_has_object_evidence(slot, object_key):
                latest_position = int(position)
        if latest_position is not None:
            evidence_positions.add(int(latest_position))
    if not evidence_positions:
        evidence_positions.add(len(sample_slots))
    return sorted(evidence_positions)


def _compression_sequence_sentence(
    sample_slots: list[MemorySlot],
    kept_positions: list[int],
    past_actions: list[tuple[int, int]],
) -> str:
    has_zero = any(int(dx) == 0 and int(dy) == 0 for dx, dy in past_actions)
    has_removed = len(kept_positions) < len(sample_slots)
    has_nonzero = any(int(dx) != 0 or int(dy) != 0 for dx, dy in past_actions)
    if has_zero and has_nonzero and has_removed:
        return "This sequence mixes exploration, revisits, and stable observation."
    if has_zero and not has_nonzero:
        return "This sequence is mostly stable observation."
    if has_removed:
        return "This sequence mixes exploration and revisits."
    return "This sequence is mostly exploration."


def _compression_replacement_summary(sample_slots: list[MemorySlot], kept_slots: list[MemorySlot]) -> str:
    kept_positions = _slot_positions(sample_slots=sample_slots, kept_slots=kept_slots)
    kept_position_set = set(kept_positions)
    removed_positions = [idx + 1 for idx in range(len(sample_slots)) if (idx + 1) not in kept_position_set]
    if not removed_positions:
        return "none."

    position_to_slot = {idx + 1: slot for idx, slot in enumerate(sample_slots)}
    grid = _coverage_grid()
    coverage_cache = {
        position: _slot_coverage_indices(position_to_slot[position], grid=grid, half_fov_deg=DEFAULT_MEMORY_FOV_HALF_DEG)
        for position in range(1, len(sample_slots) + 1)
    }
    replacement_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for removed_position in removed_positions:
        removed_coverage = coverage_cache.get(removed_position, set())
        replacement_position = None
        replacement_score = (-1, -1)
        for kept_position in kept_positions:
            if kept_position <= removed_position:
                continue
            overlap = len(removed_coverage & coverage_cache.get(kept_position, set()))
            if overlap <= 0:
                continue
            score = (overlap, kept_position)
            if score > replacement_score:
                replacement_position = kept_position
                replacement_score = score
        if replacement_position is None and kept_positions:
            later_positions = [position for position in kept_positions if position > removed_position]
            if later_positions:
                replacement_position = later_positions[-1]
        if replacement_position is None:
            continue
        pair = (int(replacement_position), int(removed_position))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        replacement_pairs.append(pair)

    if not replacement_pairs:
        return "none."

    pair_text = "; ".join(f"frame {newer} over frame {older}" for newer, older in replacement_pairs)
    reason = "because the newer frames revisit the same area with newer valid memory."
    return f"{pair_text} {reason}"


def _compression_protection_summary(kept_positions: list[int], evidence_positions: list[int]) -> str:
    protected_positions = [position for position in evidence_positions if position in set(kept_positions) and position != max(kept_positions)]
    if not protected_positions:
        return "none."
    if len(protected_positions) == 1:
        return f"keep frame {protected_positions[0]} as the latest valid task evidence."
    return f"keep frames {_format_frame_field(protected_positions)} as the latest valid task evidence."


def _subset_variant_indices(
    discarded_slots: list[MemorySlot],
    extras_needed: int,
    max_variants: int = MAX_COMPRESSION_VARIANTS_PER_SIZE,
) -> list[tuple[str, tuple[int, ...]]]:
    if extras_needed <= 0:
        return [("kept_only", tuple())]

    total_discarded = len(discarded_slots)
    if extras_needed > total_discarded:
        return []

    max_variants = max(int(max_variants), 1)
    total_combinations = math.comb(total_discarded, extras_needed)
    variants: dict[tuple[int, ...], str] = {}

    def _register(name: str, indices: list[int] | tuple[int, ...]) -> None:
        key = tuple(sorted(int(idx) for idx in indices))
        if len(key) != extras_needed:
            return
        if any(idx < 0 or idx >= total_discarded for idx in key):
            return
        variants.setdefault(key, str(name))

    _register("oldest", tuple(range(extras_needed)))
    _register("newest", tuple(range(total_discarded - extras_needed, total_discarded)))
    spread_slots = _select_spread_slots(discarded_slots, extras_needed)
    spread_indices = [discarded_slots.index(slot) for slot in spread_slots if slot in discarded_slots]
    _register("spread", spread_indices)

    if total_combinations <= max_variants:
        for combo in combinations(range(total_discarded), extras_needed):
            _register(f"combo_{len(variants):03d}", combo)
    else:
        seed = int(
            sum((idx + 1) * int(slot.frame_idx) for idx, slot in enumerate(discarded_slots))
            + extras_needed * 1009
            + total_discarded * 917
        )
        rng = np.random.default_rng(seed)
        target_count = min(max_variants, total_combinations)
        while len(variants) < int(target_count):
            choice = tuple(sorted(int(idx) for idx in rng.choice(total_discarded, size=extras_needed, replace=False).tolist()))
            _register(f"sample_{len(variants):03d}", choice)

    return [(name, key) for key, name in sorted(variants.items())]


def _compression_subset_variants(
    event: CompressionEvent,
) -> list[tuple[str, list[MemorySlot], list[MemorySlot]]]:
    before_slots = list(event.before_slots)
    optimal_slots = compress_memory_slots(before_slots)
    optimal_slot_set = {_slot_identity(slot) for slot in optimal_slots}
    discarded_slots = [slot for slot in before_slots if _slot_identity(slot) not in optimal_slot_set]
    min_size = max(len(optimal_slots), 4)
    max_size = min(len(before_slots), DEFAULT_MAX_CONTEXT_FRAMES)
    variants: dict[tuple[tuple[int, int], ...], tuple[str, list[MemorySlot], list[MemorySlot]]] = {}

    for size in range(int(min_size), int(max_size) + 1):
        extras_needed = int(size - len(optimal_slots))
        if extras_needed < 0 or extras_needed > len(discarded_slots):
            continue
        for variant_name, extra_indices in _subset_variant_indices(discarded_slots, extras_needed):
            chosen_slot_set = set(optimal_slot_set)
            chosen_slot_set.update(_slot_identity(discarded_slots[idx]) for idx in extra_indices)
            sample_slots = _ordered_slot_subset(before_slots, chosen_slot_set)
            recomputed_optimal = compress_memory_slots(sample_slots)
            if len(recomputed_optimal) >= len(sample_slots):
                continue
            key = tuple(_slot_identity(slot) for slot in sample_slots)
            variants.setdefault(key, (variant_name, sample_slots, recomputed_optimal))
    return [variants[key] for key in sorted(variants.keys())]


def _build_memory_compression_user_prompt(metadata: dict[str, Any], sample_slots: list[MemorySlot]) -> str:
    latest_slot = sample_slots[-1]
    phrase = _resolve_target_phrase(metadata, latest_slot, with_articles=True)
    image_tokens = "".join("<image>" for _ in sample_slots)
    return (
        f'{image_tokens}Your task is: "Track only the useful memory for {phrase}." '
        "The input images are ordered from earliest to latest, and the last image is the current view. "
        "Please keep the most relevant and reliable frames, and output the filtered frames. "
        "Your response should be in the format of: "
        "<think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>."
    )


def _render_memory_compression_response(
    metadata: dict[str, Any],
    sample_slots: list[MemorySlot],
    kept_slots: list[MemorySlot],
) -> str:
    kept_positions = _slot_positions(sample_slots=sample_slots, kept_slots=kept_slots)
    latest_slot = sample_slots[-1]
    past_actions = _slot_sequence_view_deltas(sample_slots)
    evidence_positions = _latest_valid_evidence_positions(sample_slots)
    object_phrase = _resolve_target_phrase(metadata, latest_slot, with_articles=True)
    think = (
        f"{_format_frame_summary(len(sample_slots))} "
        f"Past actions: {_format_rotation_pairs(past_actions)}. "
        f"{_compression_sequence_sentence(sample_slots, kept_positions, past_actions)} "
        f"Spatially, keep frames {_format_frame_field(kept_positions)} for distinct coverage. "
        f"Replacement: {_compression_replacement_summary(sample_slots, kept_slots)} "
        f"Protection: {_compression_protection_summary(kept_positions, evidence_positions)} "
        f"Latest valid task evidence comes from frames {_format_frame_field(evidence_positions)}. "
        f"Keep frames {_format_frame_field(kept_positions)}. "
        f"Info is sufficient now. The latest valid observations still cover {object_phrase}."
    )
    frame_field = _format_frame_field(kept_positions)
    return (
        f"<think>{think}</think>"
        "<info>1</info>"
        f"<frame>{frame_field}</frame>"
        "<camera></camera>"
        "<action></action>"
    )


def _build_memory_compression_sample(
    save_dir: Path,
    episode_idx: int,
    metadata: dict[str, Any],
    context: EpisodeContext,
    event: CompressionEvent,
    variant_name: str,
    sample_slots: list[MemorySlot],
    kept_slots: list[MemorySlot],
) -> dict[str, Any]:
    frame_indices = [int(slot.frame_idx) for slot in sample_slots]
    kept_frame_indices = [int(slot.frame_idx) for slot in kept_slots]
    images = _history_or_current_images(save_dir, episode_idx, context, frame_indices)
    return {
        "images": images,
        "messages": _build_messages(
            _build_memory_compression_user_prompt(metadata, sample_slots),
            _render_memory_compression_response(metadata, sample_slots, kept_slots),
        ),
        "metadata": {
            "episode_idx": int(episode_idx),
            "task_name": str(metadata.get("task_name", "")),
            "task_type": "memory_compression_vqa",
            "trigger": str(event.trigger),
            "trigger_frame_idx": int(event.trigger_frame_idx),
            "variant": str(variant_name),
            "prompt_image_count": int(len(images)),
            "prompt_frame_indices": frame_indices,
            "prompt_view_deltas": _metadata_action_pairs(_slot_sequence_view_deltas(sample_slots)),
            "before_frame_indices": frame_indices,
            "optimal_frame_indices": kept_frame_indices,
            "removed_input_positions": [
                idx + 1 for idx, frame_idx in enumerate(frame_indices) if int(frame_idx) not in set(kept_frame_indices)
            ],
        },
    }


def _collect_episode_pairs(save_dir: Path) -> list[tuple[int, Path, Path]]:
    metadata_dir = save_dir / "subtask_metadata"
    data_dir = save_dir / "data"
    if not metadata_dir.exists():
        return []
    pairs: list[tuple[int, Path, Path]] = []
    for metadata_path in sorted(metadata_dir.glob("episode*.json")):
        match = re.search(r"episode(\d+)\.json$", metadata_path.name)
        if match is None:
            continue
        episode_idx = int(match.group(1))
        hdf5_path = data_dir / f"episode{episode_idx}.hdf5"
        if hdf5_path.exists():
            pairs.append((episode_idx, metadata_path, hdf5_path))
    return pairs


def export_task_vlm_dataset(
    save_dir: str,
    overwrite: bool = True,
    max_context_frames: int = DEFAULT_MAX_CONTEXT_FRAMES,
    action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
    task_types: list[str] | None = None,
) -> dict[str, Any]:
    save_path = Path(save_dir)
    task_types = list(REGISTERED_TASK_TYPES if task_types is None else task_types)
    output_dir = save_path / "vlm"
    output_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        task_type: _JsonArrayWriter(output_dir / f"{task_type}.json", overwrite=overwrite)
        for task_type in task_types
    }
    samples_paths = {task_type: str(output_dir / f"{task_type}.json") for task_type in task_types}
    task_type_counts = {task_type: 0 for task_type in task_types}
    total = 0
    episode_count = 0
    annotated_video_count = 0
    annotated_video_frame_count = 0

    try:
        for episode_idx, metadata_path, hdf5_path in _collect_episode_pairs(save_path):
            metadata = _read_json(metadata_path)
            context = build_episode_context(
                metadata=metadata,
                hdf5_path=str(hdf5_path),
                action_chunk_size=int(action_chunk_size),
                max_context_frames=int(max_context_frames),
            )
            episode_count += 1
            annotated_video_frame_count += int(len(context.frames))

            annotated_video_path = metadata.get("annotated_video_path", None) or default_annotated_video_path(str(save_path), episode_idx)
            if not Path(annotated_video_path).exists():
                export_annotated_video(context.frames, list(metadata.get("frame_annotations", []) or []), annotated_video_path)
            if Path(annotated_video_path).exists():
                annotated_video_count += 1

            if "object_search" in writers:
                for snapshot in context.snapshots:
                    writers["object_search"].append(
                        _build_object_search_sample(save_path, episode_idx, metadata, context, snapshot)
                    )

            if "angle_delta" in writers:
                for previous_slot, current_slot, angle_delta_deg in _collect_angle_delta_pairs(context.memory_slots):
                    writers["angle_delta"].append(
                        _build_angle_delta_sample(
                            save_path,
                            episode_idx,
                            metadata,
                            context,
                            previous_slot,
                            current_slot,
                            angle_delta_deg,
                        )
                    )

            if "memory_compression_vqa" in writers:
                for event in context.compression_events:
                    for variant_name, sample_slots, kept_slots in _compression_subset_variants(event):
                        writers["memory_compression_vqa"].append(
                            _build_memory_compression_sample(
                                save_dir=save_path,
                                episode_idx=episode_idx,
                                metadata=metadata,
                                context=context,
                                event=event,
                                variant_name=variant_name,
                                sample_slots=sample_slots,
                                kept_slots=kept_slots,
                            )
                        )
        for task_type, writer in writers.items():
            writer.commit()
            task_type_counts[task_type] = int(writer.count)
            total += int(writer.count)
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise

    manifest = {
        "episode_count": int(episode_count),
        "annotated_video_count": int(annotated_video_count),
        "annotated_video_frame_count": int(annotated_video_frame_count),
        "action_chunk_size": int(action_chunk_size),
        "max_context_frames": int(max_context_frames),
    }
    _write_json(output_dir / "manifest.json", manifest)

    return {
        "sample_count": int(total),
        "task_type_counts": task_type_counts,
        "samples_paths": samples_paths,
    }
