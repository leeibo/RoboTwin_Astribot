from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .annotated_video import default_annotated_video_path, export_annotated_video
from .models import CompressionEvent, EpisodeContext, EpisodeSnapshot, MemorySlot
from .snapshots import build_episode_context


DEFAULT_MAX_CONTEXT_FRAMES = 16
DEFAULT_ACTION_CHUNK_SIZE = 10
REGISTERED_TASK_TYPES = ("object_search", "angle_delta", "memory_compression_vqa")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return str(path)


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


def _format_action_pairs(actions: list[tuple[int, int]]) -> str:
    if not actions:
        return "none"
    return str([(int(a), int(b)) for a, b in actions])


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
    phrase = _resolve_search_phrase(metadata, current_slot)
    total_frames = len(snapshot.prompt_frame_indices)
    history_frames = max(total_frames - 1, 0)
    frames_text = "Frames: current only." if total_frames <= 1 else f"Frames: {total_frames} total ({history_frames} history + current)."
    past_actions_text = f"Past actions: {_format_action_pairs(snapshot.prompt_planned_actions)}."
    camera_delta = _search_camera_delta(snapshot)
    info_complete = _search_info_complete(snapshot)
    has_evidence_frame = snapshot.evidence_prompt_index is not None and snapshot.evidence_uv_norm is not None
    primary_arm = _infer_primary_arm(metadata)

    if info_complete and has_evidence_frame and snapshot.evidence_from_history:
        evidence_text = (
            f"在第{int(snapshot.evidence_prompt_index)}帧{_format_uv_1000(snapshot.evidence_uv_norm)}中发现了目标物体the {phrase}。"
            f"信息已足够。Next: Rotate({int(camera_delta)}, 0)."
        )
    elif info_complete and has_evidence_frame:
        evidence_text = (
            f"当前视角中发现了目标物体the {phrase}，位置约在{_format_uv_1000(snapshot.evidence_uv_norm)}。"
            f"信息已足够。Next: Rotate({int(camera_delta)}, 0)."
        )
    elif info_complete:
        evidence_text = (
            f"目标物体the {phrase}已在历史搜索阶段完成定位，当前根据已有定位信息继续执行。"
            f"信息已足够。Next: Rotate({int(camera_delta)}, 0)."
        )
    else:
        evidence_text = f"当前视角中还没看到the {phrase}。信息不足。Next: Rotate({int(camera_delta)}, 0)."

    carry_text = ""
    carried_keys = current_slot.carried_keys()
    if carried_keys and primary_arm is not None:
        carried_name = _format_object_name(metadata, carried_keys[-1])
        carry_text = f" the {carried_name} is held by the {primary_arm} hand."

    action_text = ""
    if "stage3_chunk" in current_slot.roles:
        action_text = " 已经位于中心附近，正在执行动作操作。"

    think = f"{frames_text} {past_actions_text} {evidence_text}{carry_text}{action_text}".strip()
    info_field = "1" if info_complete else "0"
    frame_field = str(int(snapshot.evidence_prompt_index)) if info_complete and snapshot.evidence_prompt_index is not None else "[]"
    camera_field = f"Rotate({int(camera_delta)}, 0)"
    action_field = "<action_chunk>" if "stage3_chunk" in current_slot.roles else ""
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


def _build_object_search_sample(
    save_dir: Path,
    episode_idx: int,
    metadata: dict[str, Any],
    context: EpisodeContext,
    snapshot: EpisodeSnapshot,
) -> dict[str, Any]:
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
            "prompt_planned_actions": _metadata_action_pairs(snapshot.prompt_planned_actions),
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
    if "stage3_chunk" in snapshot.roles:
        sample["action"] = _build_stage_action_chunk(context, snapshot.current_slot)
    return sample


def _angle_direction_text(angle_delta_deg: int) -> str:
    if angle_delta_deg > 0:
        return f"向左旋转{abs(int(angle_delta_deg))}度"
    if angle_delta_deg < 0:
        return f"向右旋转{abs(int(angle_delta_deg))}度"
    return "无需水平旋转"


def _build_angle_delta_user_prompt(metadata: dict[str, Any], previous_slot: MemorySlot, current_slot: MemorySlot) -> str:
    instruction = _subtask_instruction(metadata, current_slot.subtask_id)
    return (
        '<image><image>'
        f'给定同一子任务的历史帧与当前帧，任务是"{instruction}"。'
        "输入图像按时间从早到晚排序。请根据两张图回答：从历史帧到当前帧，机器人累计水平转了多少度？"
        "请输出 <think>...</think><answer>...</answer>。"
    )


def _render_angle_delta_response(previous_slot: MemorySlot, current_slot: MemorySlot) -> str:
    angle_delta_deg = int(round(float(previous_slot.planned_delta_deg)))
    direction_text = _angle_direction_text(angle_delta_deg)
    think = (
        "这两张图像来自同一个子任务，并且按时间顺序排列。"
        f"从历史帧到当前帧，中间累计的水平转动约为[{(int(angle_delta_deg), 0)}]。"
        f"因此从历史帧到当前帧，总共需要{direction_text}。"
    )
    answer = (
        f"角度差值是{abs(int(angle_delta_deg))}度，表示需要{'向左旋转' if angle_delta_deg > 0 else ('向右旋转' if angle_delta_deg < 0 else '保持不动')}。"
    )
    return f"<think>{think}</think><answer>{answer}</answer>"


def _build_angle_delta_sample(
    save_dir: Path,
    episode_idx: int,
    metadata: dict[str, Any],
    context: EpisodeContext,
    previous_slot: MemorySlot,
    current_slot: MemorySlot,
) -> dict[str, Any]:
    frame_indices = [int(previous_slot.frame_idx), int(current_slot.frame_idx)]
    images = _history_or_current_images(save_dir, episode_idx, context, frame_indices)
    angle_delta_deg = int(round(float(previous_slot.planned_delta_deg)))
    return {
        "images": images,
        "messages": _build_messages(
            _build_angle_delta_user_prompt(metadata, previous_slot, current_slot),
            _render_angle_delta_response(previous_slot, current_slot),
        ),
        "metadata": {
            "episode_idx": int(episode_idx),
            "task_name": str(metadata.get("task_name", "")),
            "task_type": "angle_delta",
            "subtask_id": int(current_slot.subtask_id),
            "frame_indices": frame_indices,
            "source_slot_frame_indices": [int(previous_slot.frame_idx), int(current_slot.frame_idx)],
            "angle_delta_deg": int(angle_delta_deg),
            "planned_actions": _metadata_action_pairs([(int(angle_delta_deg), 0)]),
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


def _compression_subset_variants(event: CompressionEvent) -> list[tuple[str, list[MemorySlot]]]:
    before_slots = list(event.before_slots)
    kept_frame_set = {int(slot.frame_idx) for slot in event.after_slots}
    discarded_slots = [slot for slot in before_slots if int(slot.frame_idx) not in kept_frame_set]
    min_size = max(len(event.after_slots), 4)
    variants: dict[tuple[int, ...], tuple[str, list[MemorySlot]]] = {}

    for size in range(int(min_size), len(before_slots) + 1):
        extras_needed = int(size - len(event.after_slots))
        if extras_needed < 0 or extras_needed > len(discarded_slots):
            continue

        candidates: list[tuple[str, list[MemorySlot]]] = []
        if extras_needed == 0:
            candidates.append(("kept_only", list(event.after_slots)))
        else:
            candidates.append(("spread", _select_spread_slots(discarded_slots, extras_needed)))
            candidates.append(("newest", list(discarded_slots[-extras_needed:])))
            candidates.append(("oldest", list(discarded_slots[:extras_needed])))

        for variant_name, extra_slots in candidates:
            slot_set = {int(slot.frame_idx) for slot in event.after_slots}
            slot_set.update(int(slot.frame_idx) for slot in extra_slots)
            sample_slots = [slot for slot in before_slots if int(slot.frame_idx) in slot_set]
            key = tuple(int(slot.frame_idx) for slot in sample_slots)
            variants.setdefault(key, (variant_name, sample_slots))
    return [variants[key] for key in sorted(variants.keys())]


def _build_memory_compression_user_prompt(metadata: dict[str, Any], sample_slots: list[MemorySlot]) -> str:
    latest_slot = sample_slots[-1]
    phrase = _resolve_search_phrase(metadata, latest_slot)
    image_tokens = "".join("<image>" for _ in sample_slots)
    return (
        f'{image_tokens}Your task is: "Build a concise visual memory for {phrase}." '
        "The input images are ordered from earliest to latest, and the last image is the current view. "
        "Please remove redundant memory views while preserving the most relevant target evidence. "
        "Your response should be in the format of: "
        "<think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>."
    )


def _render_memory_compression_response(
    sample_slots: list[MemorySlot],
    kept_slots: list[MemorySlot],
) -> str:
    frame_to_position = {int(slot.frame_idx): idx + 1 for idx, slot in enumerate(sample_slots)}
    kept_positions = [int(frame_to_position[int(slot.frame_idx)]) for slot in kept_slots if int(slot.frame_idx) in frame_to_position]
    removed_positions = [
        idx + 1 for idx, slot in enumerate(sample_slots) if int(slot.frame_idx) not in {int(item.frame_idx) for item in kept_slots}
    ]
    view_changes = [(int(round(float(slot.planned_delta_deg))), 0) for slot in sample_slots[:-1]]
    think = (
        f"Frames: {len(sample_slots)} total. "
        f"View changes: {view_changes}. "
        f"为了尽可能保留更大的视野覆盖，并在重叠区域优先保留新帧，我保留帧{kept_positions}。 "
        f"帧{removed_positions}与这些保留帧高度重叠，或者只是更旧的弱证据，因此被移除。 Info sufficient."
    )
    return (
        f"<think>{think}</think>"
        "<info>1</info>"
        f"<frame>{kept_positions}</frame>"
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
) -> dict[str, Any]:
    frame_indices = [int(slot.frame_idx) for slot in sample_slots]
    kept_frame_indices = [int(slot.frame_idx) for slot in event.after_slots]
    images = _history_or_current_images(save_dir, episode_idx, context, frame_indices)
    return {
        "images": images,
        "messages": _build_messages(
            _build_memory_compression_user_prompt(metadata, sample_slots),
            _render_memory_compression_response(sample_slots, event.after_slots),
        ),
        "metadata": {
            "episode_idx": int(episode_idx),
            "task_name": str(metadata.get("task_name", "")),
            "task_type": "memory_compression_vqa",
            "trigger": str(event.trigger),
            "trigger_frame_idx": int(event.trigger_frame_idx),
            "variant": str(variant_name),
            "before_frame_indices": frame_indices,
            "optimal_frame_indices": kept_frame_indices,
            "removed_input_positions": [
                idx for idx, frame_idx in enumerate(frame_indices) if int(frame_idx) not in set(kept_frame_indices)
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
    samples_by_type = {task_type: [] for task_type in task_types}
    episode_contexts: list[EpisodeContext] = []

    for episode_idx, metadata_path, hdf5_path in _collect_episode_pairs(save_path):
        metadata = _read_json(metadata_path)
        context = build_episode_context(
            metadata=metadata,
            hdf5_path=str(hdf5_path),
            action_chunk_size=int(action_chunk_size),
            max_context_frames=int(max_context_frames),
        )
        episode_contexts.append(context)

        annotated_video_path = metadata.get("annotated_video_path", None) or default_annotated_video_path(str(save_path), episode_idx)
        if not Path(annotated_video_path).exists():
            export_annotated_video(context.frames, list(metadata.get("frame_annotations", []) or []), annotated_video_path)

        if "object_search" in samples_by_type:
            for snapshot in context.snapshots:
                samples_by_type["object_search"].append(
                    _build_object_search_sample(save_path, episode_idx, metadata, context, snapshot)
                )

        if "angle_delta" in samples_by_type:
            non_stage3_slots = [slot for slot in context.memory_slots if "stage3_chunk" not in slot.roles]
            for previous_slot, current_slot in zip(non_stage3_slots[:-1], non_stage3_slots[1:]):
                if int(previous_slot.subtask_id) != int(current_slot.subtask_id):
                    continue
                samples_by_type["angle_delta"].append(
                    _build_angle_delta_sample(save_path, episode_idx, metadata, context, previous_slot, current_slot)
                )

        if "memory_compression_vqa" in samples_by_type:
            for event in context.compression_events:
                for variant_name, sample_slots in _compression_subset_variants(event):
                    samples_by_type["memory_compression_vqa"].append(
                        _build_memory_compression_sample(
                            save_dir=save_path,
                            episode_idx=episode_idx,
                            metadata=metadata,
                            context=context,
                            event=event,
                            variant_name=variant_name,
                            sample_slots=sample_slots,
                        )
                    )

    output_dir = save_path / "vlm"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_paths: dict[str, str] = {}
    task_type_counts: dict[str, int] = {}
    total = 0

    for task_type, samples in samples_by_type.items():
        output_path = output_dir / f"{task_type}.json"
        if not overwrite and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as file:
                existing = json.load(file)
            samples = list(existing) + list(samples)
        samples_paths[task_type] = _write_json(output_path, samples)
        task_type_counts[task_type] = len(samples)
        total += len(samples)

    manifest = {
        "episode_count": len(episode_contexts),
        "annotated_video_count": len(list((save_path / "video").glob("episode*_annotated.mp4"))),
        "annotated_video_frame_count": int(sum(len(context.frames) for context in episode_contexts)),
        "action_chunk_size": int(action_chunk_size),
        "max_context_frames": int(max_context_frames),
    }
    _write_json(output_dir / "manifest.json", manifest)

    return {
        "sample_count": int(total),
        "task_type_counts": task_type_counts,
        "samples_paths": samples_paths,
    }
