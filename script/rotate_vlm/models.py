from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemorySlot:
    slot_idx: int
    frame_idx: int
    subtask_id: int
    stage: int
    current_annotation: dict[str, Any]
    current_heading_deg: float
    planned_delta_deg: float
    planned_heading_deg: float
    roles: list[str]
    action_chunk_frame_indices: list[int] = field(default_factory=list)
    action_chunk_actual_size: int = 0
    action_chunk_pad_count: int = 0

    @property
    def frame_number(self) -> int:
        return int(self.frame_idx)

    def target_keys(self) -> list[str]:
        ann = self.current_annotation
        keys = ann.get("search_target_keys", None) or []
        if keys:
            return [str(key) for key in keys]
        keys = ann.get("action_target_keys", None) or []
        if keys:
            return [str(key) for key in keys]
        focus_key = ann.get("focus_object_key", None)
        if focus_key:
            return [str(focus_key)]
        return []

    def target_key(self) -> str | None:
        keys = self.target_keys()
        return keys[0] if keys else None

    def target_uv_norm(self) -> list[float] | None:
        val = self.current_annotation.get("target_uv_norm", None)
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            return None
        return [float(val[0]), float(val[1])]

    def visible_object_uv_map(self) -> dict[str, list[float]]:
        raw = self.current_annotation.get("visible_object_uv_map", {}) or {}
        result: dict[str, list[float]] = {}
        if not isinstance(raw, dict):
            return result
        for key, value in raw.items():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            result[str(key)] = [float(value[0]), float(value[1])]
        return result

    def discovered_last_uv_map(self) -> dict[str, list[float]]:
        raw = self.current_annotation.get("discovered_last_uv_map", {}) or {}
        result: dict[str, list[float]] = {}
        if not isinstance(raw, dict):
            return result
        for key, value in raw.items():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            result[str(key)] = [float(value[0]), float(value[1])]
        return result

    def uv_for_object_key(self, object_key: str | None) -> list[float] | None:
        if object_key is None:
            return self.target_uv_norm()
        visible_map = self.visible_object_uv_map()
        if str(object_key) in visible_map:
            return list(visible_map[str(object_key)])
        if self.target_key() == str(object_key):
            return self.target_uv_norm()
        return None

    def visible_keys(self) -> list[str]:
        return [str(key) for key in self.current_annotation.get("visible_object_keys", []) or []]

    def discovered_keys(self) -> list[str]:
        return [str(key) for key in self.current_annotation.get("discovered_object_keys", []) or []]

    def carried_keys(self) -> list[str]:
        return [str(key) for key in self.current_annotation.get("carried_object_keys", []) or []]

    def has_target_evidence(self) -> bool:
        target_key = self.target_key()
        uv = self.uv_for_object_key(target_key)
        if uv is not None and float(uv[0]) >= 0.0 and float(uv[1]) >= 0.0:
            return True
        return bool(target_key is not None and target_key in set(self.visible_keys()))


@dataclass
class EpisodeSnapshot:
    current_slot: MemorySlot
    prompt_slots: list[MemorySlot] = field(default_factory=list)
    prompt_frame_indices: list[int] = field(default_factory=list)
    prompt_planned_actions: list[tuple[int, int]] = field(default_factory=list)
    evidence_frame_idx: int | None = None
    evidence_prompt_index: int | None = None
    evidence_uv_norm: list[float] | None = None
    evidence_from_history: bool = False
    memory_support_ready: bool = False

    @property
    def current_frame_idx(self) -> int:
        return int(self.current_slot.frame_idx)

    @property
    def subtask_id(self) -> int:
        return int(self.current_slot.subtask_id)

    @property
    def stage(self) -> int:
        return int(self.current_slot.stage)

    @property
    def roles(self) -> list[str]:
        return list(self.current_slot.roles)

    @property
    def current_annotation(self) -> dict[str, Any]:
        return dict(self.current_slot.current_annotation)


@dataclass
class CompressionEvent:
    trigger: str
    trigger_frame_idx: int
    before_slots: list[MemorySlot]
    after_slots: list[MemorySlot]


@dataclass
class EpisodeContext:
    metadata: dict[str, Any]
    hdf5_path: str
    action_chunk_size: int
    max_context_frames: int
    frames: list[Any]
    left_arm_actions: Any
    left_gripper_actions: Any
    right_arm_actions: Any
    right_gripper_actions: Any
    memory_slots: list[MemorySlot] = field(default_factory=list)
    snapshots: list[EpisodeSnapshot] = field(default_factory=list)
    compression_events: list[CompressionEvent] = field(default_factory=list)
