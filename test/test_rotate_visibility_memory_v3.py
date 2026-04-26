import re
import sys
from collections import Counter
from math import radians
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.export_rotate_object_search_visibility_memory_v3 import (  # noqa: E402
    _build_stage3_shifted_action_samples,
)
from script.rotate_vlm.models import EpisodeContext  # noqa: E402


def _metadata() -> dict:
    return {
        "task_name": "unit_test_rotate_view",
        "task_instruction": "Find and use the target object.",
        "subtask_instruction_map": {"0": "Find the target object."},
        "object_key_to_name": {"target": "001_target"},
    }


def _annotation(frame_idx: int, stage: int) -> dict:
    return {
        "frame_idx": int(frame_idx),
        "subtask": 0,
        "stage": int(stage),
        "waist_heading_deg": 0.0,
        "camera_target_theta": None,
        "search_target_keys": ["target"],
        "visible_object_keys": ["target"],
        "visible_object_uv_map": {"target": [0.5, 0.5]},
        "target_uv_norm": [0.5, 0.5],
    }


def _hidden_target_annotation(frame_idx: int, stage: int) -> dict:
    annotation = _annotation(frame_idx, stage)
    annotation["visible_object_keys"] = []
    annotation["visible_object_uv_map"] = {}
    annotation["target_uv_norm"] = [-1.0, -1.0]
    return annotation


def _extract_tag(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", content, flags=re.DOTALL)
    return "" if match is None else str(match.group(1))


def test_stage3_v3_shifted_action_lattice_expands_every_frame(tmp_path: Path):
    chunk_size = 16
    annotations = [_annotation(frame_idx, 3) for frame_idx in range(20)]
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(20)]
    context = EpisodeContext(
        metadata=_metadata(),
        hdf5_path="",
        action_chunk_size=chunk_size,
        max_context_frames=16,
        frames=frames,
        left_arm_actions=np.arange(20, dtype=np.float64).reshape(20, 1),
        left_gripper_actions=np.arange(100, 120, dtype=np.float64),
        right_arm_actions=np.arange(200, 220, dtype=np.float64).reshape(20, 1),
        right_gripper_actions=np.arange(300, 320, dtype=np.float64),
    )

    samples = _build_stage3_shifted_action_samples(
        save_dir=tmp_path,
        output_dir_name="vlm_object_search_visibility_memory_v3",
        episode_idx=0,
        metadata=_metadata(),
        context=context,
        annotations=annotations,
        max_context_frames=16,
        action_chunk_size=chunk_size,
    )

    assert len(samples) == 20
    by_frame = {sample["metadata"]["current_frame_idx"]: sample for sample in samples}
    assert by_frame[16]["metadata"]["prompt_frame_indices"] == [0, 16]
    assert by_frame[17]["metadata"]["prompt_frame_indices"] == [1, 17]
    assert by_frame[19]["metadata"]["prompt_frame_indices"] == [3, 19]
    assert by_frame[17]["metadata"]["action_chunk_frame_indices"] == [17, 18, 19]
    assert by_frame[17]["metadata"]["action_chunk_actual_size"] == 3
    assert by_frame[17]["metadata"]["action_chunk_pad_count"] == 13
    assert len(by_frame[17]["action"]) == chunk_size
    assert by_frame[17]["action"][-1] == [19.0, 119.0, 219.0, 319.0]
    assert _extract_tag(by_frame[17]["messages"][1]["content"], "camera") == "Rotate(0, 0)"
    assert _extract_tag(by_frame[17]["messages"][1]["content"], "action")

    offset_counts = Counter(sample["metadata"]["action_lattice_offset"] for sample in samples)
    assert offset_counts == Counter({0: 2, 1: 2, 2: 2, 3: 2, **{idx: 1 for idx in range(4, 16)}})


def test_stage3_v3_shifted_action_lattice_waits_for_trigger_before_compression(tmp_path: Path):
    chunk_size = 16
    annotations = [_annotation(frame_idx, 3) for frame_idx in range(36)]
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(36)]
    context = EpisodeContext(
        metadata=_metadata(),
        hdf5_path="",
        action_chunk_size=chunk_size,
        max_context_frames=16,
        frames=frames,
        left_arm_actions=np.zeros((36, 1), dtype=np.float64),
        left_gripper_actions=np.zeros((36,), dtype=np.float64),
        right_arm_actions=np.zeros((36, 1), dtype=np.float64),
        right_gripper_actions=np.zeros((36,), dtype=np.float64),
    )

    samples = _build_stage3_shifted_action_samples(
        save_dir=tmp_path,
        output_dir_name="vlm_object_search_visibility_memory_v3",
        episode_idx=0,
        metadata=_metadata(),
        context=context,
        annotations=annotations,
        max_context_frames=16,
        action_chunk_size=chunk_size,
    )

    by_frame = {sample["metadata"]["current_frame_idx"]: sample for sample in samples}
    assert len(samples) == 36
    assert by_frame[16]["metadata"]["prompt_frame_indices"] == [0, 16]
    assert by_frame[32]["metadata"]["prompt_frame_indices"] == [0, 16, 32]
    assert by_frame[33]["metadata"]["prompt_frame_indices"] == [1, 17, 33]


def test_stage3_v3_shifted_action_lattice_compresses_after_context_trigger(tmp_path: Path):
    chunk_size = 16
    annotations = [_annotation(frame_idx, 3) for frame_idx in range(257)]
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(257)]
    context = EpisodeContext(
        metadata=_metadata(),
        hdf5_path="",
        action_chunk_size=chunk_size,
        max_context_frames=16,
        frames=frames,
        left_arm_actions=np.zeros((257, 1), dtype=np.float64),
        left_gripper_actions=np.zeros((257,), dtype=np.float64),
        right_arm_actions=np.zeros((257, 1), dtype=np.float64),
        right_gripper_actions=np.zeros((257,), dtype=np.float64),
    )

    samples = _build_stage3_shifted_action_samples(
        save_dir=tmp_path,
        output_dir_name="vlm_object_search_visibility_memory_v3",
        episode_idx=0,
        metadata=_metadata(),
        context=context,
        annotations=annotations,
        max_context_frames=16,
        action_chunk_size=chunk_size,
    )

    by_frame = {sample["metadata"]["current_frame_idx"]: sample for sample in samples}
    assert by_frame[240]["metadata"]["prompt_frame_indices"] == list(range(0, 241, 16))
    assert by_frame[256]["metadata"]["prompt_frame_indices"] == [240, 256]


def test_stage3_v3_shifted_action_lattice_keeps_nonzero_pre_action_memory(tmp_path: Path):
    chunk_size = 16
    annotations = [_annotation(0, 2)] + [_annotation(frame_idx, 3) for frame_idx in range(1, 21)]
    annotations[0]["waist_heading_deg"] = 90.0
    annotations[0]["camera_target_theta"] = radians(120.0)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(21)]
    context = EpisodeContext(
        metadata=_metadata(),
        hdf5_path="",
        action_chunk_size=chunk_size,
        max_context_frames=16,
        frames=frames,
        left_arm_actions=np.zeros((21, 1), dtype=np.float64),
        left_gripper_actions=np.zeros((21,), dtype=np.float64),
        right_arm_actions=np.zeros((21, 1), dtype=np.float64),
        right_gripper_actions=np.zeros((21,), dtype=np.float64),
    )

    samples = _build_stage3_shifted_action_samples(
        save_dir=tmp_path,
        output_dir_name="vlm_object_search_visibility_memory_v3",
        episode_idx=0,
        metadata=_metadata(),
        context=context,
        annotations=annotations,
        max_context_frames=16,
        action_chunk_size=chunk_size,
    )

    by_frame = {sample["metadata"]["current_frame_idx"]: sample for sample in samples}
    assert by_frame[17]["metadata"]["prompt_frame_indices"] == [0, 1, 17]
    assert by_frame[18]["metadata"]["prompt_frame_indices"] == [0, 2, 18]


def test_stage3_v3_frame_field_uses_history_evidence_when_current_is_hidden(tmp_path: Path):
    chunk_size = 16
    annotations = [_annotation(0, 2), _hidden_target_annotation(1, 3)]
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    context = EpisodeContext(
        metadata=_metadata(),
        hdf5_path="",
        action_chunk_size=chunk_size,
        max_context_frames=16,
        frames=frames,
        left_arm_actions=np.zeros((2, 1), dtype=np.float64),
        left_gripper_actions=np.zeros((2,), dtype=np.float64),
        right_arm_actions=np.zeros((2, 1), dtype=np.float64),
        right_gripper_actions=np.zeros((2,), dtype=np.float64),
    )

    samples = _build_stage3_shifted_action_samples(
        save_dir=tmp_path,
        output_dir_name="vlm_object_search_visibility_memory_v3",
        episode_idx=0,
        metadata=_metadata(),
        context=context,
        annotations=annotations,
        max_context_frames=16,
        action_chunk_size=chunk_size,
    )

    assert len(samples) == 1
    assistant_content = samples[0]["messages"][1]["content"]
    assert samples[0]["metadata"]["prompt_frame_indices"] == [0, 1]
    assert samples[0]["metadata"]["evidence_from_history"] is True
    assert _extract_tag(assistant_content, "frame") == "[1]"
    assert "was found in frame 1" in _extract_tag(assistant_content, "think")
