import json
import re
import sys
from math import radians
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.export_rotate_object_search_visibility_memory_v4 import (  # noqa: E402
    _sample_uniform_action_frame_indices,
    _stage12_samples,
)
from script.rotate_vlm.models import EpisodeContext  # noqa: E402


def _metadata() -> dict:
    return {
        "task_name": "unit_test_rotate_view",
        "task_instruction": "Find and use the target object.",
        "subtask_instruction_map": {"0": "Find the target object."},
        "object_key_to_name": {"target": "001_target"},
    }


def _annotation(frame_idx: int, stage: int, heading_deg: float, visible: bool) -> dict:
    visible_keys = ["target"] if visible else []
    target_uv = [0.5, 0.5] if visible else [-1.0, -1.0]
    visible_map = {"target": [0.5, 0.5]} if visible else {}
    return {
        "frame_idx": int(frame_idx),
        "subtask": 0,
        "stage": int(stage),
        "waist_heading_deg": float(heading_deg),
        "camera_target_theta": radians(45.0),
        "search_target_keys": ["target"],
        "visible_object_keys": visible_keys,
        "visible_object_uv_map": visible_map,
        "target_uv_norm": target_uv,
    }


def _context(frame_count: int, chunk_size: int = 16) -> EpisodeContext:
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(frame_count)]
    return EpisodeContext(
        metadata=_metadata(),
        hdf5_path="",
        action_chunk_size=int(chunk_size),
        max_context_frames=16,
        frames=frames,
        left_arm_actions=np.arange(frame_count, dtype=np.float64).reshape(frame_count, 1),
        left_gripper_actions=np.arange(100, 100 + frame_count, dtype=np.float64),
        right_arm_actions=np.arange(200, 200 + frame_count, dtype=np.float64).reshape(frame_count, 1),
        right_gripper_actions=np.arange(300, 300 + frame_count, dtype=np.float64),
    )


def _extract_tag(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", content, flags=re.DOTALL)
    return "" if match is None else str(match.group(1))


def test_v4_uniform_action_sampling_matches_chunk_lattice():
    indices, actual_size, pad_count = _sample_uniform_action_frame_indices(list(range(32)), action_chunk_size=16)

    assert indices == list(range(0, 32, 2))
    assert actual_size == 16
    assert pad_count == 0


def test_v4_stage12_long_turn_writes_sampled_action_chunk(tmp_path: Path):
    annotations = [
        _annotation(frame_idx=idx, stage=1, heading_deg=45.0 * idx / 31.0, visible=False)
        for idx in range(32)
    ]
    context = _context(frame_count=32)

    samples, search_count, direct_count = _stage12_samples(
        output_dir=tmp_path,
        episode_idx=0,
        metadata=_metadata(),
        context=context,
        annotations=annotations,
        max_context_frames=16,
        action_chunk_size=16,
        fovy_deg=60.0,
    )

    assert samples
    assert search_count == len(samples)
    assert direct_count == 0
    first_sample = samples[0]
    assert first_sample["metadata"]["action_chunk_frame_indices"] == list(range(0, 32, 2))
    assert first_sample["metadata"]["action_chunk_actual_size"] == 16
    assert first_sample["metadata"]["action_chunk_pad_count"] == 0
    assert first_sample["action"][0] == [0.0, 100.0, 200.0, 300.0]
    assert first_sample["action"][1] == [2.0, 102.0, 202.0, 302.0]
    assert first_sample["action"][-1] == [30.0, 130.0, 230.0, 330.0]

    action_text = _extract_tag(first_sample["messages"][1]["content"], "action")
    assert json.loads(action_text) == first_sample["action"]


def test_v4_stage12_short_turn_pads_with_last_action(tmp_path: Path):
    annotations = [
        _annotation(frame_idx=idx, stage=2, heading_deg=45.0 * idx / 3.0, visible=True)
        for idx in range(4)
    ]
    context = _context(frame_count=4)

    samples, _, direct_count = _stage12_samples(
        output_dir=tmp_path,
        episode_idx=0,
        metadata=_metadata(),
        context=context,
        annotations=annotations,
        max_context_frames=16,
        action_chunk_size=16,
        fovy_deg=60.0,
    )

    assert samples
    assert direct_count == len(samples)
    sample = samples[0]
    assert sample["metadata"]["action_chunk_frame_indices"] == [0, 1, 2, 3]
    assert sample["metadata"]["action_chunk_actual_size"] == 4
    assert sample["metadata"]["action_chunk_pad_count"] == 12
    assert len(sample["action"]) == 16
    assert sample["action"][-1] == [3.0, 103.0, 203.0, 303.0]
