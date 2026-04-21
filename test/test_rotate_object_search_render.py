import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.rotate_vlm import (  # noqa: E402
    _build_angle_delta_sample,
    _build_object_search_sample,
    _collect_angle_delta_pairs,
    _render_angle_delta_response,
    _render_memory_compression_response,
    _render_object_search_response,
)
from script.rotate_vlm.models import EpisodeContext, EpisodeSnapshot, MemorySlot  # noqa: E402
from script.rotate_vlm.snapshots import _build_memory_slots, build_episode_context  # noqa: E402


def _make_slot(
    *,
    slot_idx: int,
    frame_idx: int,
    stage: int,
    roles: list[str],
    planned_delta_deg: float,
    current_heading_deg: float = 0.0,
    search_target_keys: list[str] | None = None,
) -> MemorySlot:
    annotation = {
        "frame_idx": int(frame_idx),
        "subtask": 0,
        "stage": int(stage),
        "waist_heading_deg": float(current_heading_deg),
        "camera_target_theta": None,
        "search_target_keys": list(search_target_keys or ["target"]),
    }
    return MemorySlot(
        slot_idx=int(slot_idx),
        frame_idx=int(frame_idx),
        subtask_id=0,
        stage=int(stage),
        current_annotation=annotation,
        current_heading_deg=float(current_heading_deg),
        planned_delta_deg=float(planned_delta_deg),
        planned_heading_deg=float(current_heading_deg + planned_delta_deg),
        roles=list(roles),
    )


def _metadata() -> dict:
    return {
        "task_name": "unit_test_rotate_view",
        "task_instruction": "Find the target object.",
        "subtask_instruction_map": {"0": "Find the target object."},
        "object_key_to_name": {"target": "001_target"},
    }


def _extract_tag(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", content, flags=re.DOTALL)
    return "" if match is None else str(match.group(1))


def test_object_search_history_evidence_mentions_evidence_frame():
    snapshot = EpisodeSnapshot(
        current_slot=_make_slot(
            slot_idx=1,
            frame_idx=10,
            stage=2,
            roles=["stage2_end"],
            planned_delta_deg=25.0,
            current_heading_deg=0.0,
        ),
        prompt_slots=[
            _make_slot(
                slot_idx=0,
                frame_idx=4,
                stage=1,
                roles=["stage1_end"],
                planned_delta_deg=15.0,
                current_heading_deg=25.0,
            )
        ],
        prompt_frame_indices=[4, 10],
        prompt_planned_actions=[(-25, 0)],
        evidence_frame_idx=4,
        evidence_prompt_index=1,
        evidence_uv_norm=[0.2, 0.7],
        evidence_from_history=True,
        memory_support_ready=True,
    )

    content = _render_object_search_response(_metadata(), snapshot)
    think = _extract_tag(content, "think")

    assert think.startswith('Frames: 2 total (1 history + current). Past actions: [(25, 0)]. The current task is "Find the target object.".')
    assert "The target object is the target object." in think
    assert "The target object was found in frame 1 at (200,700)." in think
    assert think.endswith("Next: Rotate(-25, 0).")
    assert _extract_tag(content, "info") == "1"
    assert _extract_tag(content, "frame") == "[1]"
    assert _extract_tag(content, "camera") == "Rotate(-25, 0)"


def test_stage3_object_search_uses_real_action_chunk(tmp_path: Path):
    current_slot = _make_slot(slot_idx=0, frame_idx=0, stage=3, roles=["stage3_chunk"], planned_delta_deg=0.0)
    current_slot.action_chunk_frame_indices = [0, 1]
    current_slot.action_chunk_actual_size = 2
    current_slot.action_chunk_pad_count = 1
    snapshot = EpisodeSnapshot(
        current_slot=current_slot,
        prompt_slots=[],
        prompt_frame_indices=[0],
        prompt_planned_actions=[],
        evidence_frame_idx=0,
        evidence_prompt_index=1,
        evidence_uv_norm=[0.4, 0.5],
        evidence_from_history=False,
        memory_support_ready=True,
    )
    context = EpisodeContext(
        metadata=_metadata(),
        hdf5_path="",
        action_chunk_size=3,
        max_context_frames=16,
        frames=[np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((16, 16, 3), dtype=np.uint8)],
        left_arm_actions=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        left_gripper_actions=np.array([0.1, 0.2], dtype=np.float64),
        right_arm_actions=np.array([[5.0], [6.0]], dtype=np.float64),
        right_gripper_actions=np.array([0.3, 0.4], dtype=np.float64),
    )

    sample = _build_object_search_sample(tmp_path, 0, _metadata(), context, snapshot)
    assistant_content = str(sample["messages"][-1]["content"])

    expected_chunk = [
        [1.0, 2.0, 0.1, 5.0, 0.3],
        [3.0, 4.0, 0.2, 6.0, 0.4],
        [3.0, 4.0, 0.2, 6.0, 0.4],
    ]
    think = _extract_tag(assistant_content, "think")
    assert sample["action"] == expected_chunk
    assert sample["metadata"]["prompt_image_count"] == 1
    assert sample["metadata"]["camera_delta_deg"] == 0
    assert think.startswith('Frames: current only. Past actions: none. The current task is "Find the target object.". The target object is the target object.')
    assert "The robot is now executing the task." in think
    assert think.endswith("Next: Rotate(0, 0).")
    assert _extract_tag(assistant_content, "frame") == "[1]"
    assert _extract_tag(assistant_content, "camera") == "Rotate(0, 0)"
    assert _extract_tag(assistant_content, "action") == "[[1.0,2.0,0.1,5.0,0.3],[3.0,4.0,0.2,6.0,0.4],[3.0,4.0,0.2,6.0,0.4]]"


def test_stage3_memory_slot_uses_first_frame_in_chunk():
    annotations = [
        {
            "frame_idx": 10,
            "subtask": 0,
            "stage": 3,
            "waist_heading_deg": 1.0,
            "camera_target_theta": None,
            "search_target_keys": ["target"],
        },
        {
            "frame_idx": 11,
            "subtask": 0,
            "stage": 3,
            "waist_heading_deg": 2.0,
            "camera_target_theta": None,
            "search_target_keys": ["target"],
        },
        {
            "frame_idx": 12,
            "subtask": 0,
            "stage": 3,
            "waist_heading_deg": 3.0,
            "camera_target_theta": None,
            "search_target_keys": ["target"],
        },
    ]

    slots = _build_memory_slots(annotations=annotations, action_chunk_size=10)

    assert len(slots) == 1
    assert slots[0].frame_idx == 10
    assert slots[0].current_annotation["frame_idx"] == 10
    assert slots[0].action_chunk_frame_indices == [10, 11, 12]


def test_stage3_snapshots_keep_previous_action_chunks_in_prompt(monkeypatch):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(21)]
    left = np.zeros((21, 1), dtype=np.float64)
    left_gripper = np.zeros((21,), dtype=np.float64)
    right = np.zeros((21, 1), dtype=np.float64)
    right_gripper = np.zeros((21,), dtype=np.float64)

    def _fake_load_hdf5_episode_data(_hdf5_path: str):
        return frames, left, left_gripper, right, right_gripper

    monkeypatch.setattr("script.rotate_vlm.snapshots.load_hdf5_episode_data", _fake_load_hdf5_episode_data)

    annotations = [
        {
            "frame_idx": 0,
            "subtask": 0,
            "stage": 1,
            "waist_heading_deg": 0.0,
            "camera_target_theta": 0.0,
            "search_target_keys": ["target"],
        }
    ]
    for frame_idx in range(1, 21):
        annotations.append(
            {
                "frame_idx": frame_idx,
                "subtask": 0,
                "stage": 3,
                "waist_heading_deg": 0.0,
                "camera_target_theta": None,
                "search_target_keys": ["target"],
            }
        )

    context = build_episode_context(
        metadata={"frame_annotations": annotations},
        hdf5_path="unused.hdf5",
        action_chunk_size=10,
        max_context_frames=16,
    )

    stage3_snapshots = [snapshot for snapshot in context.snapshots if "stage3_chunk" in snapshot.roles]
    assert len(stage3_snapshots) == 2
    assert stage3_snapshots[0].prompt_frame_indices == [0, 1]
    assert stage3_snapshots[1].prompt_frame_indices == [0, 1, 11]


def test_angle_delta_uses_rotation_difference_not_cumulative_planning():
    previous_slot = _make_slot(slot_idx=0, frame_idx=0, stage=1, roles=["stage1_start"], planned_delta_deg=30.0, current_heading_deg=10.0)
    current_slot = _make_slot(slot_idx=1, frame_idx=5, stage=2, roles=["stage2_end"], planned_delta_deg=5.0, current_heading_deg=40.0)

    pairs = _collect_angle_delta_pairs([previous_slot, current_slot])

    assert len(pairs) == 1
    assert pairs[0][2] == (30, 0)
    assert _render_angle_delta_response(pairs[0][2]) == (
        "<think>Frames: 2 total (1 history + current). From frame 1 to frame 2, the rotation difference is (-30, 0).</think>"
        "<camera>Rotate(-30, 0)</camera>"
    )

    sample = _build_angle_delta_sample(Path("."), 0, _metadata(), EpisodeContext(
        metadata=_metadata(),
        hdf5_path="",
        action_chunk_size=3,
        max_context_frames=16,
        frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(6)],
        left_arm_actions=np.zeros((0, 0), dtype=np.float64),
        left_gripper_actions=np.zeros((0,), dtype=np.float64),
        right_arm_actions=np.zeros((0, 0), dtype=np.float64),
        right_gripper_actions=np.zeros((0,), dtype=np.float64),
    ), previous_slot, current_slot, pairs[0][2])
    assert sample["metadata"]["angle_delta_deg"] == -30
    assert sample["metadata"]["camera_delta_pair"] == ["-30", "0"]


def test_memory_compression_response_uses_view_deltas_and_frame_list():
    metadata = {
        "task_name": "unit_test_rotate_view",
        "object_key_to_name": {"target": "001_target"},
    }
    sample_slots = [
        _make_slot(slot_idx=0, frame_idx=0, stage=1, roles=["stage1_start"], planned_delta_deg=0.0, current_heading_deg=0.0),
        _make_slot(slot_idx=1, frame_idx=1, stage=1, roles=["stage1_end"], planned_delta_deg=0.0, current_heading_deg=30.0),
        _make_slot(slot_idx=2, frame_idx=2, stage=2, roles=["stage2_end"], planned_delta_deg=0.0, current_heading_deg=60.0),
    ]
    kept_slots = [sample_slots[0], sample_slots[2]]

    content = _render_memory_compression_response(metadata, sample_slots, kept_slots)
    think = _extract_tag(content, "think")

    assert think.startswith("Frames: 3 total (2 history + current). Past actions: [(-30, 0), (-30, 0)].")
    assert "Spatially, keep frames [1, 3] for distinct coverage." in think
    assert "Keep frames [1, 3]." in think
    assert _extract_tag(content, "info") == "1"
    assert _extract_tag(content, "frame") == "[1, 3]"
