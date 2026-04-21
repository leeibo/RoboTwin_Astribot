import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.rotate_vlm.models import MemorySlot
from script.rotate_vlm.models import CompressionEvent
from script.rotate_vlm import _compression_subset_variants
from script.rotate_vlm.snapshots import _compress_memory_slots


def _make_slot(slot_idx: int, frame_idx: int, heading_deg: float, planned_delta_deg: float) -> MemorySlot:
    return MemorySlot(
        slot_idx=int(slot_idx),
        frame_idx=int(frame_idx),
        subtask_id=1,
        stage=1,
        current_annotation={
            "frame_idx": int(frame_idx),
            "subtask": 1,
            "stage": 1,
            "waist_heading_deg": float(heading_deg),
            "camera_target_theta": None,
        },
        current_heading_deg=float(heading_deg),
        planned_delta_deg=float(planned_delta_deg),
        planned_heading_deg=float(heading_deg + planned_delta_deg),
        roles=["stage1_start"],
    )


def test_zero_rotate_blocks_keep_only_latest_frame():
    slots = [
        _make_slot(0, 0, 0.0, 0.0),
        _make_slot(1, 1, 0.0, 0.0),
        _make_slot(2, 2, 45.0, 45.0),
        _make_slot(3, 3, 45.0, 0.0),
        _make_slot(4, 4, 45.0, 0.0),
    ]
    kept = _compress_memory_slots(slots)
    assert [slot.frame_idx for slot in kept] == [1, 4]


def test_newer_frame_can_replace_middle_old_frame_when_union_coverage_is_unchanged():
    slots = [
        _make_slot(0, 0, 0.0, 10.0),
        _make_slot(1, 1, 20.0, 10.0),
        _make_slot(2, 2, 40.0, 10.0),
    ]
    kept = _compress_memory_slots(slots, half_fov_deg=35.0)
    assert [slot.frame_idx for slot in kept] == [0, 2]


def test_current_newest_frame_is_preserved_during_incremental_compression():
    slots = [
        _make_slot(0, 0, 0.0, 10.0),
        _make_slot(1, 1, 20.0, 10.0),
        _make_slot(2, 2, 40.0, 10.0),
        _make_slot(3, 3, 60.0, 10.0),
    ]
    kept = _compress_memory_slots(slots, half_fov_deg=35.0)
    assert kept[-1].frame_idx == 3
    assert [slot.frame_idx for slot in kept] == [0, 3]


def test_subtask_switch_small_compression_event_exports_single_direct_variant():
    before_slots = [
        _make_slot(0, 0, 0.0, 10.0),
        _make_slot(1, 1, 20.0, 10.0),
        _make_slot(2, 2, 40.0, 10.0),
    ]
    event = CompressionEvent(
        trigger="subtask_switch",
        trigger_frame_idx=43,
        before_slots=before_slots,
        after_slots=_compress_memory_slots(before_slots, half_fov_deg=35.0),
    )

    variants = _compression_subset_variants(event)

    assert len(variants) == 1
    variant_name, sample_slots, kept_slots = variants[0]
    assert variant_name == "subtask_switch_direct"
    assert [slot.frame_idx for slot in sample_slots] == [0, 1, 2]
    assert [slot.frame_idx for slot in kept_slots] == [0, 2]
