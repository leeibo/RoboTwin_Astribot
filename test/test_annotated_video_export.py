import importlib.util
import pickle
import sys
import types
from pathlib import Path

import cv2
import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "envs" / "utils"
MODULE_PATH = UTILS_DIR / "pkl2hdf5.py"


def _load_pkl2hdf5_module():
    envs_pkg = sys.modules.setdefault("envs", types.ModuleType("envs"))
    envs_pkg.__path__ = [str(REPO_ROOT / "envs")]

    utils_pkg = sys.modules.setdefault("envs.utils", types.ModuleType("envs.utils"))
    utils_pkg.__path__ = [str(UTILS_DIR)]

    module_name = "envs.utils.pkl2hdf5"
    module_spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


pkl2hdf5 = _load_pkl2hdf5_module()


def _write_episode_pkl(folder_path, frame_idx, rgb_value):
    rgb = np.full((72, 96, 3), rgb_value, dtype=np.uint8)
    payload = {
        "observation": {
            "camera_head": {
                "rgb": rgb,
            }
        },
        "subtask": np.int32(1),
        "stage": np.int8(2),
        "subtask_instruction_idx": np.int32(1),
    }
    with open(folder_path / f"{frame_idx}.pkl", "wb") as f:
        pickle.dump(payload, f)


def test_process_folder_exports_annotated_video(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_episode_pkl(cache_dir, 0, 255)
    _write_episode_pkl(cache_dir, 1, 245)

    hdf5_path = tmp_path / "episode0.hdf5"
    raw_video_path = tmp_path / "episode0.mp4"
    annotated_video_path = tmp_path / "episode0_annotated.mp4"

    annotation_metadata = {
        "frame_annotations": [
            {
                "frame_idx": 0,
                "subtask": 1,
                "stage": 1,
                "subtask_instruction_idx": 1,
                "waist_heading_deg": 0.0,
                "discovered_object_keys": ["A"],
            },
            {
                "frame_idx": 1,
                "subtask": 1,
                "stage": 3,
                "subtask_instruction_idx": 1,
                "waist_heading_deg": 18.0,
                "discovered_object_keys": ["A", "B"],
            },
        ],
        "subtask_instruction_map": {
            "1": "pick up the red block and align it with the tray",
        },
        "object_key_to_name": {
            "A": "red block",
            "B": "tray",
        },
    }

    pkl2hdf5.process_folder_to_hdf5_video(
        cache_dir,
        hdf5_path,
        video_path=raw_video_path,
        video_camera_names=["camera_head"],
        video_path_map={"camera_head": str(tmp_path / "episode0_camera_head.mp4")},
        main_video_camera="camera_head",
        annotated_video_path=annotated_video_path,
        annotated_video_camera="camera_head",
        annotated_video_metadata=annotation_metadata,
    )

    assert hdf5_path.exists()
    assert raw_video_path.exists()
    assert annotated_video_path.exists()

    with h5py.File(hdf5_path, "r") as hdf5_file:
        assert "subtask" in hdf5_file
        assert hdf5_file["subtask"].shape[0] == 2

    cap = cv2.VideoCapture(str(annotated_video_path))
    ok, frame = cap.read()
    cap.release()
    assert ok is True
    assert frame is not None
    assert np.any(frame < 240)
