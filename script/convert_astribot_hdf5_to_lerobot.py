import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


STATE_NAMES = (
    [f"left_arm_{i}" for i in range(7)]
    + ["left_gripper"]
    + [f"right_arm_{i}" for i in range(7)]
    + ["right_gripper", "torso_yaw", "head_2"]
)


def natural_episode_key(path):
    match = re.search(r"episode(\d+)", Path(path).stem)
    episode_idx = int(match.group(1)) if match else -1
    return (str(Path(path).parent), episode_idx, str(path))


def collect_hdf5(paths, max_episodes=None):
    files = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and path.suffix in {".hdf5", ".h5"}:
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.hdf5"))
            files.extend(path.rglob("*.h5"))
        else:
            raise FileNotFoundError(str(path))
    files = sorted({path.resolve() for path in files}, key=natural_episode_key)
    if max_episodes is not None and max_episodes >= 0:
        files = files[:max_episodes]
    return files


def find_subtask_metadata_path(ep_path):
    ep_path = Path(ep_path)
    metadata_name = f"{ep_path.stem}.json"
    for parent in ep_path.parents:
        candidate = parent / "subtask_metadata" / metadata_name
        if candidate.exists():
            return candidate
    return None


def load_subtask_metadata(ep_path):
    metadata_path = find_subtask_metadata_path(ep_path)
    if metadata_path is None:
        return {}, None
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f), metadata_path


def read_annotation_array(f, metadata, key, length, default=0):
    if key in f:
        values = np.asarray(f[key][:])
    else:
        annotations = metadata.get("frame_annotations", [])
        values = np.full(length, default, dtype=np.int64)
        for i, ann in enumerate(annotations[:length]):
            values[i] = int(ann.get(key, default))
    if len(values) < length:
        padded = np.full(length, default, dtype=values.dtype)
        padded[: len(values)] = values
        values = padded
    return values[:length]


def build_state18(f):
    left_arm = np.asarray(f["joint_action/left_arm"][:], dtype=np.float32)
    left_gripper = np.asarray(f["joint_action/left_gripper"][:], dtype=np.float32).reshape(-1, 1)
    right_arm = np.asarray(f["joint_action/right_arm"][:], dtype=np.float32)
    right_gripper = np.asarray(f["joint_action/right_gripper"][:], dtype=np.float32).reshape(-1, 1)
    torso_yaw = np.asarray(f["joint_action/torso"][:], dtype=np.float32).reshape(-1, 1)
    head = np.asarray(f["joint_action/head"][:], dtype=np.float32)
    head_2 = head[:, 1:2]
    state = np.concatenate(
        [left_arm, left_gripper, right_arm, right_gripper, torso_yaw, head_2],
        axis=1,
    )
    if state.shape[1] != 18:
        raise ValueError(f"expected 18-dim Astribot state, got shape {state.shape}")
    return state


def decode_image(rgb_bytes, image_size, hdf5_image_color_order):
    import cv2

    if isinstance(rgb_bytes, np.void):
        rgb_bytes = bytes(rgb_bytes)
    else:
        rgb_bytes = bytes(rgb_bytes)
    rgb_bytes = rgb_bytes.rstrip(b"\0")
    arr = np.frombuffer(rgb_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("failed to decode camera image bytes")
    if hdf5_image_color_order == "legacy_rgb":
        image_rgb = decoded
    elif hdf5_image_color_order == "standard_bgr":
        image_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported hdf5 image color order: {hdf5_image_color_order}")
    if image_size is not None:
        width, height = image_size
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return image_rgb


def decode_images(rgb_dataset, frame_count, image_size, decode_workers, hdf5_image_color_order):
    decode_workers = max(int(decode_workers), 1)
    if decode_workers == 1:
        return [decode_image(rgb_dataset[i], image_size, hdf5_image_color_order) for i in range(frame_count)]
    rgb_bytes = [rgb_dataset[i] for i in range(frame_count)]
    with ThreadPoolExecutor(max_workers=decode_workers) as executor:
        return list(executor.map(lambda data: decode_image(data, image_size, hdf5_image_color_order), rgb_bytes))


def load_episode(ep_path, args):
    metadata, metadata_path = load_subtask_metadata(ep_path)
    with h5py.File(ep_path, "r") as f:
        required = [
            "joint_action/left_arm",
            "joint_action/left_gripper",
            "joint_action/right_arm",
            "joint_action/right_gripper",
            "joint_action/torso",
            "joint_action/head",
            f"observation/{args.camera}/rgb",
            "subtask",
        ]
        missing = [key for key in required if key not in f]
        if missing:
            raise KeyError(f"{ep_path}: missing required datasets: {missing}")

        state18 = build_state18(f)
        t = state18.shape[0]
        if t < 2:
            raise ValueError(f"{ep_path}: episode has fewer than 2 frames")
        frame_count = t - 1

        if args.require_keyframes:
            keyframe_missing = [key for key in ("subtask_keyframe", "motion_keyframe") if key not in f]
            if keyframe_missing:
                raise KeyError(f"{ep_path}: missing keyframe datasets: {keyframe_missing}")

        subtask = read_annotation_array(f, metadata, "subtask", t, default=0)
        subtask_instruction_idx = read_annotation_array(f, metadata, "subtask_instruction_idx", t, default=0)
        stage = read_annotation_array(f, metadata, "stage", t, default=0)

        if "subtask_keyframe" in f:
            subtask_keyframe = np.asarray(f["subtask_keyframe"][:], dtype=np.int64)
        else:
            subtask_keyframe = np.zeros(t, dtype=np.int64)
            subtask_keyframe[0] = 1
            subtask_keyframe[1:] = (subtask[1:] != subtask[:-1]).astype(np.int64)

        if "motion_keyframe" in f:
            motion_keyframe = np.asarray(f["motion_keyframe"][:], dtype=np.int64)
        else:
            motion_keyframe = np.zeros(t, dtype=np.int64)
            motion_keyframe[0] = 1

        images = decode_images(
            f[f"observation/{args.camera}/rgb"],
            frame_count,
            tuple(args.image_size) if args.image_size else None,
            args.decode_workers,
            args.hdf5_image_color_order,
        )

    task_instruction = metadata.get("task_instruction") or metadata.get("task_name") or infer_task_name(ep_path)
    episode_info = {
        "episode_path": str(ep_path),
        "metadata_path": str(metadata_path) if metadata_path else "",
        "task_name": metadata.get("task_name", infer_task_name(ep_path)),
        "task_instruction": task_instruction,
        "subtask_instruction_map": metadata.get("subtask_instruction_map", {}),
        "subtask_defs": metadata.get("subtask_defs", []),
    }
    return {
        "state": state18[:frame_count],
        "action": state18[1 : frame_count + 1],
        "images": images,
        "subtask": subtask[:frame_count].astype(np.int64),
        "subtask_instruction_idx": subtask_instruction_idx[:frame_count].astype(np.int64),
        "stage": stage[:frame_count].astype(np.int64),
        "subtask_keyframe": subtask_keyframe[:frame_count].astype(np.int64),
        "motion_keyframe": motion_keyframe[:frame_count].astype(np.int64),
        "task": task_instruction,
        "episode_info": episode_info,
    }


def infer_task_name(ep_path):
    parts = Path(ep_path).parts
    if "data" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("data")
        if idx >= 2:
            return parts[idx - 2]
    return Path(ep_path).parents[2].name if len(Path(ep_path).parents) >= 3 else "astribot_task"


def import_lerobot():
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset


def build_features(args):
    width, height = args.image_size
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (18,),
            "names": [STATE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (18,),
            "names": [STATE_NAMES],
        },
        f"observation.images.{args.camera}": {
            "dtype": args.mode,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "subtask_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["subtask_index"],
        },
        "subtask_instruction_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["subtask_instruction_index"],
        },
        "stage": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["stage"],
        },
        "subtask_keyframe": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["subtask_keyframe"],
        },
        "motion_keyframe": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["motion_keyframe"],
        },
    }
    return features


def dataset_root(output_root, repo_id):
    return Path(output_root).expanduser() / repo_id


def create_lerobot_dataset(args):
    os.environ["HF_LEROBOT_HOME"] = str(Path(args.output_root).expanduser().resolve())
    os.environ.pop("LEROBOT_HOME", None)
    LeRobotDataset = import_lerobot()
    root = dataset_root(args.output_root, args.repo_id)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    common_kwargs = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        "robot_type": args.robot_type,
        "features": build_features(args),
    }
    try:
        return LeRobotDataset.create(
            **common_kwargs,
            use_videos=(args.mode == "video"),
            tolerance_s=args.tolerance_s,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            video_backend=args.video_backend,
        )
    except TypeError:
        return LeRobotDataset.create(
            **common_kwargs,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )


def add_frame_compat(dataset, frame, task, mode_holder):
    mode = mode_holder.get("mode")
    if mode == "task_kwarg":
        dataset.add_frame(frame, task=task)
        return
    if mode == "task_in_frame":
        frame = dict(frame)
        frame["task"] = task
        dataset.add_frame(frame)
        return
    if mode == "save_episode_task":
        dataset.add_frame(frame)
        return

    try:
        dataset.add_frame(frame, task=task)
        mode_holder["mode"] = "task_kwarg"
        return
    except TypeError:
        pass
    try:
        frame_with_task = dict(frame)
        frame_with_task["task"] = task
        dataset.add_frame(frame_with_task)
        mode_holder["mode"] = "task_in_frame"
        return
    except TypeError:
        pass
    dataset.add_frame(frame)
    mode_holder["mode"] = "save_episode_task"


def save_episode_compat(dataset, task, mode_holder):
    if mode_holder.get("mode") == "save_episode_task":
        dataset.save_episode(task=task)
        return
    try:
        dataset.save_episode()
    except TypeError:
        dataset.save_episode(task=task)


def finalize_dataset(dataset, args):
    if args.no_consolidate:
        return
    if hasattr(dataset, "consolidate"):
        try:
            dataset.consolidate(run_compute_stats=not args.skip_compute_stats)
        except TypeError:
            dataset.consolidate()
    elif hasattr(dataset, "finalize"):
        dataset.finalize()


def write_episode_metadata(output_root, repo_id, episode_infos):
    root = dataset_root(output_root, repo_id)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "astribot_subtask_metadata.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"episodes": episode_infos}, f, ensure_ascii=False, indent=2)
    return out_path


def populate_dataset(dataset, hdf5_files, args):
    task_mode = {}
    episode_infos = []
    for ep_idx, ep_path in enumerate(tqdm(hdf5_files, desc="episodes")):
        episode = load_episode(ep_path, args)
        frame_count = episode["state"].shape[0]
        for i in range(frame_count):
            frame = {
                "observation.state": episode["state"][i],
                "action": episode["action"][i],
                f"observation.images.{args.camera}": episode["images"][i],
                "subtask_index": np.array([episode["subtask"][i]], dtype=np.int64),
                "subtask_instruction_index": np.array([episode["subtask_instruction_idx"][i]], dtype=np.int64),
                "stage": np.array([episode["stage"][i]], dtype=np.int64),
                "subtask_keyframe": np.array([episode["subtask_keyframe"][i]], dtype=np.int64),
                "motion_keyframe": np.array([episode["motion_keyframe"][i]], dtype=np.int64),
            }
            add_frame_compat(dataset, frame, episode["task"], task_mode)
        save_episode_compat(dataset, episode["task"], task_mode)
        info = dict(episode["episode_info"])
        info["lerobot_episode_index"] = ep_idx
        info["num_frames"] = int(frame_count)
        episode_infos.append(info)
    return episode_infos


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Astribot RoboTwin HDF5 episodes to LeRobot format.")
    parser.add_argument("input_paths", nargs="+", help="HDF5 files or directories containing HDF5 episodes.")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id, e.g. local/blocks_ranking_rgb_fan_double.")
    parser.add_argument("--output-root", default="lerobot_data", help="Directory used as HF_LEROBOT_HOME/LEROBOT_HOME.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--robot-type", default="astribot")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--camera", default="camera_head")
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 480), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--hdf5-image-color-order",
        choices=["legacy_rgb", "standard_bgr"],
        default="legacy_rgb",
        help=(
            "legacy_rgb matches existing Astribot HDF5 files whose JPEG bytes were encoded from RGB arrays "
            "with cv2.imencode. Use standard_bgr for HDF5 images encoded with OpenCV's normal BGR convention."
        ),
    )
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--image-writer-processes", type=int, default=10)
    parser.add_argument("--image-writer-threads", type=int, default=5)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--tolerance-s", type=float, default=0.0001)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--allow-missing-keyframes", action="store_true")
    parser.add_argument("--no-consolidate", action="store_true")
    parser.add_argument("--skip-compute-stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.require_keyframes = not args.allow_missing_keyframes
    hdf5_files = collect_hdf5(args.input_paths, args.max_episodes)
    if not hdf5_files:
        raise RuntimeError("no HDF5 episodes found")

    print(f"Found {len(hdf5_files)} episode(s)")
    print(f"Output: {dataset_root(args.output_root, args.repo_id)}")
    dataset = create_lerobot_dataset(args)
    episode_infos = populate_dataset(dataset, hdf5_files, args)
    metadata_path = write_episode_metadata(args.output_root, args.repo_id, episode_infos)
    finalize_dataset(dataset, args)
    print(f"Wrote Astribot metadata: {metadata_path}")
    print("Done")


if __name__ == "__main__":
    main()
