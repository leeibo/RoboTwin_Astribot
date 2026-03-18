import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


_CAMERA_ALIASES = {
    "camera_head": ["head_camera"],
    "head_camera": ["camera_head"],
}


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")

def _resolve_camera_key(observation_dict, camera_name):
    if not isinstance(observation_dict, Mapping):
        return None
    candidates = [camera_name] + _CAMERA_ALIASES.get(camera_name, [])
    for key in candidates:
        cam_data = observation_dict.get(key, None)
        if isinstance(cam_data, Mapping) and "rgb" in cam_data:
            return key
    return None


def _dump_videos(data_list, video_path=None, video_camera_names=None, video_path_map=None, main_video_camera="camera_head"):
    if video_path is None and not video_path_map:
        return
    observation_dict = data_list.get("observation", {})
    if not isinstance(observation_dict, Mapping):
        print("[pkl2hdf5] observation is missing, skip video export")
        return

    video_jobs = []

    # Main legacy video path for compatibility.
    if video_path is not None:
        main_key = _resolve_camera_key(observation_dict, main_video_camera)
        if main_key is None:
            main_key = _resolve_camera_key(observation_dict, "camera_head")
        if main_key is None:
            main_key = _resolve_camera_key(observation_dict, "head_camera")
        if main_key is None and video_camera_names:
            for name in video_camera_names:
                main_key = _resolve_camera_key(observation_dict, name)
                if main_key is not None:
                    break
        if main_key is not None:
            video_jobs.append((main_key, video_path))
        else:
            print(f"[pkl2hdf5] no RGB camera found for legacy video path: {video_path}")

    # Multi-view camera videos.
    if video_path_map:
        for camera_name, out_path in video_path_map.items():
            cam_key = _resolve_camera_key(observation_dict, camera_name)
            if cam_key is None:
                print(f"[pkl2hdf5] skip video for {camera_name}: camera stream not found")
                continue
            video_jobs.append((cam_key, out_path))
    elif video_path is not None and video_camera_names:
        base, ext = os.path.splitext(video_path)
        if ext == "":
            ext = ".mp4"
        for camera_name in video_camera_names:
            cam_key = _resolve_camera_key(observation_dict, camera_name)
            if cam_key is None:
                print(f"[pkl2hdf5] skip video for {camera_name}: camera stream not found")
                continue
            video_jobs.append((cam_key, f"{base}_{camera_name}{ext}"))

    # Deduplicate by output path.
    dedup = {}
    for cam_key, out_path in video_jobs:
        dedup[out_path] = cam_key

    for out_path, cam_key in dedup.items():
        imgs = np.array(observation_dict[cam_key]["rgb"])
        if imgs.ndim != 4 or imgs.shape[-1] not in (3, 4):
            print(f"[pkl2hdf5] invalid RGB tensor for {cam_key}, skip {out_path}")
            continue
        images_to_video(imgs, out_path=out_path)


def pkl_files_to_hdf5_and_video(
    pkl_files,
    hdf5_path,
    video_path=None,
    video_camera_names=None,
    video_path_map=None,
    main_video_camera="camera_head",
):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    _dump_videos(
        data_list=data_list,
        video_path=video_path,
        video_camera_names=video_camera_names,
        video_path_map=video_path_map,
        main_video_camera=main_video_camera,
    )

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(
    folder_path,
    hdf5_path,
    video_path=None,
    video_camera_names=None,
    video_path_map=None,
    main_video_camera="camera_head",
):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(
        pkl_files,
        hdf5_path,
        video_path=video_path,
        video_camera_names=video_camera_names,
        video_path_map=video_path_map,
        main_video_camera=main_video_camera,
    )
