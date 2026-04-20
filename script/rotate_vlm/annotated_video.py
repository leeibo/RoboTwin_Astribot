from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _annotation_text_lines(annotation: dict[str, Any]) -> list[str]:
    if any(key in annotation for key in ["field_frame_text", "think_text", "field_camera_text", "field_action_text"]):
        return [
            f"frame={annotation.get('field_frame_text', '-')}"
            f" info={annotation.get('field_info_text', '-')}"
            f" camera={annotation.get('field_camera_text', '-')}",
            f"think={annotation.get('think_text', '-')}",
            f"action={annotation.get('field_action_text', '-')}",
        ]
    focus_key = annotation.get("focus_object_key", None)
    focus_text = "-" if focus_key is None else str(focus_key)
    search_keys = ",".join(str(key) for key in annotation.get("search_target_keys", []) or []) or "-"
    action_keys = ",".join(str(key) for key in annotation.get("action_target_keys", []) or []) or "-"
    carried = ",".join(str(key) for key in annotation.get("carried_object_keys", []) or []) or "-"
    visible = ",".join(str(key) for key in annotation.get("visible_object_keys", []) or []) or "-"
    discovered = ",".join(str(key) for key in annotation.get("discovered_object_keys", []) or []) or "-"
    return [
        (
            f"frame={int(annotation.get('frame_idx', 0))} "
            f"subtask={int(annotation.get('subtask', 0))} stage={int(annotation.get('stage', 0))}"
        ),
        f"focus={focus_text} search={search_keys} action={action_keys}",
        f"visible={visible} discovered={discovered} carried={carried}",
    ]


def overlay_annotation(frame_rgb: np.ndarray, annotation: dict[str, Any]) -> np.ndarray:
    frame_bgr = frame_rgb.copy()
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], 72), (0, 0, 0), thickness=-1)
    frame_bgr = cv2.addWeighted(overlay, 0.7, frame_bgr, 0.3, 0.0)
    for idx, line in enumerate(_annotation_text_lines(annotation)):
        cv2.putText(
            frame_bgr,
            line,
            (12, 22 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return frame_bgr


def export_annotated_video(
    frames: list[np.ndarray],
    frame_annotations: list[dict[str, Any]],
    output_path: str,
) -> str:
    if not frames:
        raise ValueError("annotated video export requires decoded frames")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (int(frames[0].shape[1]), int(frames[0].shape[0])),
    )
    try:
        for frame_idx, frame in enumerate(frames):
            annotation = frame_annotations[min(frame_idx, len(frame_annotations) - 1)] if frame_annotations else {}
            annotated = overlay_annotation(frame, annotation)
            writer.write(annotated)
    finally:
        writer.release()

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        temp_output.replace(output)
        return str(output)

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(temp_output),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        temp_output.replace(output)
        return str(output)

    if temp_output.exists():
        temp_output.unlink()
    return str(output)


def default_annotated_video_path(save_dir: str, episode_idx: int) -> str:
    return os.path.join(save_dir, "video", f"episode{int(episode_idx)}_annotated.mp4")


def _extract_tag_text(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", content, flags=re.DOTALL)
    return "" if match is None else str(match.group(1)).strip()


def _build_episode_object_search_entries(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for sample in samples:
        metadata = sample.get("metadata", {}) or {}
        episode_idx = int(metadata.get("episode_idx", 0) or 0)
        subtask_id = int(metadata.get("subtask_id", 0) or 0)
        current_frame_idx = int(metadata.get("current_frame_idx", 0) or 0)
        content = str((sample.get("messages", [{}, {}])[-1] or {}).get("content", ""))
        grouped.setdefault(
            (episode_idx, subtask_id),
            {"episode_idx": episode_idx, "subtask_id": subtask_id, "entries_by_frame": {}},
        )
        grouped[(episode_idx, subtask_id)]["entries_by_frame"][current_frame_idx] = {
            "frame_idx": current_frame_idx,
            "think_text": _extract_tag_text(content, "think").replace("<info>", "").replace("</info>", ""),
            "field_info_text": _extract_tag_text(content, "info"),
            "field_frame_text": _extract_tag_text(content, "frame"),
            "field_camera_text": _extract_tag_text(content, "camera"),
            "field_action_text": _extract_tag_text(content, "action"),
        }
        grouped[(episode_idx, subtask_id)]["entries_by_frame"][current_frame_idx]["think_text"] = re.sub(
            r"<[^>]+>",
            "",
            grouped[(episode_idx, subtask_id)]["entries_by_frame"][current_frame_idx]["think_text"],
        ).strip()
    return [grouped[key] for key in sorted(grouped)]


def _with_object_search_annotations(
    frame_annotations: list[dict[str, Any]],
    episode_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    entries_by_frame = dict((episode_entry.get("entries_by_frame", {}) or {}))
    merged: list[dict[str, Any]] = []
    latest_fields: dict[str, str] = {}
    for annotation in frame_annotations:
        frame_idx = int(annotation.get("frame_idx", 0) or 0)
        if frame_idx in entries_by_frame:
            latest_fields = dict(entries_by_frame[frame_idx])
        merged.append({**dict(annotation), **latest_fields})
    return merged
