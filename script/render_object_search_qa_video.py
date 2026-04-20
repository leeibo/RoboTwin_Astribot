from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
)


@dataclass
class QaEntry:
    episode_idx: int
    sample_frame_idx: int
    image_count: int
    question: str
    think: str
    info: str
    frame: str
    camera: str
    action: str
    metadata: dict[str, Any]


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _extract_tag_text(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", str(content), flags=re.DOTALL)
    return "" if match is None else str(match.group(1)).strip()


def _compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _user_prompt_text(content: str) -> str:
    text = re.sub(r"<image>", "", str(content))
    return _compact_spaces(text)


@lru_cache(maxsize=16)
def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    size = max(12, int(size))
    candidates = []
    if bold:
        candidates.append("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    candidates.extend(_FONT_CANDIDATES)
    for font_path in candidates:
        path = Path(font_path)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), str(text), font=font)
    return int(right - left), int(bottom - top)


def _split_long_token(
    draw: ImageDraw.ImageDraw,
    token: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if token == "":
        return [""]
    chunks: list[str] = []
    current = ""
    for ch in token:
        trial = current + ch
        if current and _text_size(draw, trial, font)[0] > max_width:
            chunks.append(current)
            current = ch
        else:
            current = trial
    if current:
        chunks.append(current)
    return chunks or [token]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    text = str(text).strip()
    if text == "":
        return ["-"]

    wrapped_lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split(" ")
        current_line = ""
        for raw_word in words:
            word = raw_word.strip()
            if word == "":
                continue
            if _text_size(draw, word, font)[0] > max_width:
                if current_line:
                    wrapped_lines.append(current_line)
                    current_line = ""
                wrapped_lines.extend(_split_long_token(draw, word, font, max_width))
                continue
            trial = word if current_line == "" else f"{current_line} {word}"
            if _text_size(draw, trial, font)[0] <= max_width:
                current_line = trial
            else:
                if current_line:
                    wrapped_lines.append(current_line)
                current_line = word
        if current_line:
            wrapped_lines.append(current_line)
        if paragraph == "" and not wrapped_lines:
            wrapped_lines.append("")
    return wrapped_lines or ["-"]


def _group_object_search_entries(task_dir: Path) -> dict[int, list[QaEntry]]:
    samples_path = task_dir / "vlm" / "object_search.json"
    if not samples_path.exists():
        return {}

    grouped: dict[int, list[QaEntry]] = {}
    for sample in _read_json(samples_path):
        metadata = sample.get("metadata", {}) or {}
        episode_idx = int(metadata.get("episode_idx", 0) or 0)
        sample_frame_idx = int(metadata.get("current_frame_idx", 0) or 0)
        user_content = str((sample.get("messages", [{}, {}])[0] or {}).get("content", ""))
        assistant_content = str((sample.get("messages", [{}, {}])[-1] or {}).get("content", ""))
        entry = QaEntry(
            episode_idx=episode_idx,
            sample_frame_idx=sample_frame_idx,
            image_count=int(metadata.get("prompt_image_count", len(sample.get("images", []) or []))),
            question=_user_prompt_text(user_content),
            think=_compact_spaces(_extract_tag_text(assistant_content, "think")),
            info=_compact_spaces(_extract_tag_text(assistant_content, "info")),
            frame=_compact_spaces(_extract_tag_text(assistant_content, "frame")),
            camera=_compact_spaces(_extract_tag_text(assistant_content, "camera")),
            action=_compact_spaces(_extract_tag_text(assistant_content, "action")),
            metadata=dict(metadata),
        )
        grouped.setdefault(episode_idx, []).append(entry)

    for episode_entries in grouped.values():
        episode_entries.sort(key=lambda entry: (int(entry.sample_frame_idx), int(entry.metadata.get("subtask_id", 0))))
    return grouped


def _episode_video_path(task_dir: Path, episode_idx: int) -> Path:
    return task_dir / "video" / f"episode{int(episode_idx)}_annotated.mp4"


def _output_video_path(task_dir: Path, episode_idx: int, suffix: str) -> Path:
    return task_dir / "video" / f"episode{int(episode_idx)}{suffix}.mp4"


def _temp_output_video_path(task_dir: Path, episode_idx: int, suffix: str) -> Path:
    return task_dir / "video" / f"episode{int(episode_idx)}{suffix}.tmp.mp4"


def _build_sections(task_name: str, episode_idx: int, frame_idx: int, total_frames: int, entry: QaEntry | None) -> list[tuple[str, str]]:
    if entry is None:
        return [
            ("Video Frame", f"{frame_idx}/{max(total_frames - 1, 0)}"),
            ("QA Step", "No object-search QA yet"),
            ("Q Images", "-"),
            ("Q", "-"),
            ("Q think", "-"),
            ("info", "-"),
            ("frame", "-"),
            ("camera", "-"),
        ]
    return [
        ("Video Frame", f"{frame_idx}/{max(total_frames - 1, 0)}"),
        (
            "QA Step",
            f"sample_frame={entry.sample_frame_idx} subtask={int(entry.metadata.get('subtask_id', 0))} "
            f"stage={int(entry.metadata.get('stage', 0))}",
        ),
        ("Q Images", str(int(entry.image_count))),
        ("Q", entry.question),
        ("Q think", entry.think or "-"),
        ("info", entry.info or "-"),
        ("frame", entry.frame or "-"),
        ("camera", entry.camera or "-"),
    ]


def _measure_panel_height(
    panel_width: int,
    sections: list[tuple[str, str]],
    title_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> int:
    dummy = Image.new("RGB", (panel_width, 32), color=(0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    pad = 18
    line_gap = 6
    section_gap = 10
    text_width = panel_width - pad * 2
    total_height = pad
    total_height += _text_size(draw, "Object Search QA", title_font)[1]
    total_height += section_gap
    for label, value in sections:
        label_h = _text_size(draw, label, label_font)[1]
        total_height += label_h
        total_height += line_gap
        for line in _wrap_text(draw, value, body_font, text_width):
            total_height += _text_size(draw, line, body_font)[1]
            total_height += line_gap
        total_height += section_gap
    return int(total_height + pad)


def _render_qa_frame(
    frame_bgr: np.ndarray,
    task_name: str,
    episode_idx: int,
    frame_idx: int,
    total_frames: int,
    entry: QaEntry | None,
    panel_width: int,
    canvas_h: int,
) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w = frame_rgb.shape[:2]

    title_font = _load_font(28, bold=True)
    label_font = _load_font(20, bold=True)
    body_font = _load_font(18, bold=False)

    sections = _build_sections(task_name, episode_idx, frame_idx, total_frames, entry)
    canvas_w = frame_w + panel_width

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(14, 18, 25))
    frame_img = Image.fromarray(frame_rgb)
    frame_y = max((canvas_h - frame_h) // 2, 0)
    canvas.paste(frame_img, (0, frame_y))

    draw = ImageDraw.Draw(canvas, "RGBA")
    panel_x0 = frame_w
    draw.rectangle((panel_x0, 0, canvas_w, canvas_h), fill=(22, 28, 38, 255))
    draw.rectangle((panel_x0, 0, panel_x0 + 4, canvas_h), fill=(88, 154, 255, 255))

    pad = 18
    line_gap = 6
    section_gap = 10
    text_width = panel_width - pad * 2
    x = panel_x0 + pad
    y = pad

    draw.text((x, y), "Object Search QA", font=title_font, fill=(245, 247, 250, 255))
    y += _text_size(draw, "Object Search QA", title_font)[1] + section_gap

    for label, value in sections:
        draw.text((x, y), label, font=label_font, fill=(141, 185, 255, 255))
        y += _text_size(draw, label, label_font)[1] + line_gap
        for line in _wrap_text(draw, value, body_font, text_width):
            draw.text((x, y), line, font=body_font, fill=(236, 238, 241, 255))
            y += _text_size(draw, line, body_font)[1] + line_gap
        y += section_gap

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def _episode_canvas_height(
    task_name: str,
    episode_idx: int,
    total_frames: int,
    frame_h: int,
    panel_width: int,
    entries: list[QaEntry],
) -> int:
    title_font = _load_font(28, bold=True)
    label_font = _load_font(20, bold=True)
    body_font = _load_font(18, bold=False)
    max_height = int(frame_h)
    probe_frame_idx = max(int(total_frames) - 1, 0)

    for entry in [None, *entries]:
        sections = _build_sections(task_name, episode_idx, probe_frame_idx, total_frames, entry)
        required = _measure_panel_height(
            panel_width=panel_width,
            sections=sections,
            title_font=title_font,
            label_font=label_font,
            body_font=body_font,
        )
        max_height = max(max_height, int(required))
    return int(max_height)


def render_episode_video(
    task_dir: Path,
    episode_idx: int,
    entries: list[QaEntry],
    output_suffix: str,
    overwrite: bool,
    panel_width: int | None = None,
) -> Path:
    input_path = _episode_video_path(task_dir, episode_idx)
    if not input_path.exists():
        raise FileNotFoundError(f"annotated video not found: {input_path}")
    output_path = _output_video_path(task_dir, episode_idx, output_suffix)
    temp_output_path = _temp_output_video_path(task_dir, episode_idx, output_suffix)
    if output_path.exists() and not overwrite:
        return output_path

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise RuntimeError(f"invalid video size for {input_path}")

    panel_width = int(panel_width or max(560, int(frame_w * 1.15)))
    canvas_h = _episode_canvas_height(
        task_name=task_dir.name,
        episode_idx=episode_idx,
        total_frames=total_frames,
        frame_h=frame_h,
        panel_width=panel_width,
        entries=entries,
    )

    current_entry: QaEntry | None = None
    entry_idx = 0
    writer: cv2.VideoWriter | None = None

    try:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            while entry_idx < len(entries) and int(entries[entry_idx].sample_frame_idx) <= frame_idx:
                current_entry = entries[entry_idx]
                entry_idx += 1

            rendered = _render_qa_frame(
                frame_bgr=frame_bgr,
                task_name=task_dir.name,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                total_frames=total_frames,
                entry=current_entry,
                panel_width=panel_width,
                canvas_h=canvas_h,
            )

            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if temp_output_path.exists():
                    temp_output_path.unlink()
                writer = cv2.VideoWriter(
                    str(temp_output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (int(rendered.shape[1]), int(rendered.shape[0])),
                )
            writer.write(rendered)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        temp_output_path.replace(output_path)
        return output_path

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(temp_output_path),
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
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        temp_output_path.replace(output_path)
        return output_path

    if temp_output_path.exists():
        temp_output_path.unlink()
    return output_path


def _discover_task_dirs(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.glob("*/*")
        if path.is_dir() and (path / "vlm" / "object_search.json").exists() and (path / "video").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render object-search QA panels onto annotated episode videos.")
    parser.add_argument(
        "--task-dir",
        type=str,
        default=None,
        help="Single collected task directory, e.g. data/beat_block_hammer_rotate_view/demo_clean_smoke_headcheck__easy_fan150_headcheck",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory used when --task-dir is not provided. All task dirs with vlm/object_search.json will be processed.",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of episode indices to render.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_annotated_object_search_qa",
        help="Suffix inserted before .mp4 for rendered outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rendered outputs.",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=None,
        help="Optional fixed panel width in pixels.",
    )
    args = parser.parse_args()

    task_dirs = [Path(args.task_dir)] if args.task_dir else _discover_task_dirs(Path(args.data_root))
    if not task_dirs:
        raise SystemExit("no task directories found")

    episode_filter = None if args.episode_indices is None else {int(idx) for idx in args.episode_indices}

    rendered_paths: list[Path] = []
    for task_dir in task_dirs:
        grouped_entries = _group_object_search_entries(task_dir)
        if not grouped_entries:
            continue
        for episode_idx, entries in sorted(grouped_entries.items()):
            if episode_filter is not None and int(episode_idx) not in episode_filter:
                continue
            output_path = render_episode_video(
                task_dir=task_dir,
                episode_idx=episode_idx,
                entries=entries,
                output_suffix=str(args.output_suffix),
                overwrite=bool(args.overwrite),
                panel_width=args.panel_width,
            )
            rendered_paths.append(output_path)
            print(f"[object-search-qa-video] {output_path}")

    if not rendered_paths:
        raise SystemExit("no videos rendered")


if __name__ == "__main__":
    main()
