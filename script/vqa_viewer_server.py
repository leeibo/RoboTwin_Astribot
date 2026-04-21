#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPO_ROOT / "web" / "vqa_viewer"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
TASK_TYPE_SPECS: dict[str, dict[str, str]] = {
    "object_search": {
        "dir_name": "vlm",
        "file_name": "object_search.json",
        "label": "Object Search",
    },
    "angle_delta": {
        "dir_name": "vlm",
        "file_name": "angle_delta.json",
        "label": "Angle Delta",
    },
    "memory_compression_vqa": {
        "dir_name": "vlm",
        "file_name": "memory_compression_vqa.json",
        "label": "Memory Compression",
    },
    "object_search_visibility_memory_v1": {
        "dir_name": "vlm_object_search_visibility_memory_v1",
        "file_name": "object_search_visibility_memory_v1.json",
        "label": "Object Search Visibility Memory V1",
    },
    "object_search_visibility_memory_v2": {
        "dir_name": "vlm_object_search_visibility_memory_v2",
        "file_name": "object_search_visibility_memory_v2.json",
        "label": "Object Search Visibility Memory V2",
    },
}
TASK_TYPES = tuple(TASK_TYPE_SPECS.keys())
TEXT_TAGS = ("think", "info", "frame", "camera", "action", "answer")


def _compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_tag_text(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", str(content), flags=re.DOTALL)
    return "" if match is None else _compact_spaces(str(match.group(1)))


def _truncate(text: str, limit: int = 180) -> str:
    text = _compact_spaces(text)
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _user_prompt_text(content: str) -> str:
    return _compact_spaces(re.sub(r"<image>", "", str(content)))


def _parse_response_tags(content: str) -> dict[str, str]:
    return {tag: _extract_tag_text(content, tag) for tag in TEXT_TAGS}


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _repo_url(path: Path) -> str:
    return f"/repo/{quote(_repo_relative(path))}"


def _resolve_repo_relative(rel_path: str) -> Path:
    candidate = (REPO_ROOT / unquote(rel_path)).resolve()
    repo_root_resolved = REPO_ROOT.resolve()
    if repo_root_resolved not in candidate.parents and candidate != repo_root_resolved:
        raise ValueError("path escapes repository root")
    return candidate


def _task_type_path(task_dir: Path, task_type: str) -> Path:
    if task_type not in TASK_TYPE_SPECS:
        raise ValueError(f"unsupported task type: {task_type}")
    spec = TASK_TYPE_SPECS[task_type]
    return task_dir / spec["dir_name"] / spec["file_name"]


def _iter_task_dirs(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.glob("*/*")
        if path.is_dir()
        and (path / "video").exists()
        and any(_task_type_path(path, task_type).exists() for task_type in TASK_TYPES)
    )


def _parse_episode_index(name: str) -> int | None:
    match = re.search(r"episode(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def _video_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "url": None, "fps": 30.0, "frame_count": None}
    fps = 30.0
    frame_count = None
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        try:
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                frame_count = _safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), None)
        finally:
            cap.release()
    return {
        "exists": True,
        "url": _repo_url(path),
        "fps": fps,
        "frame_count": frame_count,
    }


def _episode_assets(task_dir: Path) -> dict[str, dict[str, Any]]:
    episode_set: set[int] = set()
    for folder, pattern in [
        ("video", "episode*.mp4"),
        ("data", "episode*.hdf5"),
        ("subtask_metadata", "episode*.json"),
    ]:
        target_dir = task_dir / folder
        if not target_dir.exists():
            continue
        for path in target_dir.glob(pattern):
            episode_idx = _parse_episode_index(path.name)
            if episode_idx is not None:
                episode_set.add(int(episode_idx))

    assets: dict[str, dict[str, Any]] = {}
    for episode_idx in sorted(episode_set):
        annotated_path = task_dir / "video" / f"episode{episode_idx}_annotated.mp4"
        qa_path = task_dir / "video" / f"episode{episode_idx}_annotated_object_search_qa.mp4"
        main_path = task_dir / "video" / f"episode{episode_idx}.mp4"
        assets[str(episode_idx)] = {
            "annotated": _video_info(annotated_path),
            "qa_overlay": _video_info(qa_path),
            "main": _video_info(main_path),
        }
    return assets


@dataclass
class CachedJson:
    mtime_ns: int
    payload: Any


class VqaDataset:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self._json_cache: dict[Path, CachedJson] = {}
        self._index_cache: dict[str, Any] | None = None
        self._index_mtime_signature: tuple[tuple[str, int], ...] | None = None
        self._lock = threading.Lock()

    def _load_json(self, path: Path) -> Any:
        path = path.resolve()
        stat = path.stat()
        cached = self._json_cache.get(path)
        if cached is not None and cached.mtime_ns == stat.st_mtime_ns:
            return cached.payload
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        self._json_cache[path] = CachedJson(mtime_ns=stat.st_mtime_ns, payload=payload)
        return payload

    def _index_signature(self) -> tuple[tuple[str, int], ...]:
        signature: list[tuple[str, int]] = []
        for task_dir in _iter_task_dirs(self.data_root):
            for task_type in TASK_TYPES:
                path = _task_type_path(task_dir, task_type)
                if path.exists():
                    signature.append((_repo_relative(path), path.stat().st_mtime_ns))
        return tuple(signature)

    def build_index(self, force_refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            signature = self._index_signature()
            if not force_refresh and self._index_cache is not None and signature == self._index_mtime_signature:
                return self._index_cache

            tasks: list[dict[str, Any]] = []
            storage_names: set[str] = set()
            totals = {task_type: 0 for task_type in TASK_TYPES}

            for task_dir in _iter_task_dirs(self.data_root):
                task_name = task_dir.parent.name
                storage_name = task_dir.name
                storage_names.add(storage_name)
                episode_assets = _episode_assets(task_dir)

                counts: dict[str, int] = {}
                available_types: list[str] = []
                for task_type in TASK_TYPES:
                    path = _task_type_path(task_dir, task_type)
                    if not path.exists():
                        counts[task_type] = 0
                        continue
                    payload = self._load_json(path)
                    count = len(payload) if isinstance(payload, list) else 0
                    counts[task_type] = count
                    if count > 0 or path.exists():
                        available_types.append(task_type)
                    totals[task_type] += count

                episodes = sorted(int(key) for key in episode_assets.keys())
                tasks.append(
                    {
                        "task_name": task_name,
                        "storage_name": storage_name,
                        "task_dir": _repo_relative(task_dir),
                        "available_types": available_types,
                        "sample_counts": counts,
                        "episode_count": len(episodes),
                        "episodes": episodes,
                        "episode_assets": episode_assets,
                    }
                )

            payload = {
                "generated_at": int(time.time()),
                "data_root": _repo_relative(self.data_root),
                "storages": sorted(storage_names),
                "task_types": list(TASK_TYPES),
                "task_type_labels": {task_type: spec["label"] for task_type, spec in TASK_TYPE_SPECS.items()},
                "task_count": len(tasks),
                "totals": totals,
                "tasks": tasks,
            }
            self._index_cache = payload
            self._index_mtime_signature = signature
            return payload

    def _task_dir(self, task_name: str, storage_name: str) -> Path:
        task_dir = (self.data_root / task_name / storage_name).resolve()
        if not task_dir.exists():
            raise FileNotFoundError(f"task dir not found: {task_name}/{storage_name}")
        if self.data_root not in task_dir.parents and task_dir != self.data_root:
            raise FileNotFoundError("task dir escapes data root")
        return task_dir

    def list_samples(self, task_name: str, storage_name: str, task_type: str) -> dict[str, Any]:
        if task_type not in TASK_TYPES:
            raise ValueError(f"unsupported task type: {task_type}")
        task_dir = self._task_dir(task_name, storage_name)
        samples_path = _task_type_path(task_dir, task_type)
        samples = self._load_json(samples_path) if samples_path.exists() else []
        summaries: list[dict[str, Any]] = []

        for sample_idx, sample in enumerate(samples):
            metadata = sample.get("metadata", {}) or {}
            messages = sample.get("messages", []) or []
            user_content = str((messages[0] if messages else {}).get("content", ""))
            assistant_content = str((messages[-1] if messages else {}).get("content", ""))
            parsed = _parse_response_tags(assistant_content)
            preview = parsed.get("think") or parsed.get("answer") or assistant_content

            current_frame_idx = metadata.get("current_frame_idx", None)
            if current_frame_idx is None:
                frame_indices = metadata.get("frame_indices", []) or []
                current_frame_idx = frame_indices[-1] if frame_indices else None
            stage_value = metadata.get("stage", metadata.get("raw_stage", None))

            summaries.append(
                {
                    "sample_idx": sample_idx,
                    "episode_idx": _safe_int(metadata.get("episode_idx"), 0),
                    "subtask_id": _safe_int(metadata.get("subtask_id"), None),
                    "stage": _safe_int(stage_value, None),
                    "current_frame_idx": _safe_int(current_frame_idx, None),
                    "prompt_image_count": _safe_int(
                        metadata.get("prompt_image_count"), len(sample.get("images", []) or [])
                    ),
                    "roles": list(metadata.get("roles", []) or []),
                    "evidence_from_history": bool(metadata.get("evidence_from_history", False)),
                    "evidence_prompt_index": _safe_int(metadata.get("evidence_prompt_index"), None),
                    "camera_delta_deg": _safe_int(metadata.get("camera_delta_deg"), None),
                    "user_preview": _truncate(_user_prompt_text(user_content), 140),
                    "assistant_preview": _truncate(preview, 180),
                    "task_type": task_type,
                }
            )

        return {
            "task_name": task_name,
            "storage_name": storage_name,
            "task_type": task_type,
            "task_dir": _repo_relative(task_dir),
            "episode_assets": _episode_assets(task_dir),
            "sample_count": len(summaries),
            "samples": summaries,
        }

    def sample_detail(self, task_name: str, storage_name: str, task_type: str, sample_idx: int) -> dict[str, Any]:
        if task_type not in TASK_TYPES:
            raise ValueError(f"unsupported task type: {task_type}")
        task_dir = self._task_dir(task_name, storage_name)
        samples_path = _task_type_path(task_dir, task_type)
        samples = self._load_json(samples_path) if samples_path.exists() else []
        if sample_idx < 0 or sample_idx >= len(samples):
            raise IndexError(f"sample index out of range: {sample_idx}")

        sample = samples[sample_idx]
        metadata = sample.get("metadata", {}) or {}
        messages = sample.get("messages", []) or []
        user_content = str((messages[0] if messages else {}).get("content", ""))
        assistant_content = str((messages[-1] if messages else {}).get("content", ""))
        parsed = _parse_response_tags(assistant_content)
        images = [str(item) for item in (sample.get("images", []) or [])]
        action_payload = sample.get("action", None)
        action_rows = len(action_payload) if isinstance(action_payload, list) else 0
        action_dims = 0
        if isinstance(action_payload, list) and action_payload and isinstance(action_payload[0], list):
            action_dims = len(action_payload[0])

        episode_idx = _safe_int(metadata.get("episode_idx"), 0)
        episode_asset_map = _episode_assets(task_dir)
        episode_assets = episode_asset_map.get(str(episode_idx), {})

        return {
            "task_name": task_name,
            "storage_name": storage_name,
            "task_type": task_type,
            "sample_idx": int(sample_idx),
            "images": [_repo_url(_resolve_repo_relative(path)) for path in images],
            "image_paths": images,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "parsed": parsed,
            "metadata": metadata,
            "episode_assets": episode_assets,
            "action_stats": {
                "rows": action_rows,
                "dims": action_dims,
            },
            "raw_json": sample,
        }


class VqaViewerHandler(BaseHTTPRequestHandler):
    server_version = "RoboTwinVqaViewer/1.0"

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text_error(self, status: int, message: str) -> None:
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, head_only: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self._text_error(HTTPStatus.NOT_FOUND, f"not found: {path}")
            return

        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = max(file_size - 1, 0)

        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match is None:
                self._text_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "invalid range")
                return
            start_text, end_text = match.groups()
            if start_text:
                start = int(start_text)
            if end_text:
                end = int(end_text)
            if start >= file_size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            end = min(end, file_size - 1)
            content_length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        else:
            content_length = file_size
            self.send_response(HTTPStatus.OK)

        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if head_only:
            return

        with open(path, "rb") as file:
            file.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _handle_request(self, head_only: bool = False) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        dataset: VqaDataset = self.server.dataset  # type: ignore[attr-defined]

        try:
            if route == "/" or route == "/index.html":
                return self._serve_file(STATIC_ROOT / "index.html", head_only=head_only)
            if route == "/static/styles.css":
                return self._serve_file(STATIC_ROOT / "styles.css", head_only=head_only)
            if route == "/static/app.js":
                return self._serve_file(STATIC_ROOT / "app.js", head_only=head_only)
            if route.startswith("/repo/"):
                rel_path = route[len("/repo/") :]
                return self._serve_file(_resolve_repo_relative(rel_path), head_only=head_only)
            if route == "/api/health":
                return self._json({"ok": True, "ts": int(time.time())})
            if route == "/api/index":
                force_refresh = query.get("refresh", ["0"])[0] == "1"
                return self._json(dataset.build_index(force_refresh=force_refresh))
            if route == "/api/samples":
                task_name = query.get("task", [""])[0]
                storage_name = query.get("storage", [""])[0]
                task_type = query.get("type", ["object_search"])[0]
                if not task_name or not storage_name:
                    return self._text_error(HTTPStatus.BAD_REQUEST, "task and storage are required")
                return self._json(dataset.list_samples(task_name, storage_name, task_type))
            if route == "/api/sample-detail":
                task_name = query.get("task", [""])[0]
                storage_name = query.get("storage", [""])[0]
                task_type = query.get("type", ["object_search"])[0]
                sample_idx = _safe_int(query.get("index", ["0"])[0], None)
                if not task_name or not storage_name or sample_idx is None:
                    return self._text_error(HTTPStatus.BAD_REQUEST, "task, storage and index are required")
                return self._json(dataset.sample_detail(task_name, storage_name, task_type, sample_idx))
        except FileNotFoundError as exc:
            return self._text_error(HTTPStatus.NOT_FOUND, str(exc))
        except (ValueError, IndexError) as exc:
            return self._text_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover
            return self._text_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"internal error: {exc}")

        return self._text_error(HTTPStatus.NOT_FOUND, f"unknown route: {route}")

    def do_GET(self) -> None:  # noqa: N802
        self._handle_request(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_request(head_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local VQA Viewer for RoboTwin datasets.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise SystemExit(f"data root does not exist: {data_root}")
    if not STATIC_ROOT.exists():
        raise SystemExit(f"viewer static dir does not exist: {STATIC_ROOT}")

    dataset = VqaDataset(data_root=data_root)
    server = ThreadingHTTPServer((args.host, args.port), VqaViewerHandler)
    server.dataset = dataset  # type: ignore[attr-defined]

    print(f"[vqa-viewer] Serving {data_root} at http://{args.host}:{args.port}")
    print(f"[vqa-viewer] Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
