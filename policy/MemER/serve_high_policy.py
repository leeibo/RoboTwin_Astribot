#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("memer.high")
CONTRACT_VERSION = "memer_high_policy_qwen3vl_v2"
SYSTEM_PROMPT = (
    "You are a robot program that predicts actions.\n"
    "The video input from the egocentric camera shows the most recent actions the robot has executed. "
    "The images are selected frames of particular importance from all the actions the robot has executed so far. "
    "Based on these, output the current subtask the robot should execute and nothing else.\n\n"
    "Return a JSON with:\n"
    "- current_subtask: the action that should be executed at the current timestep\n"
    "- keyframe_positions: list of frame positions (1-indexed) from the video input where actions change\n"
)
ALLOWED_OUTPUT_FIELDS = {"current_subtask", "keyframe_positions"}
EXPECTED_PROCESSOR_BOUNDS = (50176, 115200)


def build_human_prompt(memory_count: int, context_count: int, task_instruction: str) -> str:
    lines = [f"Task: {task_instruction}"]
    if memory_count > 0:
        lines.append(
            "Here are the selected frames from the entirety of the full video that are of particular importance:"
        )
        lines.extend(["<image>"] * memory_count)
    lines.append("Here is a video of the most recent actions the robot has executed:")
    lines.extend(["<image>"] * context_count)
    return "\n".join(lines)


def build_messages(task: str, memory: list[np.ndarray], recent: list[np.ndarray]) -> list[dict[str, Any]]:
    prompt = build_human_prompt(len(memory), len(recent), task)
    images = [*memory, *recent]
    segments = re.split(r"(<image>)", prompt)
    content: list[dict[str, Any]] = []
    image_iter = iter(images)
    for segment in segments:
        if segment == "<image>":
            content.append({"type": "image", "image": Image.fromarray(next(image_iter))})
        elif segment.strip():
            # Match qwen-vl-finetune/qwenvl/data/data_processor.py exactly.
            content.append({"type": "text", "text": segment.strip()})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def validate_processor_contract(processor_dir: Path) -> tuple[int, int]:
    path = processor_dir / "preprocessor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    size = config.get("size") or {}
    bounds = (
        config.get("min_pixels", size.get("shortest_edge")),
        config.get("max_pixels", size.get("longest_edge")),
    )
    if bounds != EXPECTED_PROCESSOR_BOUNDS:
        raise ValueError(
            f"processor_pixel_range_mismatch:{bounds}/{EXPECTED_PROCESSOR_BOUNDS}:{path}"
        )
    return bounds


def parse_high_output(raw: str, recent_count: int) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json:{exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("output_not_object")
    extra = sorted(set(value) - ALLOWED_OUTPUT_FIELDS)
    missing = sorted(ALLOWED_OUTPUT_FIELDS - set(value))
    if extra:
        raise ValueError(f"unsupported_fields:{extra}")
    if missing:
        raise ValueError(f"missing_fields:{missing}")
    subtask = value["current_subtask"]
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError("current_subtask_must_be_nonempty_string")
    positions = value["keyframe_positions"]
    if not isinstance(positions, list):
        raise ValueError("keyframe_positions_must_be_list")
    clean_positions: list[int] = []
    for position in positions:
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(f"keyframe_position_not_integer:{position!r}")
        if position < 1 or position > recent_count:
            raise ValueError(f"keyframe_position_out_of_range:{position}/{recent_count}")
        if position not in clean_positions:
            clean_positions.append(position)
    return {"current_subtask": subtask.strip(), "keyframe_positions": clean_positions}


def cluster_candidate_frames(candidates: list[int], merge_distance: int = 5) -> list[int]:
    """Single-linkage 1D clusters; the representative keeps duplicate votes."""
    if not candidates:
        return []
    ordered = sorted(int(value) for value in candidates)
    clusters: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] <= merge_distance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [int(np.median(cluster)) for cluster in clusters]


def _rgb(value: Any, width: int, height: int) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"frame_must_be_hwc_rgb:{image.shape}")
    if image.dtype != np.uint8:
        if not np.all(np.isfinite(image)):
            raise ValueError("frame_contains_nonfinite_values")
        if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (height, width):
        image = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BICUBIC))
    return np.ascontiguousarray(image)


@dataclass
class EpisodeState:
    environment_type: str
    worker_id: str
    episode_id: str
    task: str
    frames: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict)
    candidate_votes: list[int] = field(default_factory=list)
    last_subtask: str | None = None
    fallback_used_last_call: bool = False
    replan_count: int = 0
    parse_failure_count: int = 0
    retry_success_count: int = 0
    fallback_reuse_count: int = 0
    terminal_parse_failure: bool = False


@dataclass(frozen=True)
class GenerationResult:
    text: str
    trace: dict[str, Any] = field(default_factory=dict)


def _tensor_trace(value: Any) -> dict[str, Any]:
    result = {
        "shape": list(value.shape) if hasattr(value, "shape") else None,
        "dtype": str(value.dtype) if hasattr(value, "dtype") else type(value).__name__,
    }
    if hasattr(value, "device"):
        result["device"] = str(value.device)
    return result


def _safe_component(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return clean[:80] or "unknown"


class VLMTraceLogger:
    def __init__(self, root: Path | None, save_images: bool = True) -> None:
        self.root = root
        self.save_images = save_images
        self.lock = threading.Lock()
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def _image_record(
        self, state: EpisodeState, role: str, position: int, step: int, image: np.ndarray
    ) -> dict[str, Any]:
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        relative: Path | None = None
        if self.root is not None and self.save_images:
            episode = _safe_component(state.episode_id)
            relative = Path("images") / _safe_component(state.environment_type) / _safe_component(state.worker_id)
            relative /= Path(episode) / f"frame_{step:06d}_{digest[:12]}.jpg"
            target = self.root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(target, format="JPEG", quality=92)
        return {
            "role": role,
            "position": position,
            "env_step": step,
            "path": str(relative) if relative is not None else None,
            "sha256": digest,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min": int(image.min()),
            "max": int(image.max()),
            "mean": round(float(image.mean()), 4),
            "std": round(float(image.std()), 4),
        }

    def write_attempt(
        self,
        *,
        state: EpisodeState,
        session_id: str,
        env_step: int,
        attempt: int,
        memory_steps: list[int],
        memory_images: list[np.ndarray],
        recent_steps: list[int],
        recent_images: list[np.ndarray],
        result: GenerationResult,
        parse_error: str | None,
        latency: float,
    ) -> str | None:
        if self.root is None:
            return None
        log_id = f"{_safe_component(state.episode_id)}_step{env_step:06d}_attempt{attempt}"
        with self.lock:
            images = [
                self._image_record(state, "memory", idx + 1, step, image)
                for idx, (step, image) in enumerate(zip(memory_steps, memory_images, strict=True))
            ]
            images.extend(
                self._image_record(state, "recent", idx + 1, step, image)
                for idx, (step, image) in enumerate(zip(recent_steps, recent_images, strict=True))
            )
            record = {
                "kind": "vlm_attempt",
                "log_id": log_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "environment_type": state.environment_type,
                "worker_id": state.worker_id,
                "episode_id": state.episode_id,
                "session_id": session_id,
                "env_step": env_step,
                "replan_count": state.replan_count + 1,
                "attempt": attempt,
                "task": state.task,
                "system_prompt": SYSTEM_PROMPT,
                "human_prompt": build_human_prompt(len(memory_images), len(recent_images), state.task),
                "image_order": images,
                "memory_frame_steps": memory_steps,
                "recent_frame_steps": recent_steps,
                "generator_trace": result.trace,
                "raw_output": result.text,
                "parse_error": parse_error,
                "generation_latency_sec": round(latency, 6),
            }
            with (self.root / "vlm_requests.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return log_id


class SessionStore:
    def __init__(self, max_sessions: int = 32) -> None:
        self.max_sessions = max_sessions
        self.states: OrderedDict[str, EpisodeState] = OrderedDict()
        self.lock = threading.RLock()

    def reset(self, session_id: str) -> bool:
        with self.lock:
            return self.states.pop(session_id, None) is not None

    def get_or_create(self, session_id: str, **kwargs: str) -> EpisodeState:
        with self.lock:
            state = self.states.get(session_id)
            if state is not None:
                identity = {
                    "environment_type": state.environment_type,
                    "worker_id": state.worker_id,
                    "episode_id": state.episode_id,
                    "task": state.task,
                }
                if identity != kwargs:
                    raise ValueError(f"session_identity_changed_without_reset:{identity!r}->{kwargs!r}")
                self.states.move_to_end(session_id)
                return state
            if len(self.states) >= self.max_sessions:
                self.states.popitem(last=False)
            state = EpisodeState(**kwargs)
            self.states[session_id] = state
            return state


class MockGenerator:
    mode = "mock"

    def generate(self, task: str, memory: list[np.ndarray], recent: list[np.ndarray]) -> GenerationResult:
        del memory, recent
        return GenerationResult(json.dumps({"current_subtask": f"mock subtask for {task}", "keyframe_positions": []}))


class QwenGenerator:
    mode = "model"

    def __init__(
        self, checkpoint: Path, processor_dir: Path, device: str, max_new_tokens: int, attn: str
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        if attn != "auto":
            kwargs["attn_implementation"] = attn
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(checkpoint), **kwargs)
        self.model.eval().to(device)
        self.model.config.use_cache = True
        self.processor = AutoProcessor.from_pretrained(str(processor_dir))
        self.device = device

    def generate(self, task: str, memory: list[np.ndarray], recent: list[np.ndarray]) -> GenerationResult:
        messages = build_messages(task, memory, recent)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        trace = {
            "processor_inputs": {key: _tensor_trace(value) for key, value in inputs.items()},
            "input_token_count": int(inputs["input_ids"].shape[1]),
            "generated_token_count": int(generated.shape[1]),
            "generated_token_ids": generated[0].detach().cpu().tolist(),
        }
        if "image_grid_thw" in inputs:
            trace["image_grid_thw"] = inputs["image_grid_thw"].detach().cpu().tolist()
        return GenerationResult(text=text, trace=trace)


class HighPolicyService:
    def __init__(
        self, generator: Any, recent_frames: int, memory_frames: int,
        recent_frame_interval: int, merge_distance: int, width: int, height: int,
        max_sessions: int, server_token: str = "", vlm_log_dir: Path | None = None,
        vlm_log_images: bool = True,
    ) -> None:
        self.generator = generator
        self.recent_frames = recent_frames
        self.memory_frames = memory_frames
        self.recent_frame_interval = recent_frame_interval
        self.merge_distance = merge_distance
        self.width = width
        self.height = height
        self.server_token = server_token
        self.store = SessionStore(max_sessions)
        self.inference_lock = threading.Lock()
        self.vlm_logger = VLMTraceLogger(vlm_log_dir, vlm_log_images)

    def metadata(self) -> dict[str, Any]:
        return {
            "ready": True, "mode": self.generator.mode, "contract_version": CONTRACT_VERSION,
            "recent_frames": self.recent_frames,
            "memory_frames": self.memory_frames, "recent_frame_interval": self.recent_frame_interval,
            "image_width": self.width, "image_height": self.height, "image_layout": "HWC RGB uint8",
            "merge_distance": self.merge_distance, "sessions": len(self.store.states),
            "server_token": self.server_token,
            "vlm_log_dir": str(self.vlm_logger.root) if self.vlm_logger.root is not None else None,
        }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id_required")
        return {"ok": True, "removed": self.store.reset(session_id), "session_id": session_id}

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = ("session_id", "environment_type", "worker_id", "episode_id", "task", "env_step", "frames")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"missing_request_fields:{missing}")
        session_id = str(payload["session_id"]).strip()
        if not session_id:
            raise ValueError("session_id_required")
        env_step = int(payload["env_step"])
        state = self.store.get_or_create(
            session_id,
            environment_type=str(payload["environment_type"]),
            worker_id=str(payload["worker_id"]),
            episode_id=str(payload["episode_id"]),
            task=str(payload["task"]).strip(),
        )
        frame_rows = payload["frames"]
        if not isinstance(frame_rows, list) or not frame_rows:
            raise ValueError("frames_must_be_nonempty_list")
        for row in frame_rows:
            step = int(row["env_step"])
            state.frames[step] = _rgb(row["image"], self.width, self.height)
        frame_steps = list(state.frames)
        recent_steps = frame_steps[::-1][::self.recent_frame_interval][:self.recent_frames][::-1]
        recent_images = [state.frames[step] for step in recent_steps]
        representatives = cluster_candidate_frames(state.candidate_votes, self.merge_distance)
        context_start = recent_steps[0]
        memory_steps = [
            step for step in representatives if step in state.frames and step < context_start
        ][-self.memory_frames:]
        memory_images = [state.frames[step] for step in memory_steps]

        previous = state.last_subtask
        attempts: list[dict[str, str | None]] = []
        vlm_log_ids: list[str] = []
        parsed: dict[str, Any] | None = None
        latency = 0.0
        for attempt in range(2):
            started = time.perf_counter()
            with self.inference_lock:
                generated = self.generator.generate(state.task, memory_images, recent_images)
            attempt_latency = time.perf_counter() - started
            latency += attempt_latency
            result = generated if isinstance(generated, GenerationResult) else GenerationResult(str(generated))
            raw = result.text
            parse_error: str | None = None
            try:
                parsed = parse_high_output(raw, len(recent_steps))
                attempts.append({"raw": raw, "error": None})
                if attempt == 1:
                    state.retry_success_count += 1
            except ValueError as exc:
                parse_error = str(exc)
                state.parse_failure_count += 1
                attempts.append({"raw": raw, "error": parse_error})
            try:
                log_id = self.vlm_logger.write_attempt(
                    state=state, session_id=session_id, env_step=env_step, attempt=attempt + 1,
                    memory_steps=memory_steps, memory_images=memory_images,
                    recent_steps=recent_steps, recent_images=recent_images,
                    result=result, parse_error=parse_error, latency=attempt_latency,
                )
                if log_id is not None:
                    vlm_log_ids.append(log_id)
            except Exception:
                LOGGER.exception("failed to write VLM trace log")
            if parsed is not None:
                break
        fallback = False
        terminal = False
        if parsed is not None:
            state.last_subtask = parsed["current_subtask"]
            state.fallback_used_last_call = False
            for position in parsed["keyframe_positions"]:
                state.candidate_votes.append(recent_steps[position - 1])
        elif state.last_subtask is not None and not state.fallback_used_last_call:
            fallback = True
            state.fallback_used_last_call = True
            state.fallback_reuse_count += 1
        else:
            terminal = True
            state.terminal_parse_failure = True

        state.replan_count += 1
        representatives = cluster_candidate_frames(state.candidate_votes, self.merge_distance)
        record = {
            "environment_type": state.environment_type,
            "worker_id": state.worker_id,
            "episode_id": state.episode_id,
            "session_id": session_id,
            "env_step": env_step,
            "previous_subtask": previous,
            "new_subtask": state.last_subtask if not terminal else None,
            "replan_reason": str(payload.get("replan_reason", "interval")),
            "high_level_latency": round(latency, 6),
            "high_level_parse_failure_count": state.parse_failure_count,
            "high_level_retry_success_count": state.retry_success_count,
            "subtask_fallback_reuse_count": state.fallback_reuse_count,
            "terminal_parse_failure": terminal,
            "fallback_reuse": fallback,
            "raw_high_level_output": [item["raw"] for item in attempts],
            "parse_error_reason": [item["error"] for item in attempts if item["error"]],
            "recent_frame_steps": recent_steps,
            "memory_frame_steps": memory_steps,
            "candidate_votes": list(state.candidate_votes),
            "memory_representatives": representatives,
            "replan_count": state.replan_count,
            "vlm_log_ids": vlm_log_ids,
        }
        LOGGER.info("event=%s", json.dumps(record, ensure_ascii=False))
        return record


def make_handler(service: HighPolicyService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            LOGGER.info("http " + fmt, *args)

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self) -> dict[str, Any]:
            size = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request_body_must_be_object")
            return value

        def do_GET(self) -> None:
            self.send_json(200, service.metadata()) if self.path == "/healthz" else self.send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            try:
                if self.path == "/reset":
                    self.send_json(200, service.reset(self.body()))
                elif self.path == "/act":
                    self.send_json(200, service.act(self.body()))
                else:
                    self.send_json(404, {"error": "not_found"})
            except Exception as exc:
                LOGGER.exception("request failed")
                self.send_json(400, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--processor-dir", type=Path,
        help="Qwen3-VL processor source; defaults to the fine-tuned checkpoint directory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recent-frames", type=int, default=8)
    parser.add_argument("--memory-frames", type=int, default=8)
    parser.add_argument("--recent-frame-interval", type=int, default=5)
    parser.add_argument("--merge-distance", type=int, default=5)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-sessions", type=int, default=32)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--server-token", default="")
    parser.add_argument("--vlm-log-dir", type=Path)
    parser.add_argument("--no-vlm-log-images", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor_dir = args.processor_dir or args.checkpoint
    required = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]
    missing = [str(args.checkpoint / name) for name in required if not (args.checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing high checkpoint files: {missing}")
    processor_required = [
        "preprocessor_config.json", "video_preprocessor_config.json", "tokenizer_config.json"
    ]
    processor_missing = [
        str(processor_dir / name) for name in processor_required if not (processor_dir / name).is_file()
    ]
    if not args.mock and processor_missing:
        raise FileNotFoundError(
            "missing Qwen3-VL processor files: "
            f"{processor_missing}. Copy preprocessor_config.json and "
            "video_preprocessor_config.json from the high-level training output directory, "
            "or pass --processor-dir /path/to/high-level-training-output"
        )
    processor_bounds = validate_processor_contract(processor_dir)
    if min(args.recent_frames, args.memory_frames, args.recent_frame_interval, args.max_new_tokens) <= 0:
        raise ValueError("frame and generation limits must be positive")
    if args.dry_run:
        print(json.dumps({
            "ok": True,
            "checkpoint": str(args.checkpoint),
            "processor_dir": str(processor_dir),
            "processor_bounds": list(processor_bounds),
            "required": required,
            "processor_required": processor_required,
        }, indent=2))
        return
    generator = MockGenerator() if args.mock else QwenGenerator(
        args.checkpoint, processor_dir, args.device, args.max_new_tokens, args.attn_implementation
    )
    service = HighPolicyService(
        generator, args.recent_frames, args.memory_frames, args.recent_frame_interval,
        args.merge_distance, args.image_width, args.image_height, args.max_sessions, args.server_token,
        args.vlm_log_dir, not args.no_vlm_log_images,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    LOGGER.info("input_contract=%s", json.dumps(service.metadata(), ensure_ascii=False))
    LOGGER.info("MemER high policy listening on http://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
