from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger("hifvla_server")
ACTION_DIM = 18
ACTION_HORIZON = 8
MOTION_SHAPE = (8, 2, 16, 16)


@dataclass(frozen=True)
class RuntimeConfig:
    base_checkpoint: Path
    checkpoint_dir: Path
    hifvla_root: Path
    unnorm_key: str = "astribot_35_mix"
    device: str = "cuda:0"
    history_length: int = 8
    action_horizon: int = 8
    action_dim: int = 18
    lora_rank: int = 32
    center_crop: bool = True
    max_sessions: int = 64

    def validate(self) -> dict[str, Any]:
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"Astribot action_dim must be {ACTION_DIM}, got {self.action_dim}")
        if self.action_horizon != ACTION_HORIZON:
            raise ValueError(f"HIF-VLA action_horizon must be {ACTION_HORIZON}, got {self.action_horizon}")
        if self.history_length != 8:
            raise ValueError(f"HIF-VLA history_length must be 8, got {self.history_length}")
        if self.lora_rank != 32:
            raise ValueError(f"HIF-VLA lora_rank must be 32, got {self.lora_rank}")
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        required_base = [
            "config.json",
            "model.safetensors.index.json",
            "processor_config.json",
        ]
        required_checkpoint = [
            "lora_adapter/adapter_config.json",
            "lora_adapter/adapter_model.safetensors",
            "action_head--150000_checkpoint.pt",
            "proprio_projector--150000_checkpoint.pt",
            "motion_encoder--150000_checkpoint.pt",
            "motion_manager--150000_checkpoint.pt",
            "dataset_statistics.json",
        ]
        errors: list[str] = []
        if not self.hifvla_root.is_dir():
            errors.append(f"missing HIF-VLA source root: {self.hifvla_root}")
        for relative in required_base:
            path = self.base_checkpoint / relative
            if not path.is_file():
                errors.append(f"missing base model file: {path}")
            elif path.stat().st_size <= 0:
                errors.append(f"empty base model file: {path}")
        index_path = self.base_checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shards = sorted(set((index.get("weight_map") or {}).values()))
                if not shards:
                    errors.append(f"base model index has no shards: {index_path}")
                for shard in shards:
                    path = self.base_checkpoint / shard
                    if not path.is_file():
                        errors.append(f"missing base model shard: {path}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"invalid base model index {index_path}: {exc}")
        for relative in required_checkpoint:
            path = self.checkpoint_dir / relative
            if not path.is_file():
                errors.append(f"missing HIF-VLA checkpoint file: {path}")
            elif path.stat().st_size <= 0:
                errors.append(f"empty HIF-VLA checkpoint file: {path}")
        adapter_config_path = self.checkpoint_dir / "lora_adapter/adapter_config.json"
        if adapter_config_path.is_file():
            try:
                adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                if int(adapter_config.get("r", -1)) != self.lora_rank:
                    errors.append(
                        f"LoRA rank mismatch: expected {self.lora_rank}, "
                        f"found {adapter_config.get('r')} in {adapter_config_path}"
                    )
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"invalid LoRA adapter config {adapter_config_path}: {exc}")
        stats_summary: dict[str, Any] = {}
        stats_path = self.checkpoint_dir / "dataset_statistics.json"
        if stats_path.is_file():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                record = stats[self.unnorm_key]
                for name in ("action", "proprio"):
                    group = record[name]
                    for key in ("q01", "q99", "mask"):
                        values = group[key]
                        if not isinstance(values, list) or len(values) != ACTION_DIM:
                            raise ValueError(
                                f"{self.unnorm_key}.{name}.{key} must contain {ACTION_DIM} values"
                            )
                stats_summary = {
                    "unnorm_key": self.unnorm_key,
                    "action_dim": len(record["action"]["q01"]),
                    "proprio_dim": len(record["proprio"]["q01"]),
                }
            except (KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(f"invalid dataset statistics {stats_path}: {exc}")
        if errors:
            raise FileNotFoundError("\n".join(errors))
        return {
            "base_checkpoint": str(self.base_checkpoint),
            "checkpoint_dir": str(self.checkpoint_dir),
            "hifvla_root": str(self.hifvla_root),
            "stats": stats_summary,
        }


def _as_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must have HWC RGB shape, got {image.shape}")
    if image.shape[0] < 8 or image.shape[1] < 8 or image.shape[0] > 4096 or image.shape[1] > 4096:
        raise ValueError(f"image dimensions are outside the supported range: {image.shape}")
    if image.dtype != np.uint8:
        if not np.all(np.isfinite(image)):
            raise ValueError("image contains non-finite values")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _as_state18(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"state must contain {ACTION_DIM} values, got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("state contains non-finite values")
    return state


def validate_act_payload(payload: Any) -> tuple[str, str, np.ndarray, np.ndarray, str]:
    if not isinstance(payload, dict):
        raise TypeError("request body must be a JSON object")
    episode_id = str(payload.get("episode_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    instruction = str(payload.get("instruction", "")).strip()
    if not episode_id or len(episode_id) > 256:
        raise ValueError("episode_id must contain 1 to 256 characters")
    if not request_id or len(request_id) > 256:
        raise ValueError("request_id must contain 1 to 256 characters")
    if not instruction or len(instruction) > 4096:
        raise ValueError("instruction must contain 1 to 4096 characters")
    image = _as_image(payload.get("image"))
    state = _as_state18(payload.get("state"))
    return episode_id, request_id, image, state, instruction


class MotionHistoryStore:
    def __init__(self, history_length: int = 8, max_sessions: int = 64) -> None:
        self.history_length = int(history_length)
        self.frame_count = self.history_length + 1
        self.max_sessions = int(max_sessions)
        self._frames: OrderedDict[str, deque[np.ndarray]] = OrderedDict()
        self._lock = threading.RLock()

    def append(self, episode_id: str, image: np.ndarray) -> list[np.ndarray]:
        with self._lock:
            if episode_id not in self._frames:
                self._frames[episode_id] = deque(maxlen=self.frame_count)
            frames = self._frames.pop(episode_id)
            frames.append(np.ascontiguousarray(image.copy()))
            self._frames[episode_id] = frames
            while len(self._frames) > self.max_sessions:
                self._frames.popitem(last=False)
            return list(frames)

    def reset(self, episode_id: str) -> bool:
        with self._lock:
            return self._frames.pop(episode_id, None) is not None

    def size(self, episode_id: str) -> int:
        with self._lock:
            frames = self._frames.get(episode_id)
            return len(frames) if frames is not None else 0

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._frames)


def build_motion_history(
    frames: list[np.ndarray],
    extractor: Callable[..., Any],
    history_length: int = 8,
) -> np.ndarray:
    frame_count = history_length + 1
    if not frames:
        raise ValueError("motion history requires at least one frame")
    if len(frames) == 1:
        sequence = np.zeros((history_length, 2, 16, 16), dtype=np.float32)
    else:
        selected = frames[-frame_count:]
        sequence = extractor(selected, fps=6, num_frames=len(selected))
        if hasattr(sequence, "detach"):
            sequence = sequence.detach().cpu().numpy()
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 4 or sequence.shape[1:] != (2, 16, 16):
            raise ValueError(f"motion extractor returned invalid shape {sequence.shape}")
        missing = history_length - sequence.shape[0]
        if missing < 0:
            sequence = sequence[-history_length:]
        elif missing > 0:
            pad_value = sequence[:1] if sequence.shape[0] else np.zeros((1, 2, 16, 16), np.float32)
            sequence = np.concatenate([np.repeat(pad_value, missing, axis=0), sequence], axis=0)
    if sequence.shape != MOTION_SHAPE or not np.all(np.isfinite(sequence)):
        raise ValueError(f"motion history must be finite with shape {MOTION_SHAPE}, got {sequence.shape}")
    return sequence[None, ...]


class MockRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.ready = True
        self.inference_lock = threading.Lock()

    def metadata(self) -> dict[str, Any]:
        return {
            "ready": True,
            "mode": "mock",
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "history_length": self.config.history_length,
            "unnorm_key": self.config.unnorm_key,
            "device": "cpu",
        }

    def infer(
        self,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
        frames: list[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, float]]:
        del image, instruction
        started = time.perf_counter()
        actions = np.repeat(state[None, :], self.config.action_horizon, axis=0).astype(np.float32)
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
        actions[:, 15] = np.clip(actions[:, 15], 0.0, 1.0)
        return actions, {
            "total_sec": round(time.perf_counter() - started, 6),
            "history_frames": float(len(frames)),
        }


class HIFVLARuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.ready = False
        self.inference_lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        self.config.validate()
        source = str(self.config.hifvla_root)
        if source not in sys.path:
            sys.path.insert(0, source)

        import torch
        from peft import PeftModel
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForVision2Seq,
            AutoProcessor,
        )
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic.vla.constants import ACTION_DIM as SOURCE_ACTION_DIM
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
        from experiments.robot import openvla_utils
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_hismotion_encoder,
            get_motion_manager,
            get_proprio_projector,
            get_vla_action,
        )
        from experiments.robot.robot_utils import extract_motion_vectors_from_images

        if (SOURCE_ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK) != (18, 18, 8):
            raise RuntimeError(
                "HIF-VLA constants mismatch; launch command must include '--platform astribot': "
                f"action={SOURCE_ACTION_DIM}, proprio={PROPRIO_DIM}, chunk={NUM_ACTIONS_CHUNK}"
            )
        device = torch.device(self.config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        openvla_utils.DEVICE = device

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        LOGGER.info("Loading OpenVLA base from %s", self.config.base_checkpoint)
        base_model = AutoModelForVision2Seq.from_pretrained(
            str(self.config.base_checkpoint),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # Use the HiF-VLA classes registered above. The OpenVLA base
            # directory carries an older modeling_prismatic.py that lacks the
            # multi-image/backbone APIs required by HiF-VLA.
            trust_remote_code=False,
        )
        if not hasattr(base_model.vision_backbone, "set_num_images_in_input"):
            raise RuntimeError(
                "Loaded an incompatible OpenVLA vision backbone. Expected the registered "
                "HiF-VLA modeling_prismatic implementation, but "
                f"{type(base_model.vision_backbone).__name__} has no "
                "set_num_images_in_input()."
            )
        LOGGER.info("Loading LoRA adapter from %s", self.config.checkpoint_dir / "lora_adapter")
        self.vla = PeftModel.from_pretrained(
            base_model,
            str(self.config.checkpoint_dir / "lora_adapter"),
        ).merge_and_unload()
        self.vla.vision_backbone.set_num_images_in_input(1)
        self.vla.eval()
        self.vla = self.vla.to(device)

        stats = json.loads(
            (self.config.checkpoint_dir / "dataset_statistics.json").read_text(encoding="utf-8")
        )
        self.vla.norm_stats = stats
        if self.config.unnorm_key not in self.vla.norm_stats:
            raise KeyError(f"unnorm key not found in checkpoint stats: {self.config.unnorm_key}")

        self.processor = AutoProcessor.from_pretrained(
            str(self.config.base_checkpoint),
            trust_remote_code=False,
        )
        component_cfg = SimpleNamespace(
            pretrained_checkpoint=str(self.config.checkpoint_dir),
            unnorm_key=self.config.unnorm_key,
            num_images_in_input=1,
            use_proprio=True,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            center_crop=self.config.center_crop,
            load_in_8bit=False,
            load_in_4bit=False,
            lora_rank=self.config.lora_rank,
        )
        self.component_cfg = component_cfg
        llm_dim = int(self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(component_cfg, llm_dim, 18)
        self.action_head = get_action_head(component_cfg, llm_dim)
        self.motion_manager = get_motion_manager(component_cfg, llm_dim)
        self.motion_token = self.motion_manager.get_motion_token(batch_size=1)
        self.motion_encoder = get_hismotion_encoder(
            component_cfg,
            in_channels=2,
            hidden_dim=llm_dim // 4,
            out_dim=llm_dim // 4,
            num_frames=self.config.history_length // 2,
            num_patches=64,
        )
        self.extract_motion = extract_motion_vectors_from_images
        self.get_vla_action = get_vla_action
        self.torch = torch
        self.device = device
        self.ready = True

    def metadata(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "mode": "model",
            "checkpoint": str(self.config.checkpoint_dir),
            "base_checkpoint": str(self.config.base_checkpoint),
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "history_length": self.config.history_length,
            "unnorm_key": self.config.unnorm_key,
            "device": str(self.device),
        }

    def infer(
        self,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
        frames: list[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, float]]:
        motion_started = time.perf_counter()
        motion = build_motion_history(frames, self.extract_motion, self.config.history_length)
        motion_tensor = self.torch.from_numpy(motion).to(
            device=self.device,
            dtype=self.torch.bfloat16,
        )
        motion_sec = time.perf_counter() - motion_started
        inference_started = time.perf_counter()
        actions = self.get_vla_action(
            cfg=self.component_cfg,
            vla=self.vla,
            processor=self.processor,
            obs={"full_image": image, "state": state.copy()},
            task_label=instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            motion_token=self.motion_token,
            motion_encoder=self.motion_encoder,
            his_motion_seq=motion_tensor,
            use_film=False,
        )
        inference_sec = time.perf_counter() - inference_started
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"model returned invalid action shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("model returned non-finite actions")
        return array, {
            "motion_sec": round(motion_sec, 6),
            "inference_sec": round(inference_sec, 6),
            "total_sec": round(motion_sec + inference_sec, 6),
            "history_frames": float(len(frames)),
        }


class HIFVLAServer:
    def __init__(
        self,
        runtime: HIFVLARuntime | MockRuntime,
        max_sessions: int = 64,
        max_cached_responses: int = 1024,
    ) -> None:
        if max_cached_responses <= 0:
            raise ValueError("max_cached_responses must be positive")
        self.runtime = runtime
        self.histories = MotionHistoryStore(runtime.config.history_length, max_sessions=max_sessions)
        self.max_cached_responses = int(max_cached_responses)
        self._response_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    def health(self) -> dict[str, Any]:
        payload = self.runtime.metadata()
        payload["sessions"] = self.histories.session_count
        return payload

    def reset(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")
        episode_id = str(payload.get("episode_id", "")).strip()
        if not episode_id or len(episode_id) > 256:
            raise ValueError("episode_id must contain 1 to 256 characters")
        with self.runtime.inference_lock:
            removed = self.histories.reset(episode_id)
            stale_keys = [key for key in self._response_cache if key[0] == episode_id]
            for key in stale_keys:
                del self._response_cache[key]
        return {"ok": True, "episode_id": episode_id, "removed": removed}

    def act(self, payload: Any) -> dict[str, Any]:
        episode_id, request_id, image, state, instruction = validate_act_payload(payload)
        with self.runtime.inference_lock:
            cache_key = (episode_id, request_id)
            cached = self._response_cache.pop(cache_key, None)
            if cached is not None:
                self._response_cache[cache_key] = cached
                return dict(cached)
            frames = self.histories.append(episode_id, image)
            actions, timing = self.runtime.infer(image, state, instruction, frames)
            response = {
                "request_id": request_id,
                "episode_id": episode_id,
                "actions": actions.tolist(),
                "server_timing": timing,
            }
            self._response_cache[cache_key] = response
            while len(self._response_cache) > self.max_cached_responses:
                self._response_cache.popitem(last=False)
            return dict(response)


def create_app(server: HIFVLAServer) -> Any:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="HIF-VLA Astribot Policy Server")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        payload = server.health()
        if not payload.get("ready"):
            raise HTTPException(status_code=503, detail=payload)
        return payload

    @app.post("/reset")
    def reset(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return server.reset(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/act")
    def act(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return server.act(payload)
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.error("Inference failed:\n%s", traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"HIF-VLA inference failed: {type(exc).__name__}: {exc}",
            ) from exc

    return app


def run_mock_http_server(server: HIFVLAServer, host: str, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def read_payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self.send_json(404, {"detail": "not found"})
                return
            self.send_json(200, server.health())

        def do_POST(self) -> None:
            try:
                payload = self.read_payload()
                if self.path == "/reset":
                    response = server.reset(payload)
                elif self.path == "/act":
                    response = server.act(payload)
                else:
                    self.send_json(404, {"detail": "not found"})
                    return
                self.send_json(200, response)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self.send_json(400, {"detail": str(exc)})
            except Exception as exc:
                LOGGER.error("Mock inference failed:\n%s", traceback.format_exc())
                self.send_json(500, {"detail": f"{type(exc).__name__}: {exc}"})

    httpd = ThreadingHTTPServer((host, port), Handler)
    LOGGER.info("Mock HIF-VLA server listening on http://%s:%s", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve HIF-VLA Astribot actions over HTTP.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hifvla-root", type=Path, required=True)
    parser.add_argument("--unnorm-key", default="astribot_35_mix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=18)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--platform", choices=["astribot"], default="astribot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5802)
    parser.add_argument("--max-sessions", type=int, default=64)
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = RuntimeConfig(
        base_checkpoint=args.base_checkpoint.expanduser().resolve(),
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        hifvla_root=args.hifvla_root.expanduser().resolve(),
        unnorm_key=args.unnorm_key,
        device=args.device,
        history_length=args.history_length,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        lora_rank=args.lora_rank,
        center_crop=not args.no_center_crop,
        max_sessions=args.max_sessions,
    )
    validation = config.validate()
    if args.dry_run:
        print(json.dumps({"ok": True, "config": asdict(config), "validation": validation}, default=str, indent=2))
        return
    runtime: HIFVLARuntime | MockRuntime
    runtime = MockRuntime(config) if args.mock else HIFVLARuntime(config)
    server = HIFVLAServer(runtime, max_sessions=config.max_sessions)
    if args.mock:
        run_mock_http_server(server, args.host, args.port)
        return
    app = create_app(server)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
