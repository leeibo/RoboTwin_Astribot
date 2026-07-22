#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("memer.low")
STATE_ACTION_ORDER = [
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "left_gripper", "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4",
    "right_arm_5", "right_arm_6", "right_gripper", "torso_yaw", "head_2",
]
DELTA_ACTION_MASK = [
    True, True, True, True, True, True, True, False,
    True, True, True, True, True, True, True, False, True, True,
]


def validate_training_contract(norm_path: Path, action_horizon: int) -> dict[str, Any]:
    payload = json.loads(norm_path.read_text(encoding="utf-8"))
    norm_stats = payload.get("norm_stats")
    if not isinstance(norm_stats, dict):
        raise ValueError(f"{norm_path} has no norm_stats object")
    for group in ("state", "actions"):
        stats = norm_stats.get(group)
        if not isinstance(stats, dict):
            raise ValueError(f"{norm_path} has no norm_stats.{group}")
        for metric in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats.get(metric), dtype=np.float64)
            if values.shape != (32,) or not np.all(np.isfinite(values)):
                raise ValueError(f"norm_stats.{group}.{metric} must be finite (32,), got {values.shape}")
    return {
        "image": {"source": "observation.images.camera_head", "layout": "HWC", "dtype": "uint8"},
        "state": {"environment_dim": 18, "model_dim_after_padding": 32, "order": STATE_ACTION_ORDER},
        "action": {
            "environment_dim": 18, "model_dim_after_padding": 32,
            "horizon": action_horizon, "order": STATE_ACTION_ORDER,
            "training_transform": "absolute_to_delta", "inference_output": "absolute",
            "delta_mask": DELTA_ACTION_MASK,
        },
        "prompt": "current subtask",
    }


class MockPolicy:
    mode = "mock"

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(observation["observation/state"], dtype=np.float32)
        return {"actions": np.repeat(state[None, :], 50, axis=0)}

    def reset(self) -> None:
        return None


class OpenPIPolicy:
    mode = "model"

    def __init__(
        self, checkpoint: Path, rlinf_root: Path, config_name: str,
        device: str, num_steps: int, action_horizon: int,
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        sys.path.insert(0, str(rlinf_root))
        os.environ["OPENPI_ASSETS_DIR"] = str(checkpoint)
        from rlinf.models.embodiment.openpi import get_model

        cfg = OmegaConf.create({
            "model_path": str(checkpoint),
            "openpi": {
                "config_name": config_name,
                "num_images_in_input": 3,
                "action_env_dim": 18,
                "action_chunk": action_horizon,
                "num_steps": num_steps,
                "train_expert_only": False,
                "detach_critic_input": True,
            },
        })
        torch.set_grad_enabled(False)
        self.model = get_model(cfg)
        self.model.eval().to(device)
        self.device = device
        self.torch = torch

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(observation["observation/image"])
        state = np.asarray(observation["observation/state"], dtype=np.float32)
        prompt = str(observation["prompt"]).strip()
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB image, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")
        if state.shape != (18,) or not np.all(np.isfinite(state)):
            raise ValueError(f"expected finite state (18,), got {state.shape}")
        if not prompt:
            raise ValueError("low-level prompt is empty")
        env_obs = {
            "main_images": self.torch.as_tensor(image[None], dtype=self.torch.uint8, device=self.device),
            "states": self.torch.as_tensor(state[None], dtype=self.torch.float32, device=self.device),
            "task_descriptions": [prompt],
            "wrist_images": None,
            "extra_view_images": None,
        }
        with self.torch.inference_mode():
            actions, _ = self.model.predict_action_batch(env_obs, mode="eval", compute_values=False)
        return {"actions": np.asarray(self.torch.as_tensor(actions).detach().cpu())}

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()


def validate_actions(response: dict[str, Any], action_horizon: int) -> np.ndarray:
    actions = np.asarray(response.get("actions"), dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != 18 or actions.shape[0] == 0:
        raise ValueError(f"expected non-empty actions (T,18), got {actions.shape}")
    if actions.shape[0] != action_horizon:
        raise ValueError(f"expected action horizon {action_horizon}, got {actions.shape[0]}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain non-finite values")
    return actions


class LowPolicyServer:
    def __init__(
        self, policy: Any, host: str, port: int, action_horizon: int,
        client_src: Path, server_token: str = "",
    ) -> None:
        sys.path.insert(0, str(client_src))
        from openpi_client import msgpack_numpy

        self.msgpack = msgpack_numpy
        self.policy = policy
        self.host = host
        self.port = port
        self.action_horizon = action_horizon
        self.server_token = server_token
        self.lock = asyncio.Lock()
        self.metadata = {
            "ready": True, "mode": policy.mode, "action_dim": 18,
            "action_horizon": action_horizon, "server_token": server_token,
            "training_contract": getattr(policy, "training_contract", {}),
        }

    async def handler(self, websocket: Any) -> None:
        from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

        packer = self.msgpack.Packer()
        await websocket.send(packer.pack(self.metadata))
        while True:
            try:
                request = self.msgpack.unpackb(await websocket.recv())
                if request.get("__command__") == "reset":
                    async with self.lock:
                        self.policy.reset()
                    await websocket.send(packer.pack({"ok": True}))
                    continue
                started = time.perf_counter()
                async with self.lock:
                    response = self.policy.infer(request)
                actions = validate_actions(response, self.action_horizon)
                elapsed = time.perf_counter() - started
                payload = {
                    "actions": actions,
                    "server_timing": {"infer_ms": elapsed * 1000.0},
                    "prompt": str(request.get("prompt", "")),
                }
                image = np.asarray(request.get("observation/image"))
                state = np.asarray(request.get("observation/state"), dtype=np.float32)
                diagnostic = {
                    "prompt": payload["prompt"],
                    "image_shape": list(image.shape), "image_dtype": str(image.dtype),
                    "image_min": int(image.min()), "image_max": int(image.max()),
                    "state_shape": list(state.shape), "state_dtype": str(state.dtype),
                    "state18": np.round(state.astype(np.float64), 6).tolist(),
                    "action_shape": list(actions.shape), "action_dtype": str(actions.dtype),
                    "first_action18": np.round(actions[0].astype(np.float64), 6).tolist(),
                    "infer_sec": round(elapsed, 6),
                }
                LOGGER.info("inference=%s", json.dumps(diagnostic, ensure_ascii=False))
                await websocket.send(packer.pack(payload))
            except ConnectionClosedOK:
                LOGGER.info("low-level client closed normally")
                return
            except ConnectionClosedError as exc:
                LOGGER.warning("low-level client disconnected without a clean close: %s", exc)
                return
            except Exception:
                LOGGER.exception("low-level request failed")
                try:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(code=1011, reason="inference failed")
                finally:
                    return

    async def run(self) -> None:
        import websockets.asyncio.server as ws_server

        async def health(connection: Any, request: Any):
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, json.dumps({
                    "ready": True, "server_token": self.server_token,
                }) + "\n")
            return None

        async with ws_server.serve(
            self.handler, self.host, self.port, compression=None, max_size=None,
            process_request=health,
        ) as server:
            LOGGER.info("MemER low policy listening on ws://%s:%s", self.host, self.port)
            await server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--openpi-client-src", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_astribot_subtask")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5902)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--server-token", default="")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.checkpoint / "model_state_dict/full_weights.pt",
        args.checkpoint / "robotwin_astribot_pi05_subtask/norm_stats.json",
        args.rlinf_root / "rlinf/models/embodiment/openpi/__init__.py",
        args.openpi_client_src / "openpi_client/msgpack_numpy.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing low-policy files: {missing}")
    if args.action_horizon != 50:
        raise ValueError("MemER low-level checkpoint requires action_horizon=50")
    training_contract = validate_training_contract(required[1], args.action_horizon)
    if args.dry_run:
        print(json.dumps({
            "ok": True, "checkpoint": str(args.checkpoint), "training_contract": training_contract,
        }, indent=2))
        return
    policy = MockPolicy() if args.mock else OpenPIPolicy(
        args.checkpoint, args.rlinf_root, args.config_name, args.device,
        args.num_steps, args.action_horizon,
    )
    policy.training_contract = training_contract
    server = LowPolicyServer(
        policy, args.host, args.port, args.action_horizon, args.openpi_client_src, args.server_token
    )
    LOGGER.info("training_contract=%s", json.dumps(training_contract, ensure_ascii=False))
    asyncio.run(server.run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
