from __future__ import annotations

import atexit
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

ACTION_ORDER = [
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "left_gripper", "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4",
    "right_arm_5", "right_arm_6", "right_gripper", "torso_yaw", "head_2",
]
DELTA_ACTION_MASK = np.asarray([
    True, True, True, True, True, True, True, False,
    True, True, True, True, True, True, True, False, True, True,
])


def _arg(args: dict[str, Any], name: str, default: Any) -> Any:
    value = args.get(name)
    return default if value is None else value


def _bool(value: Any) -> bool:
    return str(value).lower() not in {"0", "false", "no", "off"} if isinstance(value, str) else bool(value)


def _rgb(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        if not np.all(np.isfinite(image)):
            raise ValueError("image contains non-finite values")
        if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _required_array(value: Any, size: int, name: str, *, exact: bool = True) -> np.ndarray:
    if value is None:
        raise ValueError(f"missing joint state component: {name}")
    source = np.asarray(value, dtype=np.float32).reshape(-1)
    invalid_size = source.size != size if exact else source.size < size
    if invalid_size:
        expectation = str(size) if exact else f">={size}"
        raise ValueError(f"joint state {name} must have {expectation} values, got {source.size}")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"joint state {name} contains non-finite values")
    return source[:size]


def get_head_rgb(observation: dict[str, Any]) -> tuple[str, np.ndarray]:
    cameras = observation.get("observation", {})
    for name in ("camera_head", "head_camera", "camera_front", "cam_head"):
        camera = cameras.get(name)
        if isinstance(camera, dict) and "rgb" in camera:
            return name, _rgb(camera["rgb"])
    raise KeyError(f"MemER requires head RGB; cameras={sorted(cameras)}")


def build_state18(observation: dict[str, Any], torso_index: int = 0, head2_index: int = 1) -> np.ndarray:
    joints = observation.get("joint_action", {})
    if torso_index < 0 or head2_index < 0:
        raise ValueError("torso_index and head2_index must be non-negative")
    left = _required_array(joints.get("left_arm"), 7, "left_arm")
    right = _required_array(joints.get("right_arm"), 7, "right_arm")
    left_gripper = _required_array(joints.get("left_gripper"), 1, "left_gripper")
    right_gripper = _required_array(joints.get("right_gripper"), 1, "right_gripper")
    torso = _required_array(joints.get("torso"), torso_index + 1, "torso", exact=False)
    head = _required_array(joints.get("head"), head2_index + 1, "head", exact=False)
    return np.concatenate([
        left, left_gripper,
        right, right_gripper,
        [torso[torso_index], head[head2_index]],
    ]).astype(np.float32)


class HighClient:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.url = f"http://{host}:{port}"
        self.timeout = timeout
        self.http = requests.Session()
        self.http.trust_env = False
        self.wait_ready()

    def wait_ready(self) -> None:
        deadline = time.monotonic() + self.timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = self.http.get(f"{self.url}/healthz", timeout=3)
                if response.status_code == 200 and response.json().get("ready"):
                    return
            except requests.RequestException as exc:
                last = exc
            time.sleep(2)
        raise TimeoutError(f"high policy not ready: {last}")

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.http.post(f"{self.url}/act", json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"high /act HTTP {response.status_code}: {response.text[:2000]}")
        return response.json()

    def reset(self, session_id: str) -> None:
        response = self.http.post(f"{self.url}/reset", json={"session_id": session_id}, timeout=min(30, self.timeout))
        if response.status_code != 200:
            raise RuntimeError(f"high /reset HTTP {response.status_code}: {response.text[:1000]}")

    def close(self) -> None:
        self.http.close()


class LowClient:
    def __init__(self, host: str, port: int, timeout: float, client_src: str) -> None:
        if client_src and client_src not in sys.path:
            sys.path.insert(0, client_src)
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        self.ws_module = websockets.sync.client
        self.msgpack = msgpack_numpy
        self.uri = f"ws://{host}:{port}"
        self.timeout = timeout
        self.packer = msgpack_numpy.Packer()
        self.ws = None
        self.metadata: dict[str, Any] = {}
        self.connect()

    def connect(self) -> None:
        deadline = time.monotonic() + self.timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.ws = self.ws_module.connect(
                    self.uri, compression=None, max_size=None, open_timeout=min(30, self.timeout),
                    ping_interval=None, ping_timeout=None,
                )
                self.metadata = self.msgpack.unpackb(self.ws.recv())
                return
            except OSError as exc:
                last = exc
                time.sleep(2)
        raise TimeoutError(f"low policy not ready: {last}")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.ws is not None
        self.ws.send(self.packer.pack(payload))
        response = self.ws.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"low policy error:\n{response}")
        return self.msgpack.unpackb(response)

    def reset(self) -> None:
        response = self.request({"__command__": "reset"})
        if not response.get("ok"):
            raise RuntimeError(f"low reset failed: {response}")

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()


class MemERRobotwinPolicy:
    def __init__(
        self, high_host: str, high_port: int, low_host: str, low_port: int,
        request_timeout: float, openpi_client_src: str, action_horizon: int,
        action_dim: int, low_level_execution_horizon: int,
        high_level_replan_interval: int, replan_on_episode_start: bool,
        action_semantics: str, torso_index: int, head2_index: int,
        environment_type: str, worker_id: str, log_request_debug: bool,
        temp_root: str, high_image_width: int = 320, high_image_height: int = 180,
        high_client: HighClient | None = None,
        low_client: LowClient | None = None,
    ) -> None:
        if action_dim != 18 or action_horizon != 50:
            raise ValueError("MemER Astribot requires action_dim=18 and action_horizon=50")
        if low_level_execution_horizon <= 0 or high_level_replan_interval <= 0:
            raise ValueError("execution horizon and replan interval must be positive")
        if low_level_execution_horizon > action_horizon:
            raise ValueError("execution horizon cannot exceed action horizon")
        if action_semantics not in {"absolute", "delta_to_abs"}:
            raise ValueError("action_semantics must be absolute or delta_to_abs")
        if not replan_on_episode_start:
            raise ValueError("MemER requires replan_on_episode_start=true to obtain the initial subtask")
        if high_image_width <= 0 or high_image_height <= 0:
            raise ValueError("high-level image dimensions must be positive")
        self.high = high_client or HighClient(high_host, high_port, request_timeout)
        self.low = low_client or LowClient(low_host, low_port, request_timeout, openpi_client_src)
        self.action_horizon = action_horizon
        self.execution_horizon = low_level_execution_horizon
        self.replan_interval = high_level_replan_interval
        self.replan_on_episode_start = replan_on_episode_start
        self.action_semantics = action_semantics
        self.torso_index = torso_index
        self.head2_index = head2_index
        self.environment_type = environment_type
        self.worker_id = worker_id
        self.log_request_debug = log_request_debug
        self.high_image_width = high_image_width
        self.high_image_height = high_image_height
        self.temp_root = Path(temp_root) / str(os.getuid()) / uuid.uuid4().hex
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.episode_number = 0
        self.session_id = ""
        self.episode_id = ""
        self.current_subtask: str | None = None
        self.last_high_step: int | None = None
        self.pending_frames: dict[int, np.ndarray] = {}
        self.last_request_sec: float | None = None
        self.last_chunk_size = 0
        self.chunk_index = 0
        self.closed = False

    def reset(self) -> None:
        if self.session_id:
            self.high.reset(self.session_id)
        self.low.reset()
        self.episode_number += 1
        self.episode_id = f"{self.run_id}/episode-{self.episode_number:04d}"
        self.session_id = f"{self.environment_type}/{self.worker_id}/{self.episode_id}"
        self.current_subtask = None
        self.last_high_step = None
        self.pending_frames.clear()
        self.last_request_sec = None
        self.last_chunk_size = 0
        self.chunk_index = 0

    def observe(self, env_step: int, observation: dict[str, Any]) -> None:
        _, image = get_head_rgb(observation)
        if image.shape[:2] != (self.high_image_height, self.high_image_width):
            image = np.asarray(Image.fromarray(image).resize(
                (self.high_image_width, self.high_image_height), Image.Resampling.BICUBIC
            ))
        self.pending_frames[int(env_step)] = image

    def _debug_path(self, task_env: Any) -> Path:
        base = Path(task_env.eval_video_path) if getattr(task_env, "eval_video_path", None) else self.temp_root
        return base / "memer_request_debug.jsonl"

    def _write_debug(self, task_env: Any, record: dict[str, Any]) -> None:
        if not self.log_request_debug:
            return
        path = self._debug_path(task_env)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _should_replan(self, env_step: int) -> tuple[bool, str]:
        if self.current_subtask is None:
            return self.replan_on_episode_start, "episode_start"
        if self.last_high_step is None or env_step - self.last_high_step >= self.replan_interval:
            return True, "interval"
        return False, "none"

    def _replan(self, task_env: Any, env_step: int, reason: str) -> bool:
        instruction = str(task_env.get_instruction()).strip()
        frames = [
            {"env_step": step, "image": image.tolist()}
            for step, image in sorted(self.pending_frames.items())
        ]
        payload = {
            "session_id": self.session_id,
            "environment_type": self.environment_type,
            "worker_id": self.worker_id,
            "episode_id": self.episode_id,
            "task": instruction,
            "env_step": env_step,
            "replan_reason": reason,
            "frames": frames,
        }
        response = self.high.act(payload)
        self.pending_frames.clear()
        self.last_high_step = env_step
        self._write_debug(task_env, {"kind": "high", **response})
        if response.get("terminal_parse_failure"):
            print(f"[MemER] terminal high-level parse failure episode={self.episode_id} step={env_step}", flush=True)
            failure = {
                "reason": "memer_high_level_parse_failure",
                "episode_id": self.episode_id,
                "env_step": env_step,
                "parse_error_reason": response.get("parse_error_reason", []),
            }
            marker = getattr(task_env, "mark_eval_failure", None)
            if callable(marker):
                marker(failure)
            else:
                task_env.eval_failed = True
                task_env.eval_failure_reason = failure["reason"]
                task_env.eval_failure_detail = failure
            return False
        self.current_subtask = str(response["new_subtask"])
        print(
            f"[MemER][high] environment_type={self.environment_type} worker_id={self.worker_id} "
            f"episode_id={self.episode_id} env_step={env_step} previous_subtask={response.get('previous_subtask')!r} "
            f"new_subtask={self.current_subtask!r} replan_reason={reason} "
            f"high_level_latency={response.get('high_level_latency')}", flush=True,
        )
        return True

    def get_actions(self, task_env: Any, observation: dict[str, Any]) -> list[np.ndarray]:
        if not self.session_id:
            raise RuntimeError("MemER must be reset before inference")
        env_step = int(task_env.take_action_cnt)
        self.observe(env_step, observation)
        should_replan, reason = self._should_replan(env_step)
        if should_replan and not self._replan(task_env, env_step, reason):
            return []
        if not self.current_subtask:
            raise RuntimeError("no valid high-level subtask is available")
        camera_name, image = get_head_rgb(observation)
        state = build_state18(observation, self.torso_index, self.head2_index)
        request = {"observation/image": image, "observation/state": state, "prompt": self.current_subtask}
        started = time.perf_counter()
        response = self.low.request(request)
        self.last_request_sec = time.perf_counter() - started
        actions = np.asarray(response.get("actions"), dtype=np.float64)
        raw_action_shape = list(actions.shape)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.shape != (self.action_horizon, 18) or not np.all(np.isfinite(actions)):
            raise ValueError(f"expected finite low actions ({self.action_horizon},18), got {actions.shape}")
        if str(response.get("prompt")) != self.current_subtask:
            raise RuntimeError("low server did not echo the high-level subtask")
        if self.action_semantics == "delta_to_abs":
            actions[:, DELTA_ACTION_MASK] += state[DELTA_ACTION_MASK]
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
        actions[:, 15] = np.clip(actions[:, 15], 0.0, 1.0)
        steps_since_replan = env_step - int(self.last_high_step)
        steps_to_replan = self.replan_interval - steps_since_replan
        if steps_to_replan <= 0:
            raise RuntimeError("action request crossed a pending high-level replan boundary")
        actions = actions[:min(self.execution_horizon, steps_to_replan)]
        self.last_chunk_size = len(actions)
        self._write_debug(task_env, {
            "kind": "low", "environment_type": self.environment_type, "worker_id": self.worker_id,
            "episode_id": self.episode_id, "env_step": env_step, "subtask": self.current_subtask,
            "action_shape": list(actions.shape), "low_level_latency": self.last_request_sec,
            "server_timing": response.get("server_timing", {}), "action_order": ACTION_ORDER,
            "camera_name": camera_name, "image_shape": list(image.shape), "image_dtype": str(image.dtype),
            "image_min": int(image.min()), "image_max": int(image.max()),
            "state_shape": list(state.shape), "state_dtype": str(state.dtype),
            "state18": np.round(state.astype(np.float64), 6).tolist(),
            "raw_action_shape": raw_action_shape,
            "first_action18": np.round(actions[0], 6).tolist() if actions.size else [],
            "action_semantics": self.action_semantics,
            "input_contract": {
                "image": "HWC uint8 camera_head",
                "state_order": ACTION_ORDER,
                "state_dim": 18,
                "action_horizon": self.action_horizon,
                "action_dim": 18,
            },
        })
        print(f"[MemER][low] env_step={env_step} prompt={self.current_subtask!r} actions={actions.shape}", flush=True)
        return list(actions)

    def _current_extra(self, task_env: Any, observation: dict[str, Any]) -> tuple[float, float]:
        state = build_state18(observation, self.torso_index, self.head2_index)
        torso, head = float(state[16]), float(state[17])
        torso_value = task_env._get_torso_joint_state_now()
        head_value = task_env._get_head_joint_state_now()
        if torso_value is not None and np.asarray(torso_value).size:
            values = np.asarray(torso_value).reshape(-1)
            torso = float(values[min(self.torso_index, values.size - 1)])
        if head_value is not None and np.asarray(head_value).size:
            values = np.asarray(head_value).reshape(-1)
            head = float(values[min(self.head2_index, values.size - 1)])
        return torso, head

    def to_env_qpos_action(self, action18: np.ndarray, task_env: Any, observation: dict[str, Any]) -> np.ndarray:
        action = np.asarray(action18, dtype=np.float64).reshape(-1)
        if action.shape != (18,) or not np.all(np.isfinite(action)):
            raise ValueError(f"expected finite action18, got {action.shape}")
        torso, head = self._current_extra(task_env, observation)
        return np.concatenate([action[:16], [0.0, action[17] - head, action[16] - torso]])

    def log_chunk_timing(self, start: int, end: int, executed: int, sim_sec: float, success: bool, limit: int) -> None:
        print(
            f"[MemER][chunk_timing] episode_id={self.episode_id} chunk={self.chunk_index} "
            f"start_step={start} end_step={end} executed_steps={executed} chunk_actions={self.last_chunk_size} "
            f"low_request_sec={self.last_request_sec} sim_sec={sim_sec:.3f} success={success} step_limit={limit}",
            flush=True,
        )
        self.chunk_index += 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.session_id:
                self.high.reset(self.session_id)
        finally:
            self.high.close()
            self.low.close()


def get_model(args: dict[str, Any]) -> MemERRobotwinPolicy:
    model = MemERRobotwinPolicy(
        high_host=str(os.environ.get("MEMER_HIGH_HOST", _arg(args, "high_host", "127.0.0.1"))),
        high_port=int(os.environ.get("MEMER_HIGH_PORT", _arg(args, "high_port", 5901))),
        low_host=str(os.environ.get("MEMER_LOW_HOST", _arg(args, "low_host", "127.0.0.1"))),
        low_port=int(os.environ.get("MEMER_LOW_PORT", _arg(args, "low_port", 5902))),
        request_timeout=float(_arg(args, "request_timeout", 300.0)),
        openpi_client_src=str(_arg(args, "openpi_client_src", "")),
        action_horizon=int(_arg(args, "action_horizon", 50)),
        action_dim=int(_arg(args, "action_dim", 18)),
        low_level_execution_horizon=int(_arg(args, "low_level_execution_horizon", 5)),
        high_level_replan_interval=int(_arg(args, "high_level_replan_interval", 5)),
        replan_on_episode_start=_bool(_arg(args, "replan_on_episode_start", True)),
        action_semantics=str(_arg(args, "action_semantics", "absolute")),
        torso_index=int(_arg(args, "torso_index", 0)),
        head2_index=int(_arg(args, "head2_index", 1)),
        environment_type=str(os.environ.get("MEMER_ENVIRONMENT_TYPE", _arg(args, "environment_type", "clean"))),
        worker_id=str(os.environ.get("MEMER_WORKER_ID", _arg(args, "worker_id", f"worker-{os.getpid()}"))),
        log_request_debug=_bool(_arg(args, "log_request_debug", True)),
        temp_root=str(_arg(args, "temp_root", "/tmp/memer_robotwin_eval")),
        high_image_width=int(_arg(args, "high_image_width", 320)),
        high_image_height=int(_arg(args, "high_image_height", 180)),
    )
    atexit.register(model.close)
    return model


def eval(task_env: Any, model: MemERRobotwinPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    actions = model.get_actions(task_env, observation)
    start_step = int(task_env.take_action_cnt)
    started = time.perf_counter()
    executed = 0
    for action in actions:
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success or task_env.eval_done:
            break
        task_env.take_action(model.to_env_qpos_action(action, task_env, observation), action_type="qpos")
        executed += 1
        if task_env.take_action_cnt < task_env.step_lim and not task_env.eval_success and not task_env.eval_done:
            observation = task_env.get_obs()
            model.observe(int(task_env.take_action_cnt), observation)
    model.log_chunk_timing(
        start_step, int(task_env.take_action_cnt), executed, time.perf_counter() - started,
        bool(task_env.eval_success), int(task_env.step_lim),
    )
    return observation


def reset_model(model: MemERRobotwinPolicy) -> None:
    model.reset()
