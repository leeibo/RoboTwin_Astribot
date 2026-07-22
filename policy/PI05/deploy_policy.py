import json
import os
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ACTION_ORDER = [
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6", "left_gripper",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6", "right_gripper",
    "torso_yaw", "head_2",
]
DELTA_ACTION_MASK = np.array([
    True, True, True, True, True, True, True, False,
    True, True, True, True, True, True, True, False,
    True, True,
], dtype=bool)
TORSO_LIMITS = (-1.2, 1.2)


def _arg(usr_args: dict[str, Any], name: str, default: Any) -> Any:
    value = usr_args.get(name, None)
    return default if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_1d(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    if arr.shape[0] >= length:
        return arr[:length].astype(np.float32, copy=False)
    out = np.full(length, default, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def _scalar(value: Any, default: float = 0.0) -> float:
    arr = np.asarray(value if value is not None else [default], dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _round_list(value: Any, digits: int = 6) -> list:
    return np.asarray(value, dtype=np.float64).round(digits).tolist()


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_jsonable(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _format_sec(value: float | None) -> str:
    return "nan" if value is None else f"{value:.3f}"


def _get_camera_rgb(observation: dict[str, Any]) -> tuple[str, np.ndarray]:
    obs = observation.get("observation", {})
    for name in ("camera_head", "head_camera", "camera_front", "cam_head"):
        cam = obs.get(name)
        if isinstance(cam, dict) and "rgb" in cam:
            return name, _as_uint8_rgb(cam["rgb"])
    available = sorted(str(key) for key in obs.keys())
    raise KeyError(f"PI05 requires a head RGB camera; available cameras: {available}")


def build_state18(observation: dict[str, Any], torso_index: int = 0, head2_index: int = 1) -> np.ndarray:
    joint_action = observation.get("joint_action", {})
    left_arm = _as_1d(joint_action.get("left_arm"), 7)
    right_arm = _as_1d(joint_action.get("right_arm"), 7)
    left_gripper = _scalar(joint_action.get("left_gripper"), 1.0)
    right_gripper = _scalar(joint_action.get("right_gripper"), 1.0)
    torso = _as_1d(joint_action.get("torso"), max(1, int(torso_index) + 1), default=0.0)
    head = _as_1d(joint_action.get("head"), max(2, int(head2_index) + 1), default=1.0)
    return np.concatenate([
        left_arm,
        np.array([left_gripper], dtype=np.float32),
        right_arm,
        np.array([right_gripper], dtype=np.float32),
        np.array([torso[int(torso_index)], head[int(head2_index)]], dtype=np.float32),
    ])


class _OpenPIWebsocketClient:
    def __init__(self, host: str, port: int, timeout: float, openpi_client_src: str | None) -> None:
        if openpi_client_src:
            src = str(Path(openpi_client_src).expanduser().resolve())
            if src not in sys.path:
                sys.path.insert(0, src)
        try:
            import websockets.sync.client
            from openpi_client import msgpack_numpy
        except Exception as exc:
            raise RuntimeError(
                "PI05 adapter needs OpenPI websocket client dependencies in the RoboTwin Python environment. "
                "Run policy/PI05/check.sh for details. "
                f"Original import error: {type(exc).__name__}: {exc}"
            ) from exc
        self._websockets_client = websockets.sync.client
        self._msgpack_numpy = msgpack_numpy
        self.uri = f"ws://{host}:{int(port)}"
        self.timeout = float(timeout)
        self.packer = msgpack_numpy.Packer()
        self.ws, self.metadata = self._wait_for_server()

    def _wait_for_server(self):
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)
        start = time.time()
        last_error = None
        while True:
            if time.time() - start > self.timeout:
                raise TimeoutError(f"Failed to connect to PI05 server at {self.uri}: {last_error}")
            try:
                conn = self._websockets_client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    open_timeout=min(30.0, max(1.0, self.timeout)),
                    ping_interval=None,
                    ping_timeout=None,
                )
                metadata = self._msgpack_numpy.unpackb(conn.recv())
                print(f"[PI05] connected to {self.uri}, metadata={metadata}", flush=True)
                return conn, metadata
            except OSError as exc:
                last_error = exc
                print(f"[PI05] waiting for server {self.uri} ...", flush=True)
                time.sleep(2)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.ws.send(self.packer.pack(observation))
        response = self.ws.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"PI05 server error:\n{response}")
        return self._msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


class PI05RobotwinPolicy:
    def __init__(self, host: str, port: int, request_timeout: float = 300.0, max_actions_per_call: int = 5,
                 action_dim: int = 18, action_horizon: int = 50, log_chunk_timing: bool = True,
                 log_request_debug: bool = True, temp_root: str = "/tmp/pi05_robotwin_eval",
                 openpi_client_src: str | None = None, torso_index: int = 0, head2_index: int = 1,
                 action_semantics: str = "absolute") -> None:
        self.client = _OpenPIWebsocketClient(host, port, request_timeout, openpi_client_src)
        self.host = host
        self.port = int(port)
        self.max_actions_per_call = int(max_actions_per_call)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.log_chunk_timing_enabled = bool(log_chunk_timing)
        self.log_request_debug_enabled = bool(log_request_debug)
        self.temp_root = Path(temp_root).expanduser().resolve() / str(os.getuid()) / uuid.uuid4().hex
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.torso_index = int(torso_index)
        self.head2_index = int(head2_index)
        self.action_semantics = str(action_semantics).strip().lower()
        if self.action_semantics not in {"absolute", "delta_to_abs"}:
            raise ValueError("action_semantics must be 'absolute' or 'delta_to_abs'")
        self.episode_idx = 0
        self.step_idx = 0
        self.chunk_idx = 0
        self.last_request_sec = None
        self.last_request_id = None
        self.last_chunk_size = None
        self.last_debug_record = None
        self.action_queue: deque[np.ndarray] = deque()

    def reset(self) -> None:
        self.episode_idx += 1
        self.step_idx = 0
        self.chunk_idx = 0
        self.last_request_sec = None
        self.last_request_id = None
        self.last_chunk_size = None
        self.last_debug_record = None
        self.action_queue.clear()
        self.client.reset()

    def _debug_base(self, TASK_ENV) -> Path | None:
        eval_video_path = getattr(TASK_ENV, "eval_video_path", None)
        return Path(eval_video_path) if eval_video_path else None

    def _save_request_image(self, camera_name: str, image: np.ndarray, debug_base: Path | None) -> Path | None:
        if not self.log_request_debug_enabled:
            return None
        image_root = (debug_base if debug_base is not None else self.temp_root) / "pi05_request_images"
        image_dir = image_root / f"episode_{self.episode_idx:04d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{camera_name}_{self.step_idx:06d}.png"
        Image.fromarray(image).save(image_path)
        return image_path

    def _debug_jsonl_path(self, debug_base: Path | None) -> Path:
        path = (debug_base if debug_base is not None else self.temp_root) / "pi05_request_debug.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_debug(self, TASK_ENV) -> None:
        if not self.log_request_debug_enabled or self.last_debug_record is None:
            return
        path = self._debug_jsonl_path(self._debug_base(TASK_ENV))
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.last_debug_record, ensure_ascii=False) + "\n")
        print(f"[PI05][debug] jsonl={path}", flush=True)

    def _request_actions(self, TASK_ENV, observation: dict[str, Any]) -> np.ndarray:
        instruction = str(TASK_ENV.get_instruction())
        camera_name, image = _get_camera_rgb(observation)
        state18 = build_state18(observation, self.torso_index, self.head2_index)
        debug_base = self._debug_base(TASK_ENV)
        image_path = self._save_request_image(camera_name, image, debug_base)
        request_id = f"pi05_ep{self.episode_idx:04d}_step{self.step_idx:06d}"
        request = {"observation/image": image, "observation/state": state18, "prompt": instruction}
        print(
            f"[PI05] request start id={request_id} ep={self.episode_idx} step={self.step_idx} "
            f"camera={camera_name} timeout={self.client.timeout:.1f}s",
            flush=True,
        )
        start = time.perf_counter()
        try:
            response = self.client.infer(request)
        except Exception as exc:
            request_sec = time.perf_counter() - start
            print(
                f"[PI05] request failed id={request_id} after {request_sec:.3f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        request_sec = time.perf_counter() - start
        if "actions" not in response:
            raise KeyError(f"PI05 response has no 'actions': {response}")
        raw_actions = np.asarray(response["actions"], dtype=np.float64).copy()
        raw_action_shape = list(raw_actions.shape)
        if raw_actions.ndim == 3:
            raw_actions = raw_actions[0]
        if raw_actions.ndim != 2 or raw_actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected PI05 action chunk shape (T, {self.action_dim}), got {raw_actions.shape}")
        full_raw_torso_targets = raw_actions[:, 16].copy()
        full_raw_head2_targets = raw_actions[:, 17].copy()
        if raw_actions.shape[0] > self.action_horizon:
            raw_actions = raw_actions[: self.action_horizon]
        if self.max_actions_per_call > 0:
            raw_actions = raw_actions[: self.max_actions_per_call]
        actions = raw_actions.copy()
        if self.action_semantics == "delta_to_abs":
            actions[:, DELTA_ACTION_MASK] += state18[DELTA_ACTION_MASK]
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
        actions[:, 15] = np.clip(actions[:, 15], 0.0, 1.0)
        torso_real_state, head2_real_state = self._current_torso_head2(TASK_ENV, observation)
        first_env_action = self.to_env_qpos_action(actions[0], TASK_ENV, observation) if actions.size else []
        torso_target = actions[:, 16] if actions.size else np.array([], dtype=np.float64)
        torso_state = float(state18[16])
        torso_outside_limit = (torso_target < TORSO_LIMITS[0]) | (torso_target > TORSO_LIMITS[1])
        torso_push_lower = (torso_state <= TORSO_LIMITS[0] + 0.01) & (torso_target < TORSO_LIMITS[0])
        torso_push_upper = (torso_state >= TORSO_LIMITS[1] - 0.01) & (torso_target > TORSO_LIMITS[1])
        self.last_request_sec = request_sec
        self.last_request_id = request_id
        self.last_chunk_size = int(actions.shape[0])
        self.last_debug_record = {
            "episode": self.episode_idx,
            "step": self.step_idx,
            "chunk": self.chunk_idx,
            "request_id": request_id,
            "task": getattr(TASK_ENV, "task_name", None),
            "instruction": instruction,
            "camera_name": camera_name,
            "image_path": str(image_path) if image_path is not None else None,
            "state18": _round_list(state18),
            "server_host": self.host,
            "server_port": self.port,
            "request_sec": round(float(request_sec), 6),
            "raw_action_shape": raw_action_shape,
            "cached_action_shape": list(actions.shape),
            "action_semantics": self.action_semantics,
            "first_raw_action18": _round_list(raw_actions[0]) if raw_actions.size else [],
            "first_action18": _round_list(actions[0]) if actions.size else [],
            "first_env_action19": _round_list(first_env_action) if len(first_env_action) else [],
            "torso_state": round(torso_state, 6),
            "head2_state": round(float(state18[17]), 6),
            "torso_real_state": round(torso_real_state, 6),
            "head2_real_state": round(head2_real_state, 6),
            "torso_targets": _round_list(torso_target),
            "head2_targets": _round_list(actions[:, 17]) if actions.size else [],
            "full_raw_torso_targets": _round_list(full_raw_torso_targets),
            "full_raw_head2_targets": _round_list(full_raw_head2_targets),
            "torso_target_outside_limit": torso_outside_limit.astype(bool).tolist(),
            "torso_target_outside_limit_ratio": round(float(np.mean(torso_outside_limit)), 6) if torso_target.size else None,
            "torso_at_limit_push_outside": bool(np.any(torso_push_lower | torso_push_upper)) if torso_target.size else False,
            "action_order": ACTION_ORDER,
            "server_timing": _jsonable(response.get("server_timing", {})),
        }
        print(f"[PI05] ep={self.episode_idx} step={self.step_idx} camera={camera_name} chunk={actions.shape[0]} request_sec={request_sec:.3f}", flush=True)
        self.step_idx += 1
        return actions

    def get_actions(self, TASK_ENV, observation: dict[str, Any]) -> list[np.ndarray]:
        if not self.action_queue:
            actions = self._request_actions(TASK_ENV, observation)
            self.action_queue.extend(actions)
            self._write_debug(TASK_ENV)
        out = []
        while self.action_queue and len(out) < self.max_actions_per_call:
            out.append(self.action_queue.popleft())
        return out

    def _current_torso_head2(self, TASK_ENV, observation: dict[str, Any]) -> tuple[float, float]:
        state18 = build_state18(observation, self.torso_index, self.head2_index)
        torso_now = float(state18[16])
        head2_now = float(state18[17])

        real_torso = TASK_ENV._get_torso_joint_state_now()
        real_head = TASK_ENV._get_head_joint_state_now()
        if real_torso is not None:
            real_torso = np.asarray(real_torso, dtype=np.float64).reshape(-1)
            if real_torso.size:
                torso_idx = min(max(self.torso_index, 0), real_torso.size - 1)
                torso_now = float(real_torso[torso_idx])
        if real_head is not None:
            real_head = np.asarray(real_head, dtype=np.float64).reshape(-1)
            if real_head.size:
                head_idx = min(max(self.head2_index, 0), real_head.size - 1)
                head2_now = float(real_head[head_idx])
        return torso_now, head2_now

    def to_env_qpos_action(self, action18: np.ndarray, TASK_ENV, observation: dict[str, Any]) -> np.ndarray:
        action18 = np.asarray(action18, dtype=np.float64).reshape(-1)
        if action18.shape[0] != 18:
            raise ValueError(f"Expected PI05 action dim 18, got {action18.shape[0]}")
        torso_now, head2_now = self._current_torso_head2(TASK_ENV, observation)
        torso_delta = action18[16] - torso_now
        head2_delta = action18[17] - head2_now
        extra_delta = np.array([0.0, head2_delta, torso_delta], dtype=np.float64)
        return np.concatenate([action18[:16], extra_delta], axis=0)

    def log_chunk_timing(self, start_step: int, end_step: int, executed_steps: int, sim_sec: float, success: bool, step_limit: int) -> None:
        if self.log_chunk_timing_enabled:
            total_sec = sim_sec + self.last_request_sec if self.last_request_sec is not None else None
            print(
                f"[PI05][chunk_timing] ep={self.episode_idx} chunk={self.chunk_idx} request_id={self.last_request_id} "
                f"start_step={start_step} end_step={end_step} executed_steps={executed_steps} chunk_actions={self.last_chunk_size} "
                f"request_sec={_format_sec(self.last_request_sec)} sim_sec={sim_sec:.3f} total_sec={_format_sec(total_sec)} "
                f"success={bool(success)} step_limit={step_limit}",
                flush=True,
            )
        self.chunk_idx += 1

    def close(self) -> None:
        self.client.close()


def get_model(usr_args):
    host = str(os.environ.get("PI05_HOST", _arg(usr_args, "host", "127.0.0.1")))
    port = int(os.environ.get("PI05_PORT", _arg(usr_args, "port", 5702)))
    openpi_client_src = str(os.environ.get("OPENPI_CLIENT_SRC", _arg(usr_args, "openpi_client_src", ""))) or None
    return PI05RobotwinPolicy(
        host=host,
        port=port,
        request_timeout=float(_arg(usr_args, "request_timeout", 300.0)),
        max_actions_per_call=int(_arg(usr_args, "max_actions_per_call", 5)),
        action_dim=int(_arg(usr_args, "action_dim", 18)),
        action_horizon=int(_arg(usr_args, "action_horizon", 50)),
        log_chunk_timing=_as_bool(_arg(usr_args, "log_chunk_timing", True)),
        log_request_debug=_as_bool(_arg(usr_args, "log_request_debug", True)),
        temp_root=str(_arg(usr_args, "temp_root", "/tmp/pi05_robotwin_eval")),
        openpi_client_src=openpi_client_src,
        torso_index=int(_arg(usr_args, "torso_index", 0)),
        head2_index=int(_arg(usr_args, "head2_index", 1)),
        action_semantics=str(os.environ.get("PI05_ACTION_SEMANTICS", _arg(usr_args, "action_semantics", "absolute"))),
    )


def eval(TASK_ENV, model, observation):
    actions18 = model.get_actions(TASK_ENV, observation)
    start_step = int(TASK_ENV.take_action_cnt)
    sim_start = time.perf_counter()
    executed_steps = 0
    for action18 in actions18:
        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success or TASK_ENV.eval_done:
            break
        env_action = model.to_env_qpos_action(action18, TASK_ENV, observation)
        TASK_ENV.take_action(env_action, action_type="qpos")
        executed_steps += 1
        if TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_success and not TASK_ENV.eval_done:
            observation = TASK_ENV.get_obs()
    sim_sec = time.perf_counter() - sim_start
    model.log_chunk_timing(start_step, int(TASK_ENV.take_action_cnt), executed_steps, sim_sec, bool(TASK_ENV.eval_success), int(TASK_ENV.step_lim))
    return observation


def reset_model(model):
    model.reset()
