from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

ACTION_ORDER = [
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "left_gripper",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5",
    "right_arm_6", "right_gripper", "torso_yaw", "head_2",
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


def _as_uint8_rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if not np.all(np.isfinite(arr)):
            raise ValueError("RGB image contains non-finite values")
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_1d(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    if arr.shape[0] >= length:
        out = arr[:length].astype(np.float32, copy=False)
    else:
        out = np.full(length, default, dtype=np.float32)
        out[: arr.shape[0]] = arr
    if not np.all(np.isfinite(out)):
        raise ValueError("Joint state contains non-finite values")
    return out


def _scalar(value: Any, default: float = 0.0) -> float:
    arr = np.asarray(value if value is not None else [default], dtype=np.float32).reshape(-1)
    result = float(arr[0]) if arr.size else float(default)
    if not np.isfinite(result):
        raise ValueError("Joint state contains a non-finite scalar")
    return result


def _round_list(value: Any, digits: int = 6) -> list[Any]:
    return np.asarray(value, dtype=np.float64).round(digits).tolist()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _format_sec(value: float | None) -> str:
    return "nan" if value is None else f"{value:.3f}"


def _get_camera_rgb(observation: dict[str, Any]) -> tuple[str, np.ndarray]:
    cameras = observation.get("observation", {})
    for name in ("camera_head", "head_camera", "camera_front", "cam_head"):
        camera = cameras.get(name)
        if isinstance(camera, dict) and "rgb" in camera:
            return name, _as_uint8_rgb(camera["rgb"])
    available = sorted(str(key) for key in cameras)
    raise KeyError(f"HIF-VLA requires a head RGB camera; available cameras: {available}")


def build_state18(observation: dict[str, Any], torso_index: int = 0, head2_index: int = 1) -> np.ndarray:
    joint_action = observation.get("joint_action", {})
    left_arm = _as_1d(joint_action.get("left_arm"), 7)
    right_arm = _as_1d(joint_action.get("right_arm"), 7)
    left_gripper = _scalar(joint_action.get("left_gripper"), 1.0)
    right_gripper = _scalar(joint_action.get("right_gripper"), 1.0)
    torso = _as_1d(joint_action.get("torso"), max(1, torso_index + 1), default=0.0)
    head = _as_1d(joint_action.get("head"), max(2, head2_index + 1), default=1.0)
    return np.concatenate([
        left_arm,
        np.asarray([left_gripper], dtype=np.float32),
        right_arm,
        np.asarray([right_gripper], dtype=np.float32),
        np.asarray([torso[torso_index], head[head2_index]], dtype=np.float32),
    ])


class HIFVLAHttpClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be non-negative")
        self.base_url = f"http://{host}:{int(port)}"
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.session = requests.Session()
        self.session.trust_env = False
        self.metadata = self._wait_for_server()

    def _wait_for_server(self) -> dict[str, Any]:
        start = time.monotonic()
        last_error: str | None = None
        url = f"{self.base_url}/healthz"
        while time.monotonic() - start < self.timeout:
            try:
                response = self.session.get(url, timeout=min(5.0, self.timeout))
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("ready") is True:
                        print(f"[HIF-VLA] connected to {url}, metadata={payload}", flush=True)
                        return payload
                    last_error = f"server not ready: {payload}"
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
        raise TimeoutError(f"Failed to connect to HIF-VLA server at {url}: {last_error}")

    def _post_with_retry(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            try:
                return self.session.post(url, json=payload, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_backoff * (2 ** attempt)
                print(
                    f"[HIF-VLA] transient HTTP failure path={path} "
                    f"retry={attempt + 1}/{self.max_retries} delay={delay:.1f}s: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if delay > 0:
                    time.sleep(delay)
        raise AssertionError("unreachable")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._post_with_retry("/act", payload, self.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"HIF-VLA /act failed with HTTP {response.status_code}: {response.text[:2000]}")
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(f"HIF-VLA /act response must be an object, got {type(data).__name__}")
        return data

    def reset(self, episode_id: str) -> None:
        response = self._post_with_retry(
            "/reset",
            {"episode_id": episode_id},
            min(30.0, self.timeout),
        )
        if response.status_code != 200:
            raise RuntimeError(f"HIF-VLA /reset failed with HTTP {response.status_code}: {response.text[:1000]}")

    def close(self) -> None:
        self.session.close()


class HIFVLARobotwinPolicy:
    def __init__(
        self,
        host: str,
        port: int,
        request_timeout: float = 300.0,
        max_actions_per_call: int = 8,
        action_dim: int = 18,
        action_horizon: int = 8,
        log_chunk_timing: bool = True,
        log_request_debug: bool = True,
        temp_root: str = "/tmp/hifvla_robotwin_eval",
        torso_index: int = 0,
        head2_index: int = 1,
        action_semantics: str = "absolute",
        request_retries: int = 2,
        retry_backoff: float = 1.0,
        client: HIFVLAHttpClient | None = None,
    ) -> None:
        if action_dim != 18:
            raise ValueError(f"HIF-VLA Astribot requires action_dim=18, got {action_dim}")
        if action_horizon <= 0 or max_actions_per_call <= 0:
            raise ValueError("action_horizon and max_actions_per_call must be positive")
        if max_actions_per_call > action_horizon:
            raise ValueError("max_actions_per_call cannot exceed action_horizon")
        self.host = host
        self.port = int(port)
        self.client = (
            client
            if client is not None
            else HIFVLAHttpClient(
                host,
                port,
                request_timeout,
                max_retries=request_retries,
                retry_backoff=retry_backoff,
            )
        )
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
        self.run_id = uuid.uuid4().hex
        self.episode_idx = 0
        self.episode_id = f"{self.run_id}/session"
        self.step_idx = 0
        self.chunk_idx = 0
        self.last_request_sec: float | None = None
        self.last_request_id: str | None = None
        self.last_chunk_size: int | None = None
        self.last_debug_record: dict[str, Any] | None = None
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
        self.client.reset(self.episode_id)

    def _debug_base(self, task_env: Any) -> Path | None:
        eval_video_path = getattr(task_env, "eval_video_path", None)
        return Path(eval_video_path) if eval_video_path else None

    def _save_request_image(
        self, camera_name: str, image: np.ndarray, debug_base: Path | None
    ) -> Path | None:
        if not self.log_request_debug_enabled:
            return None
        root = (debug_base if debug_base is not None else self.temp_root) / "hifvla_request_images"
        image_dir = root / f"episode_{self.episode_idx:04d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{camera_name}_{self.step_idx:06d}.png"
        Image.fromarray(image).save(image_path)
        return image_path

    def _write_debug(self, task_env: Any) -> None:
        if not self.log_request_debug_enabled or self.last_debug_record is None:
            return
        base = self._debug_base(task_env) or self.temp_root
        path = base / "hifvla_request_debug.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.last_debug_record, ensure_ascii=False) + "\n")
        print(f"[HIF-VLA][debug] jsonl={path}", flush=True)

    def _request_actions(self, task_env: Any, observation: dict[str, Any]) -> np.ndarray:
        if not self.episode_id:
            raise RuntimeError("HIF-VLA policy must be reset before the first inference request")
        instruction = str(task_env.get_instruction()).strip()
        if not instruction:
            raise ValueError("Task instruction is empty")
        camera_name, image = _get_camera_rgb(observation)
        state18 = build_state18(observation, self.torso_index, self.head2_index)
        image_path = self._save_request_image(camera_name, image, self._debug_base(task_env))
        request_id = f"hifvla-ep{self.episode_idx:04d}-step{self.step_idx:06d}"
        request = {
            "episode_id": self.episode_id,
            "request_id": request_id,
            "image": image.tolist(),
            "state": state18.tolist(),
            "instruction": instruction,
        }
        print(
            f"[HIF-VLA] request start id={request_id} ep={self.episode_idx} "
            f"step={self.step_idx} camera={camera_name}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            response = self.client.infer(request)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"[HIF-VLA] request failed id={request_id} after {elapsed:.3f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        request_sec = time.perf_counter() - started
        if "actions" not in response:
            raise KeyError(f"HIF-VLA response has no actions: {response}")
        raw_actions = np.asarray(response["actions"], dtype=np.float64)
        raw_shape = list(raw_actions.shape)
        if raw_actions.ndim == 3 and raw_actions.shape[0] == 1:
            raw_actions = raw_actions[0]
        if raw_actions.ndim != 2 or raw_actions.shape[1] != self.action_dim or raw_actions.shape[0] == 0:
            raise ValueError(f"Expected non-empty HIF-VLA action shape (T, {self.action_dim}), got {raw_actions.shape}")
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError("HIF-VLA action response contains non-finite values")
        full_torso_targets = raw_actions[:, 16].copy()
        full_head_targets = raw_actions[:, 17].copy()
        raw_actions = raw_actions[: self.action_horizon]
        actions = raw_actions.copy()
        if self.action_semantics == "delta_to_abs":
            actions[:, DELTA_ACTION_MASK] += state18[DELTA_ACTION_MASK]
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
        actions[:, 15] = np.clip(actions[:, 15], 0.0, 1.0)
        actions = actions[: self.max_actions_per_call]

        torso_real, head_real = self._current_torso_head2(task_env, observation)
        first_env_action = self.to_env_qpos_action(actions[0], task_env, observation)
        torso_targets = actions[:, 16]
        outside = (torso_targets < TORSO_LIMITS[0]) | (torso_targets > TORSO_LIMITS[1])
        self.last_request_sec = request_sec
        self.last_request_id = request_id
        self.last_chunk_size = int(actions.shape[0])
        self.last_debug_record = {
            "episode": self.episode_idx,
            "episode_id": self.episode_id,
            "step": self.step_idx,
            "chunk": self.chunk_idx,
            "request_id": request_id,
            "task": getattr(task_env, "task_name", None),
            "instruction": instruction,
            "camera_name": camera_name,
            "image_path": str(image_path) if image_path is not None else None,
            "state18": _round_list(state18),
            "server_host": self.host,
            "server_port": self.port,
            "request_sec": round(request_sec, 6),
            "raw_action_shape": raw_shape,
            "cached_action_shape": list(actions.shape),
            "action_semantics": self.action_semantics,
            "first_raw_action18": _round_list(raw_actions[0]),
            "first_action18": _round_list(actions[0]),
            "first_env_action19": _round_list(first_env_action),
            "torso_real_state": round(torso_real, 6),
            "head2_real_state": round(head_real, 6),
            "torso_targets": _round_list(torso_targets),
            "head2_targets": _round_list(actions[:, 17]),
            "full_raw_torso_targets": _round_list(full_torso_targets),
            "full_raw_head2_targets": _round_list(full_head_targets),
            "torso_target_outside_limit": outside.tolist(),
            "action_order": ACTION_ORDER,
            "server_timing": _jsonable(response.get("server_timing", {})),
        }
        print(
            f"[HIF-VLA] request done id={request_id} chunk={actions.shape[0]} "
            f"request_sec={request_sec:.3f}",
            flush=True,
        )
        self.step_idx += 1
        return actions

    def get_actions(self, task_env: Any, observation: dict[str, Any]) -> list[np.ndarray]:
        if not self.action_queue:
            self.action_queue.extend(self._request_actions(task_env, observation))
            self._write_debug(task_env)
        actions: list[np.ndarray] = []
        while self.action_queue and len(actions) < self.max_actions_per_call:
            actions.append(self.action_queue.popleft())
        return actions

    def _current_torso_head2(
        self, task_env: Any, observation: dict[str, Any]
    ) -> tuple[float, float]:
        state18 = build_state18(observation, self.torso_index, self.head2_index)
        torso_now = float(state18[16])
        head_now = float(state18[17])
        get_torso = getattr(task_env, "_get_torso_joint_state_now", None)
        get_head = getattr(task_env, "_get_head_joint_state_now", None)
        real_torso = get_torso() if callable(get_torso) else None
        real_head = get_head() if callable(get_head) else None
        if real_torso is not None:
            values = np.asarray(real_torso, dtype=np.float64).reshape(-1)
            if values.size:
                torso_now = float(values[min(max(self.torso_index, 0), values.size - 1)])
        if real_head is not None:
            values = np.asarray(real_head, dtype=np.float64).reshape(-1)
            if values.size:
                head_now = float(values[min(max(self.head2_index, 0), values.size - 1)])
        return torso_now, head_now

    def to_env_qpos_action(
        self, action18: np.ndarray, task_env: Any, observation: dict[str, Any]
    ) -> np.ndarray:
        action = np.asarray(action18, dtype=np.float64).reshape(-1)
        if action.shape != (18,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected finite HIF-VLA action dim 18, got {action.shape}")
        torso_now, head_now = self._current_torso_head2(task_env, observation)
        extra_delta = np.asarray([0.0, action[17] - head_now, action[16] - torso_now])
        return np.concatenate([action[:16], extra_delta])

    def log_chunk_timing(
        self,
        start_step: int,
        end_step: int,
        executed_steps: int,
        sim_sec: float,
        success: bool,
        step_limit: int,
    ) -> None:
        if self.log_chunk_timing_enabled:
            total_sec = sim_sec + self.last_request_sec if self.last_request_sec is not None else None
            print(
                f"[HIF-VLA][chunk_timing] ep={self.episode_idx} chunk={self.chunk_idx} "
                f"request_id={self.last_request_id} start_step={start_step} end_step={end_step} "
                f"executed_steps={executed_steps} chunk_actions={self.last_chunk_size} "
                f"request_sec={_format_sec(self.last_request_sec)} sim_sec={sim_sec:.3f} "
                f"total_sec={_format_sec(total_sec)} success={success} step_limit={step_limit}",
                flush=True,
            )
        self.chunk_idx += 1

    def close(self) -> None:
        self.client.close()


def get_model(usr_args: dict[str, Any]) -> HIFVLARobotwinPolicy:
    host = str(os.environ.get("HIFVLA_HOST", _arg(usr_args, "host", "127.0.0.1")))
    port = int(os.environ.get("HIFVLA_PORT", _arg(usr_args, "port", 5802)))
    return HIFVLARobotwinPolicy(
        host=host,
        port=port,
        request_timeout=float(_arg(usr_args, "request_timeout", 300.0)),
        max_actions_per_call=int(_arg(usr_args, "max_actions_per_call", 8)),
        action_dim=int(_arg(usr_args, "action_dim", 18)),
        action_horizon=int(_arg(usr_args, "action_horizon", 8)),
        log_chunk_timing=_as_bool(_arg(usr_args, "log_chunk_timing", True)),
        log_request_debug=_as_bool(_arg(usr_args, "log_request_debug", True)),
        temp_root=str(_arg(usr_args, "temp_root", "/tmp/hifvla_robotwin_eval")),
        torso_index=int(_arg(usr_args, "torso_index", 0)),
        head2_index=int(_arg(usr_args, "head2_index", 1)),
        action_semantics=str(
            os.environ.get("HIFVLA_ACTION_SEMANTICS", _arg(usr_args, "action_semantics", "absolute"))
        ),
        request_retries=int(
            os.environ.get("HIFVLA_REQUEST_RETRIES", _arg(usr_args, "request_retries", 2))
        ),
        retry_backoff=float(
            os.environ.get("HIFVLA_RETRY_BACKOFF", _arg(usr_args, "retry_backoff", 1.0))
        ),
    )


def eval(task_env: Any, model: HIFVLARobotwinPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    actions18 = model.get_actions(task_env, observation)
    start_step = int(task_env.take_action_cnt)
    started = time.perf_counter()
    executed_steps = 0
    for action18 in actions18:
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success or task_env.eval_done:
            break
        env_action = model.to_env_qpos_action(action18, task_env, observation)
        task_env.take_action(env_action, action_type="qpos")
        executed_steps += 1
        if task_env.take_action_cnt < task_env.step_lim and not task_env.eval_success and not task_env.eval_done:
            observation = task_env.get_obs()
    model.log_chunk_timing(
        start_step,
        int(task_env.take_action_cnt),
        executed_steps,
        time.perf_counter() - started,
        bool(task_env.eval_success),
        int(task_env.step_lim),
    )
    return observation


def reset_model(model: HIFVLARobotwinPolicy) -> None:
    model.reset()
