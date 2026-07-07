import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from PIL import Image
import websockets.sync.client


ACTION_ORDER = [
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
    "left_gripper",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
    "right_gripper",
    "torso4",
    "head2",
]


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    ndarray_key = b"__ndarray__" if b"__ndarray__" in obj else "__ndarray__"
    npgeneric_key = b"__npgeneric__" if b"__npgeneric__" in obj else "__npgeneric__"
    if ndarray_key in obj:
        data = obj[b"data"] if b"data" in obj else obj["data"]
        dtype = obj[b"dtype"] if b"dtype" in obj else obj["dtype"]
        shape = obj[b"shape"] if b"shape" in obj else obj["shape"]
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=shape)
    if npgeneric_key in obj:
        data = obj[b"data"] if b"data" in obj else obj["data"]
        dtype = obj[b"dtype"] if b"dtype" in obj else obj["dtype"]
        return np.dtype(dtype).type(data)
    return obj


def _get(mapping: dict[str, Any], key: str, default=None):
    if key in mapping:
        return mapping[key]
    bkey = key.encode("utf-8")
    if bkey in mapping:
        return mapping[bkey]
    return default


def _format_sec(value: float | None) -> str:
    return "nan" if value is None else f"{value:.3f}"


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


class _WebsocketClient:
    def __init__(self, host: str, port: int, api_key: str | None = None, timeout: float = 300.0) -> None:
        self.uri = f"ws://{host}:{int(port)}"
        self.api_key = api_key
        self.timeout = float(timeout)
        self.packer = msgpack.Packer(default=_pack_array)
        self.ws, self.metadata = self._wait_for_server()

    def _wait_for_server(self):
        start = time.time()
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)
        while True:
            if time.time() - start > self.timeout:
                raise TimeoutError(f"Failed to connect to QwenOFT18D server at {self.uri}")
            try:
                headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
                conn = websockets.sync.client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=150,
                    ping_interval=20,
                    ping_timeout=20,
                )
                metadata = msgpack.unpackb(conn.recv(), object_hook=_unpack_array)
                print(f"[QwenOFT18D] connected to {self.uri}, metadata={metadata}")
                return conn, metadata
            except OSError:
                print(f"[QwenOFT18D] waiting for server {self.uri} ...")
                time.sleep(2)

    def predict_action(self, request: dict[str, Any]) -> dict[str, Any]:
        self.ws.send(self.packer.pack(request))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"QwenOFT18D server error:\n{response}")
        return msgpack.unpackb(response, object_hook=_unpack_array)

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _as_1d(value, length: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    if arr.shape[0] >= length:
        return arr[:length].astype(np.float32, copy=False)
    out = np.full(length, default, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def _scalar(value, default: float = 0.0) -> float:
    arr = np.asarray(value if value is not None else [default], dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _build_qwengroot18d_state(observation: dict[str, Any], torso_index: int = 0, head2_index: int = 1) -> np.ndarray:
    joint_action = observation.get("joint_action", {})
    left_arm = _as_1d(joint_action.get("left_arm"), 7)
    right_arm = _as_1d(joint_action.get("right_arm"), 7)
    left_gripper = _scalar(joint_action.get("left_gripper"), 1.0)
    right_gripper = _scalar(joint_action.get("right_gripper"), 1.0)
    torso = _as_1d(joint_action.get("torso"), max(1, int(torso_index) + 1), default=0.0)
    head = _as_1d(joint_action.get("head"), max(2, int(head2_index) + 1), default=1.0)
    return np.concatenate(
        [
            left_arm,
            np.array([left_gripper], dtype=np.float32),
            right_arm,
            np.array([right_gripper], dtype=np.float32),
            np.array([torso[int(torso_index)], head[int(head2_index)]], dtype=np.float32),
        ]
    )


class QwenOFT18DRobotwinPolicy:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5702,
        request_timeout: float = 300.0,
        max_actions_per_call: int = 16,
        send_state: bool = True,
        swap_rgb_channels: bool = True,
        log_chunk_timing: bool = True,
        log_request_debug: bool = True,
        temp_root: str = "/tmp/qwengroot18d_robotwin",
        torso_index: int = 0,
        head2_index: int = 1,
    ) -> None:
        self.client = _WebsocketClient(host=host, port=port, timeout=request_timeout)
        self.max_actions_per_call = int(max_actions_per_call)
        self.send_state = bool(send_state)
        self.swap_rgb_channels = bool(swap_rgb_channels)
        self.log_chunk_timing_enabled = bool(log_chunk_timing)
        self.log_request_debug_enabled = bool(log_request_debug)
        self.temp_root = Path(temp_root).expanduser().resolve() / str(os.getuid()) / uuid.uuid4().hex
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.torso_index = int(torso_index)
        self.head2_index = int(head2_index)
        self.episode_idx = 0
        self.step_idx = 0
        self.chunk_idx = 0
        self.last_request_id = None
        self.last_request_sec = None
        self.last_chunk_size = None
        self.last_debug_record = None
        self.last_debug_path = None

    def reset(self) -> None:
        self.episode_idx += 1
        self.step_idx = 0
        self.chunk_idx = 0
        self.last_request_id = None
        self.last_request_sec = None
        self.last_chunk_size = None
        self.last_debug_record = None
        self.last_debug_path = None

    def _save_request_image(self, camera_name: str, image: np.ndarray, debug_base: Path | None) -> Path | None:
        if not self.log_request_debug_enabled:
            return None
        if debug_base is None:
            image_dir = self.temp_root / "qwengroot18d_debug_images"
        else:
            image_dir = Path(debug_base) / "qwengroot18d_debug_images"
        image_dir = image_dir / f"episode_{self.episode_idx:04d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{camera_name}_{self.step_idx:06d}.png"
        Image.fromarray(image).save(image_path)
        return image_path

    def _debug_jsonl_path(self, debug_base: Path | None) -> Path:
        if debug_base is None:
            path = self.temp_root / "qwengroot18d_debug.jsonl"
        else:
            path = Path(debug_base) / "qwengroot18d_debug.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_request_debug(self, TASK_ENV) -> None:
        if not self.log_request_debug_enabled or self.last_debug_record is None:
            return
        eval_video_path = getattr(TASK_ENV, "eval_video_path", None)
        debug_base = Path(eval_video_path) if eval_video_path is not None else None
        debug_path = self._debug_jsonl_path(debug_base)
        with debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.last_debug_record, ensure_ascii=False) + "\n")
        self.last_debug_path = str(debug_path)
        print(
            f"[QwenOFT18D][debug] ep={self.last_debug_record['episode']} "
            f"step={self.last_debug_record['step']} jsonl={debug_path} "
            f"images={self.last_debug_record['request']['image_paths']}",
            flush=True,
        )

    def _request_actions(
        self,
        instruction: str,
        observation: dict[str, Any],
        debug_base: Path | None = None,
    ) -> np.ndarray:
        obs = observation["observation"]
        if "camera_head" not in obs or "rgb" not in obs["camera_head"]:
            available = sorted(str(key) for key in obs.keys())
            raise KeyError(f"QwenOFT18D requires observation['camera_head']['rgb']; available cameras: {available}")
        camera_name = "camera_head"
        request_step = self.step_idx
        head_rgb = _as_uint8_rgb(obs["camera_head"]["rgb"])
        if self.swap_rgb_channels:
            # Match checkpoints trained on LeRobot videos with swapped R/B channels.
            head_rgb = np.ascontiguousarray(head_rgb[..., ::-1])
        image_path = self._save_request_image(camera_name, head_rgb, debug_base)
        example = {
            "id": f"qwengroot18d_ep{self.episode_idx:04d}_step{self.step_idx:06d}",
            "image": [head_rgb],
            "lang": str(instruction),
        }
        state = None
        if self.send_state:
            state = _build_qwengroot18d_state(
                observation,
                torso_index=self.torso_index,
                head2_index=self.head2_index,
            ).reshape(1, -1)
            example["state"] = state

        request = {
            "type": "infer",
            "request_id": example["id"],
            "examples": [example],
            "state_normalized": False,
        }
        request_start = time.perf_counter()
        response = self.client.predict_action(request)
        request_sec = time.perf_counter() - request_start
        if not _get(response, "ok", False):
            raise RuntimeError(f"QwenOFT18D inference failed: {response}")
        data = _get(response, "data", {})
        actions = _get(data, "actions", None)
        if actions is None:
            raise KeyError(f"QwenOFT18D response has no unnormalized 'actions': {response}")
        actions = np.asarray(actions, dtype=np.float64)
        raw_action_shape = list(actions.shape)
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != 18:
            raise ValueError(f"Expected action chunk shape (T, 18), got {actions.shape}")
        if self.max_actions_per_call > 0:
            actions = actions[: self.max_actions_per_call]
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
        actions[:, 15] = np.clip(actions[:, 15], 0.0, 1.0)
        self.last_request_id = example["id"]
        self.last_request_sec = request_sec
        self.last_chunk_size = int(actions.shape[0])
        action_order = _get(data, "action_order", ACTION_ORDER)
        self.last_debug_record = {
            "episode": self.episode_idx,
            "step": request_step,
            "chunk": self.chunk_idx,
            "request": {
                "request_id": request["request_id"],
                "image_paths": [str(image_path)] if image_path is not None else [],
                "prompt": str(instruction),
                "lang": str(instruction),
                "state": _round_list(state) if state is not None else None,
                "state_normalized": request["state_normalized"],
                "send_state": self.send_state,
                "swap_rgb_channels": self.swap_rgb_channels,
                "camera_name": camera_name,
            },
            "response": {
                "ok": bool(_get(response, "ok", False)),
                "action_order": _jsonable(action_order),
                "raw_action_shape": raw_action_shape,
                "action_shape": list(actions.shape),
            },
            "request_sec": round(float(request_sec), 6),
            "actions": _round_list(actions),
            "first_action": _round_list(actions[0]) if actions.size else [],
            "last_action": _round_list(actions[-1]) if actions.size else [],
        }
        print(
            f"[QwenOFT18D] ep={self.episode_idx} step={self.step_idx} "
            f"camera={camera_name} chunk={actions.shape[0]} "
            f"request_sec={request_sec:.3f} "
            f"action_order={action_order}",
            flush=True,
        )
        self.step_idx += 1
        return actions

    def get_actions(self, TASK_ENV, observation: dict[str, Any]) -> np.ndarray:
        eval_video_path = getattr(TASK_ENV, "eval_video_path", None)
        debug_base = Path(eval_video_path) if eval_video_path is not None else None
        actions = self._request_actions(TASK_ENV.get_instruction(), observation, debug_base=debug_base)
        self._write_request_debug(TASK_ENV)
        return actions

    def log_chunk_timing(
        self,
        *,
        start_step: int,
        end_step: int,
        executed_steps: int,
        sim_sec: float,
        success: bool,
        step_limit: int,
    ) -> None:
        if not self.log_chunk_timing_enabled:
            self.chunk_idx += 1
            return
        request_sec = self.last_request_sec
        total_sec = sim_sec + request_sec if request_sec is not None else None
        print(
            f"[QwenOFT18D][chunk_timing] ep={self.episode_idx} chunk={self.chunk_idx} "
            f"request_id={self.last_request_id} start_step={start_step} end_step={end_step} "
            f"executed_steps={executed_steps} chunk_actions={self.last_chunk_size} "
            f"request_sec={_format_sec(request_sec)} sim_sec={sim_sec:.3f} "
            f"total_sec={_format_sec(total_sec)} success={bool(success)} step_limit={step_limit}",
            flush=True,
        )
        self.chunk_idx += 1

    def to_env_qpos_action(self, action18: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        action18 = np.asarray(action18, dtype=np.float64).reshape(-1)
        if action18.shape[0] != 18:
            raise ValueError(f"Expected QwenOFT18D action dim 18, got {action18.shape[0]}")

        current = _build_qwengroot18d_state(
            observation,
            torso_index=self.torso_index,
            head2_index=self.head2_index,
        ).astype(np.float64, copy=False)
        torso_delta = action18[16] - current[16]
        head2_delta = action18[17] - current[17]

        # RoboTwin qpos extras are deltas in order [head_joint_1, head_joint_2, torso_joint_4].
        extra_delta = np.array([0.0, head2_delta, torso_delta], dtype=np.float64)
        return np.concatenate([action18[:16], extra_delta], axis=0)

    def close(self) -> None:
        self.client.close()


def _arg(usr_args: dict[str, Any], name: str, default: Any) -> Any:
    value = usr_args.get(name, None)
    return default if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def get_model(usr_args):
    host = str(os.environ.get("QWENGROOT18D_HOST", _arg(usr_args, "host", "127.0.0.1")))
    port = int(os.environ.get("QWENGROOT18D_PORT", _arg(usr_args, "port", 5702)))
    return QwenOFT18DRobotwinPolicy(
        host=host,
        port=port,
        request_timeout=float(_arg(usr_args, "request_timeout", 300.0)),
        max_actions_per_call=int(_arg(usr_args, "max_actions_per_call", 16)),
        send_state=_as_bool(_arg(usr_args, "send_state", True)),
        swap_rgb_channels=_as_bool(_arg(usr_args, "swap_rgb_channels", True)),
        log_chunk_timing=_as_bool(_arg(usr_args, "log_chunk_timing", True)),
        log_request_debug=_as_bool(_arg(usr_args, "log_request_debug", True)),
        temp_root=str(_arg(usr_args, "temp_root", "/tmp/qwengroot18d_robotwin")),
        torso_index=int(_arg(usr_args, "torso_index", 0)),
        head2_index=int(_arg(usr_args, "head2_index", 1)),
    )


def eval(TASK_ENV, model, observation):
    actions18 = model.get_actions(TASK_ENV, observation)
    start_step = int(TASK_ENV.take_action_cnt)
    sim_start = time.perf_counter()
    executed_steps = 0
    for action18 in actions18:
        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
            break
        env_action = model.to_env_qpos_action(action18, observation)
        TASK_ENV.take_action(env_action, action_type="qpos")
        executed_steps += 1
        if TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_success:
            observation = TASK_ENV.get_obs()
    sim_sec = time.perf_counter() - sim_start
    model.log_chunk_timing(
        start_step=start_step,
        end_step=int(TASK_ENV.take_action_cnt),
        executed_steps=executed_steps,
        sim_sec=sim_sec,
        success=bool(TASK_ENV.eval_success),
        step_limit=int(TASK_ENV.step_lim),
    )
    return observation


def reset_model(model):
    model.reset()
