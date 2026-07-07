import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


STARGVLA_ROOT = Path(os.environ.get("STARGVLA_REPO_ROOT", "/private/zjb/workspace/starVLA-A"))
if str(STARGVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARGVLA_ROOT))

QWENFAST_COT_PROMPT = (
    'Your task is: "{instruction}" The input images are ordered from earliest to latest, '
    'and the last image is the current view. Please think about the next action and output it. '
    'Your response should be in the format of: <think>...</think><action>...</action>.'
)


class ConstantActionChunkError(RuntimeError):
    pass


def _is_constant_action_chunk(actions: np.ndarray, tol: float = 1e-6) -> tuple[bool, float]:
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] <= 1:
        return False, 0.0
    max_delta = float(np.max(np.abs(np.diff(actions, axis=0))))
    return max_delta <= float(tol), max_delta


class StarVLAFastClient:
    def __init__(
        self,
        host: str,
        port: int,
        unnorm_key: Optional[str] = None,
        image_size: tuple[int, int] = (224, 224),
        action_steps: Optional[int] = None,
        action_type: str = "qpos",
        history_frames: int = 12,
        history_stride: Optional[int] = None,
        request_log_path: Optional[str] = None,
        request_image_dir: Optional[str] = None,
    ) -> None:
        try:
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        except ModuleNotFoundError as exc:
            if exc.name == "websockets":
                raise ModuleNotFoundError(
                    "The StarVLA websocket client requires the `websockets` package. "
                    "Run RoboTwin eval in an environment that has StarVLA dependencies, "
                    "or install it with `pip install websockets`."
                ) from exc
            raise

        self.client = WebsocketClientPolicy(host=host, port=port)
        self.unnorm_key = unnorm_key
        self.image_size = tuple(image_size)
        self.action_type = action_type
        self.server_metadata = self.client.get_server_metadata()
        self.action_chunk_size = int(self.server_metadata.get("action_chunk_size", action_steps or 16))
        self.action_steps = int(action_steps or self.action_chunk_size)
        self.max_history_frames = max(int(history_frames), 0)
        self.history_stride = int(history_stride or self.action_chunk_size)
        if self.history_stride <= 0:
            raise ValueError(f"history_stride must be positive, got {self.history_stride}")
        history_buffer_frames = self.max_history_frames * self.history_stride + 1
        self.image_history = deque(maxlen=history_buffer_frames)
        self.request_log_path = Path(request_log_path).expanduser() if request_log_path else None
        if self.request_log_path is not None:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.request_image_dir = Path(request_image_dir).expanduser() if request_image_dir else None
        if self.request_image_dir is not None:
            self.request_image_dir.mkdir(parents=True, exist_ok=True)
        self.language: Optional[str] = None
        self.cached_actions: Optional[np.ndarray] = None
        self.cache_start_step = 0
        self.request_seed = None

        print(
            "[StarVLAFastClient] connected: "
            f"host={host}, port={port}, action_chunk_size={self.action_chunk_size}, "
            f"action_steps={self.action_steps}, unnorm_key={self.unnorm_key}, "
            f"max_history_frames={self.max_history_frames}, "
            f"history_stride={self.history_stride}, "
            f"history_buffer_frames={self.image_history.maxlen}, "
            f"request_log_path={self.request_log_path}, "
            f"request_image_dir={self.request_image_dir}, "
            "image_channel_order=BGR, "
            f"metadata={self.server_metadata}"
        )

    def reset(self) -> None:
        self.language = None
        self.cached_actions = None
        self.cache_start_step = 0
        self.image_history.clear()

    def _select_image_history(self, env_step: int) -> tuple[list[int], list[np.ndarray]]:
        by_step = {int(step): image for step, image in self.image_history}
        current_step = int(env_step)
        selected_steps = []
        step = current_step - self.history_stride
        while step >= 0 and len(selected_steps) < self.max_history_frames:
            if step in by_step:
                selected_steps.append(step)
            step -= self.history_stride
        selected_steps = list(reversed(selected_steps)) + [current_step]
        return selected_steps, [by_step[step] for step in selected_steps if step in by_step]

    def _resize_images(self, images: list[np.ndarray]) -> list[np.ndarray]:
        width, height = self.image_size
        resized = [cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) for img in images]
        return [np.ascontiguousarray(img[..., ::-1]) for img in resized]

    def _write_request_log(self, record: Dict[str, Any]) -> None:
        if self.request_log_path is None:
            return
        with self.request_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_name_part(value: Any) -> str:
        text = "unknown" if value is None else str(value)
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)

    def set_request_seed(self, seed: Any) -> None:
        self.request_seed = seed

    def _summarize_images(
        self,
        images: list[np.ndarray],
        frame_indices: list[int],
        env_step: int,
    ) -> list[Dict[str, Any]]:
        summaries = []
        for idx, image in enumerate(images):
            arr = np.asarray(image)
            frame_step = frame_indices[idx] if idx < len(frame_indices) else env_step
            image_ref = self._save_request_image(arr, frame_step=frame_step)
            summaries.append(
                {
                    "index": idx,
                    "frame_step": int(frame_step),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "mean": float(arr.mean()) if arr.size else None,
                    "min": int(arr.min()) if arr.size else None,
                    "max": int(arr.max()) if arr.size else None,
                    **image_ref,
                }
            )
        return summaries

    def _save_request_image(self, image: np.ndarray, frame_step: int) -> Dict[str, Any]:
        arr = np.ascontiguousarray(np.asarray(image))
        if self.request_image_dir is None:
            return {"path": None, "saved": False}

        seed_part = self._safe_name_part(self.request_seed)
        path = self.request_image_dir / f"{seed_part}_{int(frame_step)}.png"
        ok = cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write request image: {path}")
        return {"path": str(path), "saved": True}

    def _build_example(
        self,
        resized_images: list[np.ndarray],
        instruction: str,
        history_frame_indices: list[int],
    ) -> Dict[str, Any]:
        num_frames = len(resized_images)
        return {
            "image": resized_images,
            "lang": str(instruction),
            "robot_tag": "robotwin_astribot",
            "history_mode": "action_keyframe",
            "history_frame_indices": list(history_frame_indices),
            "num_frames": num_frames,
            "num_history_frames": max(num_frames - 1, 0),
        }

    def _build_prompt_preview(self, instruction: str) -> str:
        return QWENFAST_COT_PROMPT.replace("{instruction}", str(instruction))

    def _request_actions(
        self,
        image_history: list[np.ndarray],
        instruction: str,
        env_step: int,
        camera_key: str,
        history_frame_indices: list[int],
    ) -> np.ndarray:
        resized_images = self._resize_images(image_history)
        example = self._build_example(resized_images, instruction, history_frame_indices)
        prompt_preview = self._build_prompt_preview(instruction)
        payload: Dict[str, Any] = {
            "examples": [example],
            "do_sample": False,
        }
        if self.unnorm_key is not None:
            payload["unnorm_key"] = self.unnorm_key

        request_time = time.time()
        response = self.client.predict_action(payload)
        if not response.get("ok", False):
            self._write_request_log(
                {
                    "timestamp": request_time,
                    "env_step": int(env_step),
                    "instruction": str(instruction),
                    "request": {
                        "prompt_preview": prompt_preview,
                        "example_fields": {
                            key: value
                            for key, value in example.items()
                            if key not in {"image", "state"}
                        },
                        "camera_key": camera_key,
                        "image_history": self._summarize_images(
                            resized_images,
                            history_frame_indices,
                            env_step,
                        ),
                        "unnorm_key": self.unnorm_key,
                        "do_sample": False,
                        "image_channel_order": "BGR",
                    },
                    "response": response,
                }
            )
            raise RuntimeError(f"StarVLA server inference failed: {response.get('error', response)}")

        actions = np.asarray(response["data"]["actions"][0], dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk with shape (T, D), got {actions.shape}")
        const_fail, temporal_max_delta = _is_constant_action_chunk(actions)
        self._write_request_log(
            {
                "timestamp": request_time,
                "env_step": int(env_step),
                "instruction": str(instruction),
                "request": {
                    "prompt_preview": prompt_preview,
                    "example_fields": {
                        key: value
                        for key, value in example.items()
                        if key not in {"image", "state"}
                    },
                    "camera_key": camera_key,
                    "image_history": self._summarize_images(
                        resized_images,
                        history_frame_indices,
                        env_step,
                    ),
                    "unnorm_key": self.unnorm_key,
                    "do_sample": False,
                    "image_channel_order": "BGR",
                },
                "response": {
                    "status": response.get("status"),
                    "ok": response.get("ok"),
                    "type": response.get("type"),
                    "actions_shape": list(actions.shape),
                    "decode_status": "const_fail" if const_fail else "ok",
                    "temporal_max_delta": temporal_max_delta,
                    "actions": actions.tolist(),
                },
                "action_mapping": {
                    "model_order": "left_arm7,left_gripper,right_arm7,right_gripper,torso,head2",
                    "robotwin_order": "left_arm7,left_gripper,right_arm7,right_gripper,head1_delta,head2_delta,torso_delta",
                    "extra_conversion": "model torso/head2 absolute targets are converted to RobotWin deltas at execution time",
                },
            }
        )
        if const_fail:
            raise ConstantActionChunkError(
                f"FAST decode produced a constant action chunk at env_step={env_step}; "
                f"history_frame_indices={history_frame_indices}"
            )
        return actions

    def get_actions(
        self,
        head_image: np.ndarray,
        instruction: str,
        env_step: int,
        camera_key: str,
    ) -> np.ndarray:
        if instruction != self.language:
            self.reset()
            self.language = str(instruction)

        self.image_history.append((int(env_step), np.asarray(head_image)))
        history_frame_indices, image_history = self._select_image_history(env_step)

        cache_idx = env_step - self.cache_start_step
        if self.cached_actions is None or cache_idx < 0 or cache_idx >= len(self.cached_actions):
            self.cached_actions = self._request_actions(
                image_history,
                instruction,
                env_step,
                camera_key,
                history_frame_indices,
            )
            self.cache_start_step = env_step
            cache_idx = 0

        actions = self.cached_actions[cache_idx:cache_idx + self.action_steps]
        if len(actions) == 0:
            self.cached_actions = self._request_actions(
                image_history,
                instruction,
                env_step,
                camera_key,
                history_frame_indices,
            )
            self.cache_start_step = env_step
            actions = self.cached_actions[:self.action_steps]
        return actions


def encode_obs(observation):
    obs = observation["observation"]
    camera_key = "camera_head"
    if camera_key not in obs:
        raise KeyError(f"Observation has no camera_head key. Available cameras: {list(obs.keys())}")
    head_image = obs[camera_key]["rgb"]
    return head_image, camera_key


def _parse_image_size(value) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError(f"image_size string must be 'width,height', got {value!r}")
        return parts[0], parts[1]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"image_size must have two values, got {value!r}")


def _current_joint_value(values, default=0.0, index=0) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] == 0:
        return float(default)
    index = min(max(int(index), 0), values.shape[0] - 1)
    return float(values[index])


def adapt_model_action_to_robotwin(action: np.ndarray, task_env) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] != 18:
        raise ValueError(f"Expected StarVLA Astribot action dim 18, got {action.shape[0]}")

    head_now = task_env._get_head_joint_state_now()
    torso_now = task_env._get_torso_joint_state_now()
    head2_now = _current_joint_value(head_now, index=1)
    torso_now = _current_joint_value(torso_now, index=0)
    torso_target = float(action[16])
    head2_target = float(action[17])

    env_action = np.zeros(19, dtype=np.float64)
    env_action[:16] = action[:16]
    env_action[16] = 0.0                       # head1 delta: unused
    env_action[17] = head2_target - head2_now  # head2 absolute target -> delta
    env_action[18] = torso_target - torso_now  # torso absolute target -> delta
    return env_action


def get_model(usr_args):
    return StarVLAFastClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 7980)),
        unnorm_key=usr_args.get("unnorm_key", None),
        image_size=_parse_image_size(usr_args.get("image_size", [224, 224])),
        action_steps=int(usr_args.get("action_steps", 16)),
        action_type=usr_args.get("action_type", "qpos"),
        history_frames=int(usr_args.get("history_frames", 6)),
        history_stride=usr_args.get("history_stride", None),
        request_log_path=usr_args.get("request_log_path", None),
        request_image_dir=usr_args.get("request_image_dir", None),
    )


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()
    head_image, camera_key = encode_obs(observation)
    if hasattr(model, "set_request_seed"):
        model.set_request_seed(
            getattr(
                TASK_ENV,
                "eval_seed",
                getattr(TASK_ENV, "seed", None),
            )
        )
    try:
        actions = model.get_actions(
            head_image=head_image,
            instruction=str(instruction),
            env_step=TASK_ENV.take_action_cnt,
            camera_key=camera_key,
        )
    except ConstantActionChunkError as exc:
        print(f"[StarVLAFastClient] {exc}; mark current episode as failed.")
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return

    for action in actions:
        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
            break
        TASK_ENV.take_action(adapt_model_action_to_robotwin(action, TASK_ENV), action_type=model.action_type)
        observation = TASK_ENV.get_obs()


def reset_model(model):
    model.reset()
