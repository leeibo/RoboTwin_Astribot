from __future__ import annotations

import importlib.util
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


deploy = load_module("hifvla_deploy_test", POLICY_DIR / "deploy_policy.py")
server_module = load_module("hifvla_server_test", POLICY_DIR / "serve_policy.py")


def make_observation() -> dict[str, Any]:
    return {
        "observation": {
            "camera_head": {
                "rgb": np.zeros((12, 16, 3), dtype=np.uint8),
            }
        },
        "joint_action": {
            "left_arm": np.arange(7, dtype=np.float32),
            "left_gripper": np.asarray([0.25], dtype=np.float32),
            "right_arm": np.arange(10, 17, dtype=np.float32),
            "right_gripper": np.asarray([0.75], dtype=np.float32),
            "torso": np.asarray([0.2], dtype=np.float32),
            "head": np.asarray([0.1, 0.5], dtype=np.float32),
        },
    }


class FakeTask:
    def __init__(self) -> None:
        self.task_name = "fake_task"
        self.eval_video_path = None
        self.take_action_cnt = 0
        self.step_lim = 20
        self.eval_success = False
        self.eval_done = False
        self.actions: list[np.ndarray] = []
        self.observation = make_observation()

    def get_instruction(self) -> str:
        return "move the object"

    def _get_torso_joint_state_now(self):
        return np.asarray([0.2])

    def _get_head_joint_state_now(self):
        return np.asarray([0.1, 0.5])

    def take_action(self, action, action_type: str) -> None:
        assert action_type == "qpos"
        self.actions.append(np.asarray(action))
        self.take_action_cnt += 1

    def get_obs(self):
        return self.observation


class MockHandler(BaseHTTPRequestHandler):
    resets: list[str] = []
    requests: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, {"ready": True, "mode": "test"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        body = self._body()
        if self.path == "/reset":
            self.resets.append(body["episode_id"])
            self._send(200, {"ok": True})
            return
        if self.path == "/act":
            self.requests.append(body)
            state = np.asarray(body["state"], dtype=float)
            actions = np.repeat(state[None, :], 8, axis=0)
            actions[:, 16] = 0.7
            actions[:, 17] = 0.9
            self._send(200, {"actions": actions.tolist(), "server_timing": {"inference_sec": 0.01}})
            return
        self._send(404, {"error": "not found"})


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        MockHandler.resets.clear()
        MockHandler.requests.clear()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def test_state_order(self) -> None:
        state = deploy.build_state18(make_observation())
        expected = np.asarray([
            0, 1, 2, 3, 4, 5, 6, 0.25,
            10, 11, 12, 13, 14, 15, 16, 0.75,
            0.2, 0.5,
        ], dtype=np.float32)
        np.testing.assert_allclose(state, expected)

    def test_http_reset_action_queue_and_qpos_conversion(self) -> None:
        with TemporaryDirectory() as temp:
            task = FakeTask()
            policy = deploy.HIFVLARobotwinPolicy(
                "127.0.0.1",
                self.httpd.server_address[1],
                request_timeout=5,
                max_actions_per_call=8,
                action_horizon=8,
                log_request_debug=False,
                temp_root=temp,
            )
            policy.reset()
            self.assertEqual(MockHandler.resets[-1], policy.episode_id)
            session_id = policy.episode_id
            actions = policy.get_actions(task, task.observation)
            self.assertEqual(len(actions), 8)
            request = MockHandler.requests[-1]
            self.assertEqual(request["episode_id"], policy.episode_id)
            self.assertEqual(np.asarray(request["image"]).shape, (12, 16, 3))
            self.assertEqual(len(request["state"]), 18)
            env_action = policy.to_env_qpos_action(actions[0], task, task.observation)
            self.assertEqual(env_action.shape, (19,))
            np.testing.assert_allclose(env_action[-3:], [0.0, 0.4, 0.5], atol=1e-6)
            policy.reset()
            self.assertEqual(policy.episode_id, session_id)
            self.assertEqual(MockHandler.resets[-1], session_id)
            self.assertEqual(len(policy.action_queue), 0)
            policy.close()

    def test_eval_executes_one_chunk(self) -> None:
        with TemporaryDirectory() as temp:
            task = FakeTask()
            policy = deploy.HIFVLARobotwinPolicy(
                "127.0.0.1",
                self.httpd.server_address[1],
                request_timeout=5,
                max_actions_per_call=8,
                action_horizon=8,
                log_request_debug=False,
                log_chunk_timing=False,
                temp_root=temp,
            )
            policy.reset()
            deploy.eval(task, policy, task.observation)
            self.assertEqual(task.take_action_cnt, 8)
            self.assertTrue(all(action.shape == (19,) for action in task.actions))
            policy.close()

    def test_http_client_retries_transient_connection_error(self) -> None:
        class Response:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {"actions": [[0.0] * 18] * 8}

        class FlakySession:
            def __init__(self) -> None:
                self.calls = 0

            def post(self, url, json, timeout):
                del url, json, timeout
                self.calls += 1
                if self.calls == 1:
                    raise deploy.requests.ConnectionError("connection dropped")
                return Response()

        client = deploy.HIFVLAHttpClient.__new__(deploy.HIFVLAHttpClient)
        client.base_url = "http://127.0.0.1:1"
        client.timeout = 5.0
        client.max_retries = 2
        client.retry_backoff = 0.0
        client.session = FlakySession()
        response = client.infer({"request_id": "request"})
        self.assertIn("actions", response)
        self.assertEqual(client.session.calls, 2)


class ServerCoreTests(unittest.TestCase):
    def test_motion_history_padding(self) -> None:
        def extractor(frames, fps, num_frames):
            self.assertEqual(fps, 6)
            self.assertEqual(num_frames, len(frames))
            return np.stack([
                np.full((2, 16, 16), index + 1, dtype=np.float32)
                for index in range(len(frames) - 1)
            ])

        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        first = server_module.build_motion_history([frame], extractor)
        self.assertEqual(first.shape, (1, 8, 2, 16, 16))
        self.assertTrue(np.all(first == 0))
        padded = server_module.build_motion_history([frame, frame, frame], extractor)
        self.assertEqual(padded.shape, (1, 8, 2, 16, 16))
        self.assertTrue(np.all(padded[0, :7] == 1))
        self.assertTrue(np.all(padded[0, 7] == 2))

    def test_sessions_are_isolated_and_reset(self) -> None:
        store = server_module.MotionHistoryStore(history_length=8, max_sessions=2)
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        store.append("episode-a", frame)
        store.append("episode-b", frame)
        self.assertEqual(store.size("episode-a"), 1)
        self.assertEqual(store.size("episode-b"), 1)
        self.assertTrue(store.reset("episode-a"))
        self.assertEqual(store.size("episode-a"), 0)
        self.assertEqual(store.size("episode-b"), 1)

    def test_payload_validation(self) -> None:
        payload = {
            "episode_id": "episode",
            "request_id": "request",
            "image": np.zeros((12, 16, 3), dtype=np.uint8).tolist(),
            "state": np.zeros(18).tolist(),
            "instruction": "test",
        }
        episode, request, image, state, instruction = server_module.validate_act_payload(payload)
        self.assertEqual((episode, request, instruction), ("episode", "request", "test"))
        self.assertEqual(image.shape, (12, 16, 3))
        self.assertEqual(state.shape, (18,))
        payload["state"] = [0.0] * 17
        with self.assertRaises(ValueError):
            server_module.validate_act_payload(payload)

    def test_duplicate_request_is_idempotent(self) -> None:
        class Config:
            history_length = 8

        class Runtime:
            config = Config()

            def __init__(self) -> None:
                self.inference_lock = threading.Lock()
                self.calls = 0

            def infer(self, image, state, instruction, frames):
                del image, instruction
                self.calls += 1
                actions = np.repeat(state[None, :], 8, axis=0)
                return actions, {"history_frames": len(frames)}

        payload = {
            "episode_id": "episode",
            "request_id": "request",
            "image": np.zeros((12, 16, 3), dtype=np.uint8).tolist(),
            "state": np.zeros(18).tolist(),
            "instruction": "test",
        }
        runtime = Runtime()
        server = server_module.HIFVLAServer(runtime)
        first = server.act(payload)
        second = server.act(payload)
        self.assertEqual(first, second)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(server.histories.size("episode"), 1)

        server.reset({"episode_id": "episode"})
        server.act(payload)
        self.assertEqual(runtime.calls, 2)


if __name__ == "__main__":
    unittest.main()
