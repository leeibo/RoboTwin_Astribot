from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, POLICY_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


high = load("memer_high_test", "serve_high_policy.py")
deploy = load("memer_deploy_test", "deploy_policy.py")
low = load("memer_low_test", "serve_low_policy.py")


def observation() -> dict[str, Any]:
    return {
        "observation": {"camera_head": {"rgb": np.zeros((12, 16, 3), dtype=np.uint8)}},
        "joint_action": {
            "left_arm": np.arange(7, dtype=np.float32),
            "left_gripper": np.asarray([0.25], dtype=np.float32),
            "right_arm": np.arange(10, 17, dtype=np.float32),
            "right_gripper": np.asarray([0.75], dtype=np.float32),
            "torso": np.asarray([0.2], dtype=np.float32),
            "head": np.asarray([0.1, 0.5], dtype=np.float32),
        },
    }


class SequenceGenerator:
    mode = "test"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls = 0

    def generate(self, task, memory, recent):
        del task, memory, recent
        self.calls += 1
        return next(self.outputs)


def service(generator: Any) -> Any:
    return high.HighPolicyService(generator, 8, 8, 1, 5, 16, 12, 8)


def high_payload(session: str, episode: str, step: int = 0) -> dict[str, Any]:
    return {
        "session_id": session,
        "environment_type": "clean",
        "worker_id": session,
        "episode_id": episode,
        "task": "move the object",
        "env_step": step,
        "replan_reason": "episode_start" if step == 0 else "interval",
        "frames": [{"env_step": step, "image": np.zeros((12, 16, 3), dtype=np.uint8).tolist()}],
    }


class PromptMemoryTests(unittest.TestCase):
    def test_processor_contract_rejects_base_model_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "preprocessor_config.json"
            path.write_text(json.dumps({"min_pixels": 50176, "max_pixels": 115200}))
            self.assertEqual(high.validate_processor_contract(Path(temp_dir)), (50176, 115200))
            path.write_text(json.dumps({"min_pixels": 65536, "max_pixels": 16777216}))
            with self.assertRaisesRegex(ValueError, "processor_pixel_range_mismatch"):
                high.validate_processor_contract(Path(temp_dir))

    def test_training_prompt_is_reproduced(self) -> None:
        prompt = high.build_human_prompt(2, 3, "do task")
        self.assertEqual(
            prompt,
            "Task: do task\n"
            "Here are the selected frames from the entirety of the full video that are of particular importance:\n"
            "<image>\n<image>\n"
            "Here is a video of the most recent actions the robot has executed:\n"
            "<image>\n<image>\n<image>",
        )
        messages = high.build_messages("do task", [np.zeros((2, 2, 3), np.uint8)], [np.zeros((2, 2, 3), np.uint8)])
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": high.SYSTEM_PROMPT}])
        user_text = "".join(part.get("text", "") for part in messages[1]["content"])
        self.assertTrue(user_text.startswith("Task: do task"))
        self.assertEqual(sum(part["type"] == "image" for part in messages[1]["content"]), 2)

    def test_strict_schema_and_candidate_clustering(self) -> None:
        parsed = high.parse_high_output('{"current_subtask":"pick","keyframe_positions":[1,2]}', 2)
        self.assertEqual(parsed["current_subtask"], "pick")
        for raw in (
            '{"subtask":"pick","keyframe_positions":[]}',
            '{"current_subtask":"pick","keyframe_positions":[],"action":"x"}',
            '{"current_subtask":"pick","keyframe_positions":[0]}',
            '{"current_subtask":"pick","keyframe_positions":[true]}',
        ):
            with self.assertRaises(ValueError):
                high.parse_high_output(raw, 2)
        self.assertEqual(high.cluster_candidate_frames([0, 0, 4, 20, 25, 25], 5), [0, 25])

    def test_retry_fallback_terminal_and_session_reset(self) -> None:
        generator = SequenceGenerator([
            "bad", '{"current_subtask":"pick","keyframe_positions":[1]}',
            "bad", "bad", "bad", "bad",
        ])
        svc = service(generator)
        first = svc.act(high_payload("a", "ep-a"))
        self.assertEqual(first["new_subtask"], "pick")
        self.assertEqual(first["high_level_retry_success_count"], 1)
        fallback = svc.act(high_payload("a", "ep-a", 5))
        self.assertTrue(fallback["fallback_reuse"])
        self.assertEqual(fallback["new_subtask"], "pick")
        terminal = svc.act(high_payload("a", "ep-a", 10))
        self.assertTrue(terminal["terminal_parse_failure"])
        other = svc.store.get_or_create(
            "b", environment_type="randomized", worker_id="b", episode_id="ep-b", task="task"
        )
        self.assertEqual(other.frames, {})
        self.assertTrue(svc.reset({"session_id": "a"})["removed"])
        self.assertIn("b", svc.store.states)

    def test_vlm_trace_contains_prompt_images_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            svc = high.HighPolicyService(
                SequenceGenerator(['{"current_subtask":"pick","keyframe_positions":[1]}']),
                8, 8, 1, 5, 16, 12, 8, vlm_log_dir=Path(temp_dir),
            )
            response = svc.act(high_payload("trace-worker", "trace-episode"))
            self.assertEqual(len(response["vlm_log_ids"]), 1)
            rows = [json.loads(line) for line in (Path(temp_dir) / "vlm_requests.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["raw_output"], '{"current_subtask":"pick","keyframe_positions":[1]}')
            self.assertIn("Task: move the object", rows[0]["human_prompt"])
            self.assertEqual(rows[0]["recent_frame_steps"], [0])
            image_path = Path(temp_dir) / rows[0]["image_order"][0]["path"]
            self.assertTrue(image_path.is_file())
            self.assertEqual(rows[0]["image_order"][0]["shape"], [12, 16, 3])

    def test_memory_does_not_overlap_recent_training_context(self) -> None:
        generator = SequenceGenerator([
            '{"current_subtask":"pick","keyframe_positions":[1]}',
            '{"current_subtask":"place","keyframe_positions":[]}',
        ])
        svc = high.HighPolicyService(generator, 2, 8, 1, 5, 16, 12, 8)
        first = svc.act(high_payload("memory", "memory-episode", 0))
        self.assertEqual(first["candidate_votes"], [0])
        second_payload = high_payload("memory", "memory-episode", 3)
        second_payload["frames"] = [
            {"env_step": step, "image": np.zeros((12, 16, 3), dtype=np.uint8).tolist()}
            for step in (1, 2, 3)
        ]
        second = svc.act(second_payload)
        self.assertEqual(second["recent_frame_steps"], [2, 3])
        self.assertEqual(second["memory_frame_steps"], [0])
        self.assertTrue(set(second["memory_frame_steps"]).isdisjoint(second["recent_frame_steps"]))


class FakeHighClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.resets: list[str] = []

    def act(self, payload):
        self.requests.append(payload)
        return {
            "previous_subtask": None,
            "new_subtask": "predicted subtask",
            "terminal_parse_failure": False,
            "high_level_latency": 0.01,
        }

    def reset(self, session_id):
        self.resets.append(session_id)

    def close(self):
        pass


class TerminalHighClient(FakeHighClient):
    def act(self, payload):
        self.requests.append(payload)
        return {
            "previous_subtask": None,
            "new_subtask": None,
            "terminal_parse_failure": True,
            "parse_error_reason": ["invalid_json"],
        }


class FakeLowClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reset_count = 0

    def request(self, payload):
        self.requests.append(payload)
        if payload.get("__command__") == "reset":
            return {"ok": True}
        actions = np.repeat(np.asarray(payload["observation/state"])[None, :], 50, axis=0)
        actions[:, 16] = 0.7
        actions[:, 17] = 0.9
        return {"actions": actions, "prompt": payload["prompt"]}

    def reset(self):
        self.reset_count += 1

    def close(self):
        pass


class FakeTask:
    def __init__(self) -> None:
        self.task_name = "fake"
        self.take_action_cnt = 0
        self.step_lim = 20
        self.eval_success = False
        self.eval_done = False
        self.eval_video_path = None
        self.obs = observation()
        self.actions: list[np.ndarray] = []

    def get_instruction(self):
        return "full task instruction"

    def get_obs(self):
        return self.obs

    def take_action(self, action, action_type):
        self.assert_action_type = action_type
        self.actions.append(np.asarray(action))
        self.take_action_cnt += 1

    def _get_torso_joint_state_now(self):
        return np.asarray([0.2])

    def _get_head_joint_state_now(self):
        return np.asarray([0.1, 0.5])

    def mark_eval_failure(self, failure):
        self.eval_failed = True
        self.eval_done = True
        self.eval_failure_reason = failure["reason"]
        self.eval_failure_detail = failure


class AdapterTests(unittest.TestCase):
    def make_policy(self):
        return deploy.MemERRobotwinPolicy(
            "x", 1, "x", 2, 5, "", 50, 18, 5, 5, True, "absolute", 0, 1,
            "clean", "worker-test", False, tempfile.mkdtemp(), 320, 180, FakeHighClient(), FakeLowClient(),
        )

    def test_action_chunk_stops_at_env_step_replan_boundary(self) -> None:
        policy = deploy.MemERRobotwinPolicy(
            "x", 1, "x", 2, 5, "", 50, 18, 3, 5, True, "absolute", 0, 1,
            "clean", "worker-test", False, tempfile.mkdtemp(), 320, 180, FakeHighClient(), FakeLowClient(),
        )
        task = FakeTask()
        policy.reset()
        deploy.eval(task, policy, task.obs)
        self.assertEqual(task.take_action_cnt, 3)
        deploy.eval(task, policy, task.obs)
        self.assertEqual(task.take_action_cnt, 5)
        self.assertEqual(len(policy.high.requests), 1)
        deploy.eval(task, policy, task.obs)
        self.assertEqual(len(policy.high.requests), 2)
        self.assertEqual(policy.high.requests[-1]["env_step"], 5)

    def test_state_low_prompt_execution_and_qpos(self) -> None:
        state = deploy.build_state18(observation())
        np.testing.assert_allclose(state, [0, 1, 2, 3, 4, 5, 6, .25, 10, 11, 12, 13, 14, 15, 16, .75, .2, .5])
        policy = self.make_policy()
        task = FakeTask()
        policy.reset()
        deploy.eval(task, policy, task.obs)
        self.assertEqual(task.take_action_cnt, 5)
        self.assertTrue(all(action.shape == (19,) for action in task.actions))
        self.assertEqual(policy.low.requests[-1]["prompt"], "predicted subtask")
        self.assertNotEqual(policy.low.requests[-1]["prompt"], task.get_instruction())
        self.assertEqual(policy.high.requests[-1]["env_step"], 0)
        self.assertEqual(np.asarray(policy.high.requests[-1]["frames"][0]["image"]).shape, (180, 320, 3))
        self.assertEqual(sorted(policy.pending_frames), [1, 2, 3, 4, 5])
        np.testing.assert_allclose(task.actions[0][-3:], [0.0, 0.4, 0.5])
        old_session = policy.session_id
        policy.reset()
        self.assertIn(old_session, policy.high.resets)
        self.assertEqual(policy.pending_frames, {})

    def test_state_components_are_not_silently_padded(self) -> None:
        bad = observation()
        bad["joint_action"]["left_arm"] = np.zeros(6, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "left_arm must have 7"):
            deploy.build_state18(bad)

    def test_low_action_validation(self) -> None:
        valid = low.validate_actions({"actions": np.zeros((50, 18))}, 50)
        self.assertEqual(valid.shape, (50, 18))
        with self.assertRaises(ValueError):
            low.validate_actions({"actions": np.zeros((49, 18))}, 50)

    def test_norm_stats_training_contract(self) -> None:
        stats = {
            "norm_stats": {
                group: {metric: [0.0] * 32 for metric in ("mean", "std", "q01", "q99")}
                for group in ("state", "actions")
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "norm_stats.json"
            path.write_text(json.dumps(stats), encoding="utf-8")
            contract = low.validate_training_contract(path, 50)
            self.assertEqual(contract["state"]["environment_dim"], 18)
            self.assertEqual(contract["action"]["horizon"], 50)
            self.assertEqual(contract["action"]["inference_output"], "absolute")

    def test_low_normal_connection_close_is_not_reported_as_inference_error(self) -> None:
        from websockets.exceptions import ConnectionClosedOK

        class Packer:
            def pack(self, value):
                return value

        msgpack = type("Msgpack", (), {"Packer": Packer, "unpackb": staticmethod(lambda value: value)})

        class WebSocket:
            def __init__(self):
                self.sent = []
                self.closed = False

            async def send(self, value):
                self.sent.append(value)

            async def recv(self):
                raise ConnectionClosedOK(None, None)

            async def close(self, **kwargs):
                del kwargs
                self.closed = True

        server = object.__new__(low.LowPolicyServer)
        server.msgpack = msgpack
        server.metadata = {"ready": True}
        websocket = WebSocket()
        asyncio.run(server.handler(websocket))
        self.assertEqual(websocket.sent, [{"ready": True}])
        self.assertFalse(websocket.closed)
        values = np.zeros((50, 18)); values[0, 0] = np.nan
        with self.assertRaises(ValueError):
            low.validate_actions({"actions": values}, 50)

    def test_terminal_parse_failure_marks_episode_failed_without_low_action(self) -> None:
        policy = deploy.MemERRobotwinPolicy(
            "x", 1, "x", 2, 5, "", 50, 18, 5, 5, True, "absolute", 0, 1,
            "clean", "worker-test", False, tempfile.mkdtemp(), 320, 180,
            TerminalHighClient(), FakeLowClient(),
        )
        task = FakeTask()
        policy.reset()
        deploy.eval(task, policy, task.obs)
        self.assertTrue(task.eval_done)
        self.assertTrue(task.eval_failed)
        self.assertEqual(task.eval_failure_reason, "memer_high_level_parse_failure")
        self.assertEqual(task.take_action_cnt, 0)
        self.assertEqual(policy.low.requests, [])


if __name__ == "__main__":
    unittest.main()
