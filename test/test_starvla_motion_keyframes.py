import ast
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.starvla_astribot.deploy_policy import FrameRecord, StarVLAOFTClient
from policy.starvla_astribot.motion_keyframes import (
    CausalMotionKeyframeDetector,
    MotionKeyframeConfig,
)


def load_batch_reference():
    path = ROOT / "script" / "update_lerobot_motion_keyframes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {
        "DetectorConfig",
        "causal_run_events",
        "mark_causal_boundaries",
        "suppress_nearby",
        "compute_motion_keyframes",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    namespace = {
        "dataclass": dataclass,
        "np": np,
        "STATE_COLUMN": "observation.state",
        "REQUIRED_STATE_NAMES": (
            "left_gripper",
            "right_gripper",
            "torso_yaw",
            "head_2",
        ),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["DetectorConfig"], namespace["compute_motion_keyframes"]


class MotionKeyframeStrategyTest(unittest.TestCase):
    def test_online_detector_matches_lerobot_batch_script(self):
        reference_config, compute_reference = load_batch_reference()
        state_names = [
            *(f"left_arm_{index}" for index in range(7)),
            "left_gripper",
            *(f"right_arm_{index}" for index in range(7)),
            "right_gripper",
            "torso_yaw",
            "head_2",
        ]
        states = np.zeros((80, 18), dtype=np.float64)
        states[:, 7] = 1.0
        states[:, 15] = 1.0
        states[8:18, 7] = 0.0
        states[25:38, 15] = 0.0
        states[45:56, 16] = np.arange(11) * 0.01
        states[56:, 16] = states[55, 16]
        states[62:70, 17] = np.arange(8) * -0.01
        states[70:, 17] = states[69, 17]

        expected = compute_reference(states, state_names, reference_config())
        detector = CausalMotionKeyframeDetector(MotionKeyframeConfig())
        actual = np.asarray(
            [int(detector.update(state).is_keyframe) for state in states],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_detector_reset_restores_initial_keyframe(self):
        detector = CausalMotionKeyframeDetector()
        state = np.zeros(18, dtype=np.float32)
        self.assertTrue(detector.update(state).is_keyframe)
        self.assertFalse(detector.update(state).is_keyframe)
        detector.reset()
        decision = detector.update(state)
        self.assertTrue(decision.is_keyframe)
        self.assertEqual(decision.frame_index, 0)

    def test_motion_history_keeps_twelve_prior_keyframes_plus_current(self):
        client = object.__new__(StarVLAOFTClient)
        client.max_history_frames = 12
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        state = np.zeros(18, dtype=np.float32)
        prior = [
            FrameRecord(index, image, state, {}, motion_keyframe=True)
            for index in range(20)
        ]
        current = FrameRecord(20, image, state, {}, motion_keyframe=False)
        client.records = prior + [current]

        selected = client._select_keyframe_history(current, "motion_keyframe")
        self.assertEqual([record.step for record in selected], list(range(8, 21)))

    def test_target_policy_uses_training_history_settings(self):
        config_path = ROOT / "policy" / "oft_subtask_motion_12_ws" / "deploy_policy.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["history_mode"], "motion_keyframe")
        self.assertEqual(config["history_frames"], 12)
        self.assertTrue(config["send_state"])
        self.assertEqual(config["motion_keyframe_strategy"], "causal_gripper_rotation")
        self.assertFalse(config["subtask_planner_enabled"])


if __name__ == "__main__":
    unittest.main()
