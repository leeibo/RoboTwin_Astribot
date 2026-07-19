import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from policy.starvla_astribot.deploy_policy import FrameRecord, StarVLAOFTClient


class FakePlannerClient:
    def __init__(self, text):
        self.text = text

    def predict_action(self, _payload):
        return {"ok": True, "data": {"planner_text": [self.text]}}


class PlannerOFTOutputTest(unittest.TestCase):
    def test_keeps_valid_zero_based_indices(self):
        text = '<subtask>pick up block</subtask><retrieval>[0, 4, 4]</retrieval>'

        subtask, retrieval = StarVLAOFTClient._parse_planner_oft_output(text, num_frames=5)

        self.assertEqual(subtask, "pick up block")
        self.assertEqual(retrieval, [0, 4])

    def test_normalizes_frame_count_to_last_index(self):
        text = '<subtask>pick up block</subtask><retrieval>[0, 5]</retrieval>'

        _, retrieval = StarVLAOFTClient._parse_planner_oft_output(text, num_frames=5)

        self.assertEqual(retrieval, [0, 4])

    def test_ignores_other_out_of_range_indices(self):
        for index in (-1, 6, 10, 12):
            with self.subTest(index=index):
                text = f'<subtask>pick up block</subtask><retrieval>[0, {index}, 4]</retrieval>'

                _, retrieval = StarVLAOFTClient._parse_planner_oft_output(text, num_frames=5)

                self.assertEqual(retrieval, [0, 4])

    def test_allows_empty_retrieval_after_filtering(self):
        text = '<subtask>pick up block</subtask><retrieval>[10, 12]</retrieval>'

        _, retrieval = StarVLAOFTClient._parse_planner_oft_output(text, num_frames=9)

        self.assertEqual(retrieval, [])

    def test_malformed_output_reuses_previous_subtask_without_retrieval(self):
        malformed = '<think>repeated partial output</think><retrieval>[0, 4]</retrieval>'
        current = FrameRecord(
            step=0,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            state=np.zeros(18, dtype=np.float32),
            annotation={},
        )
        client = StarVLAOFTClient.__new__(StarVLAOFTClient)
        client.planner_oft_client = FakePlannerClient(malformed)
        client.records = [current]
        client.history_stride = 16
        client.planner_oft_max_history_frames = 12
        client.planner_oft_max_new_tokens = 192
        client.image_size = (2, 2)
        client.swap_rgb_channels = False
        client.episode_idx = 0
        client.chunk_idx = 0
        client.last_planner_oft_subtask = "previous valid subtask"
        client.planner_memory_steps = set()

        with tempfile.TemporaryDirectory() as tmpdir:
            client.request_log_path = Path(tmpdir) / "requests.jsonl"
            memory, subtask = client._request_planner_oft(current, "full task instruction")
            log_text = client.request_log_path.read_text()

        self.assertEqual(memory, [current])
        self.assertEqual(subtask, "previous valid subtask")
        self.assertIn("planner_parse_error", log_text)
        self.assertIn("missing <subtask>", log_text)


if __name__ == "__main__":
    unittest.main()
