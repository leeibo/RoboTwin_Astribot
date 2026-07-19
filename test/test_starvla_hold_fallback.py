import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from policy.starvla_astribot import deploy_policy as MODULE


def test_invalid_fast_action_error_is_detected():
    error = RuntimeError(
        "StarVLA server inference failed: QwenFast generation produced invalid FAST action token sequence(s)"
    )
    assert MODULE._is_invalid_fast_action_error(error)
    assert not MODULE._is_invalid_fast_action_error(RuntimeError("connection closed"))


def test_hold_actions_repeat_current_18d_state():
    state = np.arange(18, dtype=np.float32)
    record = MODULE.FrameRecord(step=7, image=np.zeros((2, 2, 3)), state=state, annotation={})

    actions = MODULE._hold_current_state_actions([record], action_steps=16)

    assert actions.shape == (16, 18)
    np.testing.assert_allclose(actions, np.repeat(state[None, :], 16, axis=0))
