from ._put_block_target_fan_double_base import PutSingleBlockTargetFanDoubleBase


class put_block_breadbasket_fan_double(PutSingleBlockTargetFanDoubleBase):
    TARGET_MODEL_NAME = "076_breadbasket"
    TARGET_MODEL_ID = 0
    TARGET_PADDING = 0.08
    TARGET_TASK_PREPOSITION = "into"
    TARGET_LAYER = "upper"
    TARGET_LAYER_SPECS = {
        "lower": {
            "r": 0.48,
            "theta_deg": -18.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": None,
        },
        "upper": {
            "r": 0.68,
            "theta_deg": 0.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": None,
        },
    }
    SUCCESS_XY_TOL = 0.09
    SUCCESS_Z_TOL = 0.08
