from ._put_block_target_fan_double_base import PutSingleBlockTargetFanDoubleBase


class put_block_skillet_fan_double(PutSingleBlockTargetFanDoubleBase):
    TARGET_MODEL_NAME = "106_skillet"
    TARGET_MODEL_ID = 0
    TARGET_PADDING = 0.06
    TARGET_TASK_PREPOSITION = "on"
    TARGET_LAYER = "upper"
    TARGET_LAYER_SPECS = {
        "lower": {
            "r": 0.47,
            "theta_deg": 0.0,
            "z_offset": 0.0,
            "qpos": [0.0, 0.0, 0.70710678, 0.70710678],
            "scale": None,
        },
        "upper": {
            "r": 0.68,
            "theta_deg": 0.0,
            "z_offset": 0.0,
            "qpos": [0.0, 0.0, 0.70710678, 0.70710678],
            "scale": None,
        },
    }
    SUCCESS_XY_TOL = 0.06
    SUCCESS_Z_TOL = 0.05
