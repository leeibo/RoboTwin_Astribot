from ._base_task import Base_Task, PutBlockFanDoubleMixin


class put_block_plasticbox_fan_double(PutBlockFanDoubleMixin, Base_Task):
    ROTATE_TABLE_SHAPE = "fan_double"
    ROTATE_TABLE_CONFIG_KEY = "fan"
    ROTATE_FAN_DOUBLE_LAYER_CONFIG_KEY = "left_support"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    BLOCK_COUNT = 1
    BLOCK_LAYER_SEQUENCE = ("lower",)
    BLOCK_SIZE_RANGE = (0.018, 0.022)
    BLOCK_COLOR_CANDIDATES = ((0.10, 0.80, 0.20),)

    FIXED_LAYER_HEAD_JOINT2_ONLY = True
    LOWER_PLACE_PRE_DIS = 0.12
    LOWER_PLACE_DIS = 0.02
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.25

    TARGET_THETA_JITTER_DEG = 5.0
    TARGET_MODEL_NAME = "062_plasticbox"
    TARGET_MODEL_ID = 3
    TARGET_PADDING = 0.10
    TARGET_TASK_PREPOSITION = "into"
    TARGET_LAYER = "upper"
    TARGET_LAYER_SPECS = {
        "lower": {
            "r": 0.50,
            "theta_deg": 18.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": None,
        },
        "upper": {
            "r": 0.68,
            "theta_deg": 5.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": None,
        },
    }
    SUCCESS_XY_TOL = 0.08
    SUCCESS_Z_TOL = 0.06
