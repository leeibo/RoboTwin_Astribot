from ._base_task import Base_Task, PutBlockFanDoubleMixin


class put_block_on_upper_hard(PutBlockFanDoubleMixin, Base_Task):
    ROTATE_TABLE_SHAPE = "fan_double"
    ROTATE_TABLE_CONFIG_KEY = "fan"
    ROTATE_FAN_DOUBLE_LAYER_CONFIG_KEY = "left_support"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    BLOCK_COUNT = 3
    BLOCK_LAYER_SEQUENCE = ("lower", "lower", "upper")
    BLOCK_LAYER_SPECS = {
        "lower": {
            "inner_margin": 0.10,
            "outer_margin": 0.20,
            "max_cyl_r": 0.50,
            "theta_shrink": 0.92,
            "theta_min_deg": -38.0,
            "theta_max_deg": -2.0,
        },
        "upper": {
            "inner_margin": 0.02,
            "outer_margin": 0.04,
            "max_cyl_r": 0.64,
            "theta_shrink": 0.92,
        },
    }

    PLATE_LAYER = "upper"
    PLATE_LAYER_SPECS = {
        "lower": {
            "r": 0.55,
            "theta_deg": -60.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": [0.025, 0.025, 0.025],
        },
        "upper": {
            "r": 0.70,
            "theta_deg": 0.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": [0.025, 0.025, 0.025],
        },
    }
    PLATE_PLACE_SLOT_OFFSETS = 0.025

    LOWER_PLACE_PRE_DIS = 0.15
    UPPER_PICK_ENTRY_Z_OFFSET = 0.06
    UPPER_PICK_PRE_GRASP_DIS = 0.06
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.20
