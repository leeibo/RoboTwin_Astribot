from ._put_block_target_fan_double_base import PutSingleBlockTargetFanDoubleBase
import numpy as np


class put_block_skillet_fan_double(PutSingleBlockTargetFanDoubleBase):
    TARGET_THETA_JITTER_DEG = 5.0
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
            "theta_deg": 5.0,
            "z_offset": 0.0,
            "qpos": [0.0, 0.0, 0.70710678, 0.70710678],
            "scale": None,
        },
    }
    SUCCESS_XY_TOL = 0.06
    SUCCESS_Z_TOL = 0.05

    def setup_demo(self, **kwargs):
        self._target_theta_deg_jitter_cache = {}
        super().setup_demo(**kwargs)

    def _get_plate_layer_spec(self, layer_name=None):
        target_spec = dict(super()._get_plate_layer_spec(layer_name))
        theta_cache = getattr(self, "_target_theta_deg_jitter_cache", None)
        if not isinstance(theta_cache, dict):
            theta_cache = {}
            self._target_theta_deg_jitter_cache = theta_cache
        cache_key = str(target_spec["layer"])
        if cache_key not in theta_cache:
            theta_cache[cache_key] = float(target_spec["theta_deg"]) + float(
                np.random.uniform(-self.TARGET_THETA_JITTER_DEG, self.TARGET_THETA_JITTER_DEG)
            )
        target_spec["theta_deg"] = float(theta_cache[cache_key])
        return target_spec
