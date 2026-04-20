import importlib
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

rand_clutter = importlib.import_module("envs.utils.rand_create_cluttered_actor")


def test_point_in_fan_region_respects_sector_bounds():
    fan_region = {
        "center_xy": [0.0, 0.0],
        "inner_radius": 0.55,
        "outer_radius": 0.9,
        "center_theta_rad": np.deg2rad(90.0),
        "angle_rad": np.deg2rad(150.0),
    }
    assert rand_clutter._point_in_fan_region(0.0, 0.7, fan_region) is True
    assert rand_clutter._point_in_fan_region(0.0, 0.4, fan_region) is False
    assert rand_clutter._point_in_fan_region(0.8, 0.0, fan_region) is False


def test_sample_fan_xy_prefers_back_region_and_respects_radius_floor():
    fan_region = {
        "center_xy": [0.0, 0.0],
        "inner_radius": 0.3,
        "outer_radius": 0.9,
        "center_theta_rad": np.deg2rad(90.0),
        "angle_rad": np.deg2rad(150.0),
        "radius_floor": 0.55,
        "candidate_count": 8,
        "radial_bias_power": 0.35,
    }
    radii = []
    for _ in range(64):
        sampled = rand_clutter._sample_fan_xy(fan_region=fan_region, point_radius=0.02, prefer_back=True)
        assert sampled is not None
        x, y = sampled
        radius = float(np.hypot(x, y))
        radii.append(radius)
        assert radius >= 0.55
        assert rand_clutter._point_in_fan_region(
            x,
            y,
            {
                "center_xy": [0.0, 0.0],
                "inner_radius": 0.57,
                "outer_radius": 0.88,
                "center_theta_rad": fan_region["center_theta_rad"],
                "angle_rad": fan_region["angle_rad"],
            },
        )
    assert float(np.mean(radii)) > 0.68


def test_objaverse_model_exists_requires_model_urdf(tmp_path: Path):
    base_dir = tmp_path / "objaverse"
    model_dir = base_dir / "pencil" / "001"
    model_dir.mkdir(parents=True)

    assert rand_clutter._objaverse_model_exists(base_dir, "pencil", "001") is False

    (model_dir / "model.urdf").write_text("<robot/>", encoding="utf-8")
    assert rand_clutter._objaverse_model_exists(base_dir, "pencil", "001") is True
