import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "envs" / "utils" / "camera_visibility.py"
MODULE_SPEC = importlib.util.spec_from_file_location("camera_visibility_module", MODULE_PATH)
camera_visibility = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(camera_visibility)


class DummyPose:
    def __init__(self, p, q):
        self.p = np.array(p, dtype=np.float64)
        self.q = np.array(q, dtype=np.float64)


class DummyObject:
    def __init__(self, p):
        self._pose = DummyPose(p, [1.0, 0.0, 0.0, 0.0])

    def get_pose(self):
        return self._pose


def test_world_point_in_camera_fov_center_hit():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    inside, debug = camera_visibility.is_world_point_in_camera_fov(
        world_point=[1.0, 0.0, 0.0],
        camera_pose=camera_pose,
        image_w=640,
        image_h=480,
        fovy_rad=np.deg2rad(60.0),
        ret_debug=True,
    )
    assert inside is True
    assert debug["inside_depth"] is True
    assert debug["inside_horizontal"] is True
    assert debug["inside_vertical"] is True


def test_world_point_in_camera_fov_rejects_point_behind_camera():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert (
        camera_visibility.is_world_point_in_camera_fov(
            world_point=[-1.0, 0.0, 0.0],
            camera_pose=camera_pose,
            image_w=640,
            image_h=480,
            fovy_rad=np.deg2rad(60.0),
        )
        is False
    )


def test_world_point_in_camera_fov_rejects_horizontal_and_vertical_outside():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert (
        camera_visibility.is_world_point_in_camera_fov(
            world_point=[1.0, 1.0, 0.0],
            camera_pose=camera_pose,
            image_w=640,
            image_h=480,
            fovy_rad=np.deg2rad(60.0),
        )
        is False
    )
    assert (
        camera_visibility.is_world_point_in_camera_fov(
            world_point=[1.0, 0.0, 1.0],
            camera_pose=camera_pose,
            image_w=640,
            image_h=480,
            fovy_rad=np.deg2rad(60.0),
        )
        is False
    )


def test_object_in_camera_fov_uses_object_center():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    obj = DummyObject([1.0, 0.1, 0.1])
    inside, debug = camera_visibility.is_object_in_camera_fov(
        obj=obj,
        camera_pose=camera_pose,
        image_w=640,
        image_h=480,
        fovy_rad=np.deg2rad(60.0),
        ret_debug=True,
    )
    assert inside is True
    assert np.allclose(debug["world_point"], np.array([1.0, 0.1, 0.1]))
    assert debug["mode"] == "center"
