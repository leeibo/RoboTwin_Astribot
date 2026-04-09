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


class DummyAABBComponent:
    def __init__(self, aabb):
        self._aabb = np.array(aabb, dtype=np.float64)

    def compute_global_aabb_tight(self):
        return self._aabb


class DummyAABBEntity:
    def __init__(self, aabb):
        self._components = [DummyAABBComponent(aabb)]

    def get_components(self):
        return list(self._components)


class DummyAABBObject:
    def __init__(self, center, aabb):
        self._pose = DummyPose(center, [1.0, 0.0, 0.0, 0.0])
        self.actor = DummyAABBEntity(aabb)

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


def test_object_in_camera_fov_aabb_detects_partial_visibility_when_center_is_outside():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    obj = DummyAABBObject(
        center=[1.0, 0.9, 0.0],
        aabb=[
            [0.95, 0.65, -0.05],
            [1.05, 1.15, 0.05],
        ],
    )

    assert (
        camera_visibility.is_object_in_camera_fov(
            obj=obj,
            camera_pose=camera_pose,
            image_w=640,
            image_h=480,
            fovy_rad=np.deg2rad(60.0),
            mode="center",
        )
        is False
    )

    inside, debug = camera_visibility.is_object_in_camera_fov(
        obj=obj,
        camera_pose=camera_pose,
        image_w=640,
        image_h=480,
        fovy_rad=np.deg2rad(60.0),
        mode="aabb",
        ret_debug=True,
    )
    assert inside is True
    assert debug["mode"] == "aabb"
    assert np.allclose(debug["world_point"], np.array([1.0, 0.9, 0.0]))
    assert debug["projected_bbox"]["u_min"] < debug["visible_uv_bounds"]["u_max"]


def test_project_object_to_image_uv_aabb_returns_in_frame_representative_point():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    obj = DummyAABBObject(
        center=[1.0, 0.9, 0.0],
        aabb=[
            [0.95, 0.65, -0.05],
            [1.05, 1.15, 0.05],
        ],
    )
    (u_norm, v_norm), debug = camera_visibility.project_object_to_image_uv(
        obj=obj,
        camera_pose=camera_pose,
        image_w=640,
        image_h=480,
        fovy_rad=np.deg2rad(60.0),
        mode="aabb",
        ret_debug=True,
    )
    assert debug["inside"] is True
    assert 0.0 <= u_norm <= 1.0
    assert 0.0 <= v_norm <= 1.0


def test_project_world_point_to_image_uv_center_hit():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    (u_norm, v_norm), debug = camera_visibility.project_world_point_to_image_uv(
        world_point=[1.0, 0.0, 0.0],
        camera_pose=camera_pose,
        image_w=640,
        image_h=480,
        fovy_rad=np.deg2rad(60.0),
        ret_debug=True,
    )
    assert np.isclose(u_norm, 0.5)
    assert np.isclose(v_norm, 0.5)
    assert debug["inside"] is True
    assert np.isclose(debug["pixel_x"], 319.5)
    assert np.isclose(debug["pixel_y"], 239.5)


def test_project_world_point_to_image_uv_left_of_center():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    u_norm, v_norm = camera_visibility.project_world_point_to_image_uv(
        world_point=[1.0, 0.2, 0.0],
        camera_pose=camera_pose,
        image_w=640,
        image_h=480,
        fovy_rad=np.deg2rad(60.0),
    )
    assert u_norm < 0.5
    assert np.isclose(v_norm, 0.5)


def test_image_u_to_yaw_error_rad_center_is_zero_and_sign_matches_image_side():
    fovy_rad = np.deg2rad(60.0)
    assert np.isclose(
        camera_visibility.image_u_to_yaw_error_rad(0.5, image_w=640, image_h=480, fovy_rad=fovy_rad),
        0.0,
    )
    assert camera_visibility.image_u_to_yaw_error_rad(0.25, image_w=640, image_h=480, fovy_rad=fovy_rad) > 0.0
    assert camera_visibility.image_u_to_yaw_error_rad(0.75, image_w=640, image_h=480, fovy_rad=fovy_rad) < 0.0


def test_image_u_to_yaw_error_rad_inverts_projection_for_planar_target():
    camera_pose = DummyPose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    fovy_rad = np.deg2rad(60.0)
    expected_yaw = np.deg2rad(25.0)
    world_point = [1.0, np.tan(expected_yaw), 0.0]
    u_norm, _ = camera_visibility.project_world_point_to_image_uv(
        world_point=world_point,
        camera_pose=camera_pose,
        image_w=640,
        image_h=480,
        fovy_rad=fovy_rad,
    )
    recovered_yaw = camera_visibility.image_u_to_yaw_error_rad(
        u_norm,
        image_w=640,
        image_h=480,
        fovy_rad=fovy_rad,
    )
    assert np.isclose(recovered_yaw, expected_yaw)
