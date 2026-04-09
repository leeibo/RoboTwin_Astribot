import ast
import importlib.util
import textwrap
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_TASK_PATH = REPO_ROOT / "envs" / "_base_task.py"
CAMERA_VIS_PATH = REPO_ROOT / "envs" / "utils" / "camera_visibility.py"

CAMERA_VIS_SPEC = importlib.util.spec_from_file_location("camera_visibility_module", CAMERA_VIS_PATH)
camera_visibility = importlib.util.module_from_spec(CAMERA_VIS_SPEC)
assert CAMERA_VIS_SPEC.loader is not None
CAMERA_VIS_SPEC.loader.exec_module(camera_visibility)


def _extract_base_methods(method_names):
    source = BASE_TASK_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Base_Task")

    segments = []
    found_names = set()
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name in method_names:
            segment = ast.get_source_segment(source, node)
            assert segment is not None, node.name
            segments.append(segment)
            found_names.add(node.name)

    missing = sorted(set(method_names) - found_names)
    assert not missing, f"Missing Base_Task methods: {missing}"

    namespace = {
        "np": np,
        "is_world_point_in_camera_fov": camera_visibility.is_world_point_in_camera_fov,
        "image_u_to_yaw_error_rad": camera_visibility.image_u_to_yaw_error_rad,
    }
    class_body = "\n\n".join(textwrap.indent(segment, "    ") for segment in segments)
    class_source = "class ExtractedBase:\n" + class_body + "\n"
    exec(class_source, namespace)
    return namespace["ExtractedBase"]


class DummyPose:
    def __init__(self, p, q):
        self.p = np.array(p, dtype=np.float64)
        self.q = np.array(q, dtype=np.float64)


def test_left_table_scan_completion_uses_midpoint_not_edge_pair():
    ExtractedBase = _extract_base_methods(
        [
            "_get_rotate_fan_table_edge_world_points",
            "_get_rotate_fan_table_side_mid_world_point",
            "_is_rotate_table_edge_visible_in_current_view",
        ]
    )

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.rotate_table_shape = "fan"
            self.rotate_table_center_xy = np.array([0.0, 0.4], dtype=np.float64)
            self.rotate_fan_inner_radius = 0.3
            self.rotate_fan_outer_radius = 0.9
            self.rotate_fan_theta_start_world_rad = np.deg2rad(-20.0)
            self.rotate_fan_theta_end_world_rad = np.deg2rad(110.0)
            self.rotate_table_top_z = 0.74

        def _get_scan_camera_pose(self):
            return DummyPose([0.0, 0.0, 0.9], [1.0, 0.0, 0.0, 0.0])

        def _get_scan_camera_runtime_spec(self):
            return {"w": 640, "h": 480, "fovy_rad": np.deg2rad(60.0), "far": 5.0}

    task = DummyTask()
    left_mid = task._get_rotate_fan_table_side_mid_world_point("left")
    left_edges = task._get_rotate_fan_table_edge_world_points("left")

    globals_dict = ExtractedBase._is_rotate_table_edge_visible_in_current_view.__globals__
    original_fov_fn = globals_dict["is_world_point_in_camera_fov"]

    def fake_is_world_point_in_camera_fov(world_point, **kwargs):
        point = tuple(np.round(np.array(world_point, dtype=np.float64), 6).tolist())
        midpoint = tuple(np.round(left_mid, 6).tolist())
        left_edge_points = {tuple(np.round(point_arr, 6).tolist()) for point_arr in left_edges}
        visible = point == midpoint
        if point in left_edge_points:
            visible = False
        return visible, {"yaw_err_rad": 0.0, "pitch_err_rad": 0.0}

    globals_dict["is_world_point_in_camera_fov"] = fake_is_world_point_in_camera_fov
    try:
        left_visible, left_debug = task._is_rotate_table_edge_visible_in_current_view("left")
        right_visible, right_debug = task._is_rotate_table_edge_visible_in_current_view("right")
    finally:
        globals_dict["is_world_point_in_camera_fov"] = original_fov_fn

    assert left_visible is True
    assert left_debug["mode"] == "midpoint"
    assert len(left_debug["point_visibilities"]) == 1
    assert right_visible is False
    assert right_debug["mode"] == "edge_pair"


def test_stage2_directly_rotates_to_full_target_theta():
    ExtractedBase = _extract_base_methods(["_wrap_to_pi", "_fine_center_rotate_registry_target"])

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.rotate_stage2_center_tol_rad = np.deg2rad(1.0)
            self.rotate_stage2_max_theta_step_rad = np.deg2rad(5.0)
            self.visible_objects = {"A": True}
            self.current_theta = np.deg2rad(10.0)
            self.moved_to = None
            self.state_updates = []

        def _refresh_rotate_discovery_from_current_view(self):
            return {}

        def _get_rotate_visible_target_yaw_error(self, object_key, camera_pose=None, camera_spec=None):
            return {"yaw_error_rad": np.deg2rad(28.0)}

        def _get_current_scan_camera_theta(self, camera_name=None):
            return self.current_theta

        def _set_rotate_subtask_state(self, **kwargs):
            self.state_updates.append(dict(kwargs))

        def _move_scan_camera_to_theta(self, theta_rad, **kwargs):
            self.moved_to = float(theta_rad)
            self.current_theta = float(theta_rad)
            return True

    DummyTask._wrap_to_pi = staticmethod(ExtractedBase._wrap_to_pi)
    task = DummyTask()
    task._fine_center_rotate_registry_target(
        object_key="A",
        subtask_idx=1,
        target_keys=["A"],
        action_target_keys=["A"],
        scan_r=0.62,
        scan_z=0.88,
    )

    expected_theta = task._wrap_to_pi(np.deg2rad(38.0))
    assert np.isclose(task.moved_to, expected_theta)
    assert np.isclose(task.state_updates[-1]["camera_target_theta"], expected_theta)


def test_stage1_stops_scanning_left_after_left_midpoint_is_reached():
    ExtractedBase = _extract_base_methods(
        ["_reacquire_rotate_target_from_history", "search_and_focus_rotate_subtask"]
    )

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.current_subtask_idx = 0
            self.rotate_stage1_theta_unit_rad = 0.5
            self.rotate_scan_order = "left_to_right"
            self.rotate_object_theta_half_rad = 1.5
            self.current_theta = 0.0
            self.moves = []

        def _get_rotate_subtask_def(self, subtask_idx):
            return {"search_target_keys": ["A"], "action_target_keys": ["A"]}

        def begin_rotate_subtask(self, subtask_idx):
            self.current_subtask_idx = int(subtask_idx)

        def _refresh_rotate_discovery_from_current_view(self):
            return {}

        def _get_rotate_target_key(self, candidate_keys, visible_only=False):
            return None

        def _get_rotate_table_edge_theta_limit(self, side):
            return 1.5 if str(side).lower() == "left" else -1.5

        def _get_scan_camera_pose(self):
            return object()

        def _get_scan_camera_runtime_spec(self):
            return object()

        def _is_rotate_table_edge_visible_in_current_view(self, side, camera_pose=None, camera_spec=None):
            side = str(side).lower()
            if side == "left":
                return self.current_theta >= 1.0 - 1e-8, {"side": side, "mode": "midpoint"}
            return self.current_theta <= -0.5 + 1e-8, {"side": side, "mode": "edge_pair"}

        def _get_current_scan_camera_theta(self, camera_name=None):
            return self.current_theta

        def _set_rotate_subtask_state(self, **kwargs):
            return None

        def _move_scan_camera_to_theta(self, theta_rad, **kwargs):
            self.current_theta = float(theta_rad)
            self.moves.append(float(theta_rad))
            return True

        def _fine_center_rotate_registry_target(self, *args, **kwargs):
            raise AssertionError("No target should be found in this test")

    task = DummyTask()
    result = task.search_and_focus_rotate_subtask(
        1,
        scan_r=0.62,
        scan_z=0.88,
        joint_name_prefer="astribot_torso_joint_2",
    )

    assert result is None
    assert np.allclose(task.moves[:2], [0.5, 1.0])
    assert np.allclose(task.moves[2:], [-0.5])
    assert max(task.moves) <= 1.0 + 1e-8


def test_stage1_direction_switch_uses_integer_multiple_of_unit_angle():
    ExtractedBase = _extract_base_methods(
        ["_reacquire_rotate_target_from_history", "search_and_focus_rotate_subtask"]
    )

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.current_subtask_idx = 0
            self.rotate_stage1_theta_unit_rad = np.deg2rad(45.0)
            self.rotate_scan_order = "right_to_left"
            self.rotate_object_theta_half_rad = np.deg2rad(180.0)
            self.current_theta = np.deg2rad(-50.0)
            self.moves = []
            self.edge_query_count = 0

        def _get_rotate_subtask_def(self, subtask_idx):
            return {"search_target_keys": ["B"], "action_target_keys": ["B"]}

        def begin_rotate_subtask(self, subtask_idx):
            self.current_subtask_idx = int(subtask_idx)

        def _refresh_rotate_discovery_from_current_view(self):
            return {}

        def _get_rotate_target_key(self, candidate_keys, visible_only=False):
            return None

        def _get_rotate_table_edge_theta_limit(self, side):
            return np.deg2rad(170.0) if str(side).lower() == "left" else np.deg2rad(-170.0)

        def _get_scan_camera_pose(self):
            return object()

        def _get_scan_camera_runtime_spec(self):
            return object()

        def _is_rotate_table_edge_visible_in_current_view(self, side, camera_pose=None, camera_spec=None):
            side = str(side).lower()
            if self.edge_query_count < 2:
                self.edge_query_count += 1
                if side == "right":
                    return True, {"side": side, "mode": "midpoint"}
                return False, {"side": side, "mode": "edge_pair"}
            return True, {"side": side, "mode": "edge_pair"}

        def _get_current_scan_camera_theta(self, camera_name=None):
            return self.current_theta

        def _set_rotate_subtask_state(self, **kwargs):
            return None

        def _move_scan_camera_to_theta(self, theta_rad, **kwargs):
            self.current_theta = float(theta_rad)
            self.moves.append(float(theta_rad))
            return True

        def _fine_center_rotate_registry_target(self, *args, **kwargs):
            raise AssertionError("No target should be found in this test")

    task = DummyTask()
    result = task.search_and_focus_rotate_subtask(
        1,
        scan_r=0.62,
        scan_z=0.88,
        joint_name_prefer="astribot_torso_joint_2",
    )

    assert result is None
    assert len(task.moves) == 1
    assert np.isclose(task.moves[0], np.deg2rad(40.0))


def test_history_seen_target_starts_from_stage2_before_any_stage1_scan():
    ExtractedBase = _extract_base_methods(
        ["_wrap_to_pi", "_reacquire_rotate_target_from_history", "search_and_focus_rotate_subtask"]
    )

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.current_subtask_idx = 0
            self.current_theta = 0.0
            self.visible_objects = {"A": False}
            self.discovered_objects = {
                "A": {
                    "discovered": True,
                    "last_world_point": [1.0, 1.0, 0.88],
                }
            }
            self.moves = []
            self.state_updates = []
            self.reacquired = False

        def _get_rotate_subtask_def(self, subtask_idx):
            return {"search_target_keys": ["A"], "action_target_keys": ["A"]}

        def begin_rotate_subtask(self, subtask_idx):
            self.current_subtask_idx = int(subtask_idx)

        def _refresh_rotate_discovery_from_current_view(self):
            if self.reacquired:
                self.visible_objects["A"] = True
            return {}

        def _get_rotate_target_key(self, candidate_keys, visible_only=False):
            if visible_only:
                return "A" if self.visible_objects.get("A", False) else None
            return "A" if self.discovered_objects["A"]["discovered"] else None

        def _set_rotate_subtask_state(self, **kwargs):
            self.state_updates.append(dict(kwargs))

        def _move_scan_camera_to_theta(self, theta_rad, **kwargs):
            self.moves.append(float(theta_rad))
            self.current_theta = float(theta_rad)
            self.reacquired = True
            return True

        def _get_current_scan_camera_theta(self, camera_name=None):
            return self.current_theta

        def _fine_center_rotate_registry_target(self, object_key, **kwargs):
            return f"focused:{object_key}"

    DummyTask._wrap_to_pi = staticmethod(ExtractedBase._wrap_to_pi)
    task = DummyTask()
    result = task.search_and_focus_rotate_subtask(
        1,
        scan_r=0.62,
        scan_z=0.88,
        joint_name_prefer="astribot_torso_joint_2",
    )

    assert result == "focused:A"
    assert len(task.moves) == 1
    assert task.state_updates[0]["stage"] == 2
    assert np.isclose(task.moves[0], np.deg2rad(45.0))


def test_waist_heading_deg_is_relative_to_reference_joint_state():
    ExtractedBase = _extract_base_methods(["_wrap_to_pi", "_get_current_rotate_waist_heading_deg"])

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.rotate_waist_heading_joint_index = 1
            self.rotate_waist_heading_reference_rad = np.deg2rad(12.0)
            self.robot = type("DummyRobot", (), {"torso_homestate": [0.0, 0.0]})()

        def _get_torso_joint_state_now(self):
            return np.array([0.0, np.deg2rad(27.0)], dtype=np.float64)

        def _get_rotate_waist_heading_joint_index(self):
            return 1

    DummyTask._wrap_to_pi = staticmethod(ExtractedBase._wrap_to_pi)
    task = DummyTask()
    assert np.isclose(task._get_current_rotate_waist_heading_deg(), 15.0)
