import ast
import importlib.util
import math
import sys
import textwrap
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_TASK_PATH = REPO_ROOT / "envs" / "_base_task.py"
CAMERA_VIS_PATH = REPO_ROOT / "envs" / "utils" / "camera_visibility.py"
PUT_BLOCK_ON_PATH = REPO_ROOT / "envs" / "put_block_on.py"
PUT_BOTTLE_ROTATE_HEAD_PATH = REPO_ROOT / "envs" / "put_bottle_on_cabinet_rotate_and_head.py"
FAN_DOUBLE_UTILS_PATH = REPO_ROOT / "envs" / "_fan_double_task_utils.py"
PUT_BLOCK_TARGET_FAN_DOUBLE_BASE_PATH = REPO_ROOT / "envs" / "_put_block_target_fan_double_base.py"
BLOCKS_RANKING_RGB_FAN_DOUBLE_PATH = REPO_ROOT / "envs" / "blocks_ranking_rgb_fan_double.py"
BLOCKS_RANKING_SIZE_FAN_DOUBLE_PATH = REPO_ROOT / "envs" / "blocks_ranking_size_fan_double.py"
DEMO_CLEAN_FAN_DOUBLE_CONFIG_PATH = REPO_ROOT / "task_config" / "demo_clean_fan_double.yml"
DEMO_RANDOMIZED_FAN_DOUBLE_CONFIG_PATH = REPO_ROOT / "task_config" / "demo_randomized_fan_double.yml"


class _DummyEuler:
    @staticmethod
    def quat2euler(quat, axes="sxyz"):
        assert axes == "sxyz"
        qw, qx, qy, qz = np.array(quat, dtype=np.float64).reshape(4)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def euler2quat(ai, aj, ak, axes="sxyz"):
        assert axes == "sxyz"
        cy = math.cos(float(ak) * 0.5)
        sy = math.sin(float(ak) * 0.5)
        cp = math.cos(float(aj) * 0.5)
        sp = math.sin(float(aj) * 0.5)
        cr = math.cos(float(ai) * 0.5)
        sr = math.sin(float(ai) * 0.5)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )


class _DummyT3D:
    euler = _DummyEuler()


class _DummyQuaternions:
    @staticmethod
    def quat2mat(quat):
        qw, qx, qy, qz = np.array(quat, dtype=np.float64).reshape(4)
        return np.array(
            [
                [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
                [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
                [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def mat2quat(mat):
        m = np.array(mat, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        quat = np.array([qw, qx, qy, qz], dtype=np.float64)
        quat /= np.linalg.norm(quat)
        return quat


_DummyT3D.quaternions = _DummyQuaternions()


_dummy_t3d_module = types.SimpleNamespace(
    quaternions=types.SimpleNamespace(quat2mat=_DummyQuaternions.quat2mat, mat2quat=_DummyQuaternions.mat2quat),
    euler=types.SimpleNamespace(euler2quat=_DummyEuler.euler2quat, quat2euler=_DummyEuler.quat2euler),
)
sys.modules.setdefault("transforms3d", _dummy_t3d_module)


CAMERA_VIS_SPEC = importlib.util.spec_from_file_location("camera_visibility_module", CAMERA_VIS_PATH)
camera_visibility = importlib.util.module_from_spec(CAMERA_VIS_SPEC)
assert CAMERA_VIS_SPEC.loader is not None
CAMERA_VIS_SPEC.loader.exec_module(camera_visibility)


def _extract_class_methods(source_path, class_name, method_names, extra_namespace=None):
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)

    segments = []
    found_names = set()
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name in method_names:
            segment = ast.get_source_segment(source, node)
            assert segment is not None, node.name
            segments.append(segment)
            found_names.add(node.name)

    missing = sorted(set(method_names) - found_names)
    assert not missing, f"Missing {class_name} methods: {missing}"

    namespace = {
        "np": np,
        "t3d": _DummyT3D(),
        "is_world_point_in_camera_fov": camera_visibility.is_world_point_in_camera_fov,
        "image_u_to_yaw_error_rad": camera_visibility.image_u_to_yaw_error_rad,
    }
    if extra_namespace is not None:
        namespace.update(extra_namespace)
    class_body = "\n\n".join(textwrap.indent(segment, "    ") for segment in segments)
    class_source = "class ExtractedBase:\n" + class_body + "\n"
    exec(class_source, namespace)
    return namespace["ExtractedBase"]


def _extract_base_methods(method_names):
    return _extract_class_methods(BASE_TASK_PATH, "Base_Task", method_names)


def _extract_module_functions(source_path, function_names, extra_namespace=None):
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    segments = []
    found_names = set()
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            segment = ast.get_source_segment(source, node)
            assert segment is not None, node.name
            segments.append(segment)
            found_names.add(node.name)

    missing = sorted(set(function_names) - found_names)
    assert not missing, f"Missing module functions in {source_path.name}: {missing}"

    namespace = {
        "np": np,
        "t3d": _DummyT3D(),
    }
    if extra_namespace is not None:
        namespace.update(extra_namespace)
    exec("\n\n".join(segments), namespace)
    return namespace


def _extract_class_constant(source_path, class_name, constant_name):
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == constant_name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {class_name}.{constant_name}")


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


def test_rotate_and_head_stage1_uses_fixed_discrete_search_state_machine():
    ExtractedBase = _extract_base_methods(
        [
            "_normalize_rotate_search_layer",
            "_get_rotate_discrete_search_states",
            "_set_rotate_search_cursor",
            "_ensure_rotate_search_cursor_initialized",
            "_advance_rotate_search_cursor",
            "_sync_rotate_search_cursor_from_current_view",
            "_get_head_joint2_index",
            "_get_head_home_target",
            "_get_rotate_search_head_target",
            "_move_head_to_rotate_search_layer",
            "_run_rotate_and_head_stage1_search_state",
            "search_and_focus_rotate_and_head_subtask",
        ]
    )

    class DummyJoint:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.current_subtask_idx = 0
            self.current_theta = 0.0
            self.current_head = np.array([0.0, 1.22], dtype=np.float64)
            self.rotate_table_shape = "fan_double"
            self.rotate_head_joint2_name = "astribot_head_joint_2"
            self.rotate_stage1_lower_head_joint2_rad = 1.22
            self.rotate_stage1_upper_head_joint2_rad = 0.8
            self.rotate_stage1_head_settle_steps = 12
            self.rotate_stage1_theta_unit_rad = np.deg2rad(45.0)
            self.search_cursor_state = None
            self.search_cursor_state_index = None
            self.search_cursor_theta = np.nan
            self.search_cursor_layer = None
            self.search_cursor_state_complete = False
            self.search_cursor_boundary_reached = False
            self.robot = type(
                "DummyRobot",
                (),
                {
                    "head_homestate": [0.0, 1.22],
                    "head_joints": [DummyJoint("astribot_head_joint_1"), DummyJoint("astribot_head_joint_2")],
                },
            )()
            self.head_moves = []
            self.torso_moves = []

        def _get_rotate_subtask_def(self, subtask_idx):
            return {
                "search_target_keys": ["A"],
                "action_target_keys": ["A"],
                "allow_stage2_from_memory": True,
            }

        def begin_rotate_subtask(self, subtask_idx):
            self.current_subtask_idx = int(subtask_idx)

        def _refresh_rotate_discovery_from_current_view(self):
            return {}

        def _get_rotate_target_key(self, candidate_keys, visible_only=False):
            return None

        def _clip_head_target_to_limits(self, target_rad, default_now=None):
            return np.array(target_rad, dtype=np.float64).reshape(-1)

        def _get_head_joint_state_now(self):
            return np.array(self.current_head, dtype=np.float64)

        def move_head_to(self, target_rad, settle_steps=None, save_freq=-1):
            target = np.array(target_rad, dtype=np.float64).reshape(-1)
            self.current_head = target
            self.head_moves.append((float(self.current_theta), float(target[1])))
            return True

        def _get_current_scan_camera_theta(self, camera_name=None):
            return self.current_theta

        def _get_rotate_table_edge_theta_limit(self, side):
            return np.deg2rad(45.0) if str(side).lower() == "left" else np.deg2rad(-45.0)

        def _get_scan_camera_pose(self):
            return object()

        def _get_scan_camera_runtime_spec(self):
            return object()

        def _is_rotate_table_edge_visible_in_current_view(self, side, camera_pose=None, camera_spec=None):
            side = str(side).lower()
            if side == "left":
                return self.current_theta >= np.deg2rad(45.0) - 1e-8, {"side": side}
            return self.current_theta <= np.deg2rad(-45.0) + 1e-8, {"side": side}

        def _set_rotate_subtask_state(self, **kwargs):
            return None

        def _move_scan_camera_to_theta(self, theta_rad, **kwargs):
            self.current_theta = float(theta_rad)
            self.torso_moves.append(float(theta_rad))
            return True

        def _align_rotate_registry_target_with_torso_and_head_joint2(self, *args, **kwargs):
            raise AssertionError("No target should be aligned in this test")

    DummyTask._normalize_rotate_search_layer = staticmethod(ExtractedBase._normalize_rotate_search_layer)
    task = DummyTask()
    result = task.search_and_focus_rotate_and_head_subtask(
        1,
        scan_r=0.62,
        scan_z=0.88,
        joint_name_prefer="astribot_torso_joint_2",
    )

    assert result is None
    assert np.allclose(
        task.torso_moves,
        [
            np.deg2rad(45.0),
            0.0,
            np.deg2rad(-45.0),
            0.0,
            np.deg2rad(45.0),
        ],
    )
    assert len(task.head_moves) == 7
    assert np.allclose(task.head_moves[0], [0.0, 1.22])
    assert np.allclose(task.head_moves[4], [np.deg2rad(-45.0), 0.8])
    assert np.allclose(task.head_moves[-1], [0.0, 0.8])


def test_rotate_and_head_history_hits_stage2_before_any_stage1_scan():
    ExtractedBase = _extract_base_methods(
        [
            "_normalize_rotate_search_layer",
            "_get_rotate_discrete_search_states",
            "_set_rotate_search_cursor",
            "_ensure_rotate_search_cursor_initialized",
            "search_and_focus_rotate_and_head_subtask",
        ]
    )

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.current_subtask_idx = 0
            self.rotate_table_shape = "fan_double"
            self.discovered_objects = {"A": {"discovered": True}}
            self.visible_objects = {"A": False}
            self.search_cursor_state = None
            self.search_cursor_state_index = None
            self.search_cursor_theta = np.nan
            self.search_cursor_layer = None
            self.search_cursor_state_complete = False
            self.search_cursor_boundary_reached = False
            self.focus_calls = []

        def _get_rotate_subtask_def(self, subtask_idx):
            return {
                "search_target_keys": ["A"],
                "action_target_keys": ["A"],
                "allow_stage2_from_memory": True,
            }

        def begin_rotate_subtask(self, subtask_idx):
            self.current_subtask_idx = int(subtask_idx)

        def _refresh_rotate_discovery_from_current_view(self):
            return {}

        def _get_rotate_target_key(self, candidate_keys, visible_only=False):
            if visible_only:
                return None
            return "A"

        def _focus_rotate_registry_target_with_fixed_head(self, object_key, **kwargs):
            self.focus_calls.append((str(object_key), bool(kwargs.get("prefer_history_world_point", False))))
            return f"focused:{object_key}"

        def _get_current_scan_camera_theta(self, camera_name=None):
            return 0.0

        def _set_rotate_subtask_state(self, **kwargs):
            return None

        def _move_scan_camera_to_theta(self, theta_rad, **kwargs):
            raise AssertionError("Stage1 torso scan should be skipped on historical memory hit")

    DummyTask._normalize_rotate_search_layer = staticmethod(ExtractedBase._normalize_rotate_search_layer)
    task = DummyTask()
    result = task.search_and_focus_rotate_and_head_subtask(
        1,
        scan_r=0.62,
        scan_z=0.88,
        joint_name_prefer="astribot_torso_joint_2",
    )

    assert result == "focused:A"
    assert task.focus_calls == [("A", True)]


def test_stage2_aligns_with_torso_and_fixed_head_preset():
    ExtractedBase = _extract_base_methods(
        [
            "_wrap_to_pi",
            "_compute_rotate_target_theta_from_world_point",
            "_normalize_rotate_search_layer",
            "_get_rotate_object_layer",
            "_get_head_joint2_index",
            "_get_head_home_target",
            "_get_rotate_search_head_target",
            "_move_head_to_rotate_search_layer",
            "_sync_rotate_search_cursor_from_current_view",
            "_focus_rotate_registry_target_with_fixed_head",
            "_align_rotate_registry_target_with_torso_and_head_joint2",
        ]
    )

    class DummyJoint:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    class DummyObject:
        pass

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.robot_root_xy = np.array([0.0, 0.0], dtype=np.float64)
            self.robot_yaw = 0.0
            self.current_theta = 0.0
            self.current_head = np.array([0.0, 1.22], dtype=np.float64)
            self.face_calls = []
            self.head_moves = []
            self.state_updates = []
            self.visible_objects = {"A": False}
            self.discovered_objects = {}
            self.object_registry = {"A": DummyObject()}
            self.object_layers = {"A": "upper"}
            self.search_cursor_layer = None
            self.search_cursor_theta = np.nan
            self.robot = type(
                "DummyRobot",
                (),
                {
                    "head_homestate": [0.0, 1.22],
                    "head_joints": [DummyJoint("astribot_head_joint_1"), DummyJoint("astribot_head_joint_2")],
                },
            )()
            self.rotate_stage1_lower_head_joint2_rad = 1.22
            self.rotate_stage1_upper_head_joint2_rad = 0.8

        def _resolve_rotate_registry_object(self, object_key):
            return self.object_registry[str(object_key)]

        def _resolve_object_world_point(self, obj, **kwargs):
            return np.array([1.0, 1.0, 0.9], dtype=np.float64)

        def _clip_head_target_to_limits(self, target_rad, default_now=None):
            return np.array(target_rad, dtype=np.float64).reshape(-1)

        def _get_head_joint_state_now(self):
            return np.array(self.current_head, dtype=np.float64)

        def _set_rotate_subtask_state(self, **kwargs):
            self.state_updates.append(dict(kwargs))

        def face_world_point_with_torso(self, world_point, **kwargs):
            self.face_calls.append(
                {
                    "world_point": np.array(world_point, dtype=np.float64).tolist(),
                    "kwargs": dict(kwargs),
                }
            )
            self.current_theta = np.deg2rad(45.0)
            return True

        def _refresh_rotate_discovery_from_current_view(self):
            self.visible_objects["A"] = True
            return {}

        def _get_current_scan_camera_theta(self, camera_name=None):
            return self.current_theta

        def move_head_to(self, target_rad, settle_steps=None, save_freq=-1):
            target = np.array(target_rad, dtype=np.float64).reshape(-1)
            self.current_head = target
            self.head_moves.append(target.tolist())
            return True

    DummyTask._wrap_to_pi = staticmethod(ExtractedBase._wrap_to_pi)
    DummyTask._normalize_rotate_search_layer = staticmethod(ExtractedBase._normalize_rotate_search_layer)
    task = DummyTask()
    result = task._align_rotate_registry_target_with_torso_and_head_joint2(
        "A",
        subtask_idx=2,
        target_keys=["A"],
        action_target_keys=["A"],
        joint_name_prefer="astribot_torso_joint_2",
    )

    assert result == "A"
    assert task.face_calls[0]["world_point"] == [1.0, 1.0, 0.9]
    assert task.face_calls[0]["kwargs"]["yaw_deadband_rad"] == 0.0
    assert task.face_calls[0]["kwargs"]["yaw_hysteresis_rad"] == 0.0
    assert task.head_moves[-1][1] == 0.8
    assert task.state_updates[0]["stage"] == 2
    assert np.isclose(task.state_updates[0]["camera_target_theta"], np.deg2rad(45.0))


def test_move_scan_camera_to_theta_uses_zero_deadband_for_exact_state_transition():
    ExtractedBase = _extract_class_methods(
        BASE_TASK_PATH,
        "Base_Task",
        [
            "_get_rotate_scan_world_point",
            "_move_scan_camera_to_theta",
        ],
        extra_namespace={
            "place_point_cyl": lambda point_cyl, robot_root_xy, robot_yaw_rad, ret="list": [
                float(robot_root_xy[0]) + float(point_cyl[0]) * math.cos(float(robot_yaw_rad) + float(point_cyl[1])),
                float(robot_root_xy[1]) + float(point_cyl[0]) * math.sin(float(robot_yaw_rad) + float(point_cyl[1])),
                float(point_cyl[2]),
            ],
        },
    )

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.robot_root_xy = [0.0, 0.0]
            self.robot_yaw = 0.0
            self.calls = []

        def face_world_point_with_torso(self, world_point, **kwargs):
            self.calls.append(
                {
                    "world_point": np.array(world_point, dtype=np.float64).tolist(),
                    "kwargs": dict(kwargs),
                }
            )
            return True

    task = DummyTask()
    task._move_scan_camera_to_theta(
        np.deg2rad(45.0),
        scan_r=0.62,
        scan_z=0.88,
        joint_name_prefer="astribot_torso_joint_2",
    )

    assert len(task.calls) == 1
    assert np.allclose(task.calls[0]["world_point"], [0.62 * np.cos(np.deg2rad(45.0)), 0.62 * np.sin(np.deg2rad(45.0)), 0.88])
    assert task.calls[0]["kwargs"]["yaw_deadband_rad"] == 0.0
    assert task.calls[0]["kwargs"]["yaw_hysteresis_rad"] == 0.0


def test_fan_double_collision_cuboids_include_upper_surface_and_side_columns():
    ExtractedBase = _extract_base_methods(
        [
            "_pose7_from_xyzyaw",
            "_build_fan_surface_collision_cuboids",
            "_build_fan_double_table_collision_cuboids",
        ]
    )

    class DummyTask(ExtractedBase):
        def __init__(self):
            self.rotate_table_shape = "fan_double"
            self.rotate_table_center_xy = np.array([0.0, -0.45], dtype=np.float64)
            self.rotate_table_top_z = 0.74
            self.rotate_table_thickness = 0.03
            self.rotate_fan_theta_start_world_rad = np.deg2rad(15.0)
            self.rotate_fan_theta_end_world_rad = np.deg2rad(165.0)
            self.rotate_fan_double_lower_inner_radius = 0.30
            self.rotate_fan_double_lower_outer_radius = 0.80
            self.rotate_fan_double_upper_inner_radius = 0.60
            self.rotate_fan_double_upper_outer_radius = 0.80
            self.rotate_fan_double_layer_gap = 0.40
            self.rotate_fan_double_upper_theta_start_world_rad = np.deg2rad(60.0)
            self.rotate_fan_double_upper_theta_end_world_rad = np.deg2rad(120.0)
            self.rotate_fan_double_support_theta_world_rad = np.deg2rad(50.0)
            self.rotate_fan_double_upper_collision_under_padding = 0.08

    task = DummyTask()
    cuboids = task._build_fan_double_table_collision_cuboids()

    surface = [item for item in cuboids if item["name"].startswith("fan_double_upper_surface")]
    columns = [item for item in cuboids if item["name"].startswith("fan_double_side_column")]
    assert len(surface) > 0
    assert len(columns) == 1
    assert all(item["pose"][2] > task.rotate_table_top_z for item in surface)
    assert all(np.isclose(item["dims"][2], task.rotate_table_thickness + 0.08) for item in surface)
    assert all(np.isclose(item["dims"][0], task.rotate_table_thickness) for item in columns)


def test_demo_clean_fan_double_keeps_only_scene_level_radius_params():
    if not DEMO_CLEAN_FAN_DOUBLE_CONFIG_PATH.exists():
        return
    config_text = DEMO_CLEAN_FAN_DOUBLE_CONFIG_PATH.read_text(encoding="utf-8")
    config_keys = {
        line.split(":", 1)[0].strip()
        for line in config_text.splitlines()
        if ":" in line and not line.lstrip().startswith("#")
    }

    assert "fan_outer_radius" not in config_keys
    assert "fan_inner_radius" not in config_keys
    assert "block_count" not in {key.lower() for key in config_keys}
    assert not any(key.lower().startswith("plate_") for key in config_keys)
    assert {
        "fan_double_lower_outer_radius",
        "fan_double_lower_inner_radius",
        "fan_double_upper_outer_radius",
        "fan_double_upper_inner_radius",
    }.issubset(config_keys)


def test_demo_randomized_fan_double_uses_scene_params_and_astribot_randomization():
    if not DEMO_RANDOMIZED_FAN_DOUBLE_CONFIG_PATH.exists():
        return
    config_text = DEMO_RANDOMIZED_FAN_DOUBLE_CONFIG_PATH.read_text(encoding="utf-8")
    config_keys = {
        line.split(":", 1)[0].strip()
        for line in config_text.splitlines()
        if ":" in line and not line.lstrip().startswith("#")
    }

    assert "table_shape: fan_double" in config_text
    assert "embodiment: [astribot_texture]" in config_text
    assert "head_camera_type: Large" in config_text
    assert "wrist_camera_type: D435" in config_text
    assert "random_background: true" in config_text
    assert "cluttered_table: true" in config_text
    assert "random_light: true" in config_text
    assert "fan_outer_radius" not in config_keys
    assert "fan_inner_radius" not in config_keys
    assert "block_count" not in {key.lower() for key in config_keys}
    assert not any(key.lower().startswith("plate_") for key in config_keys)
    assert {
        "fan_double_lower_outer_radius",
        "fan_double_lower_inner_radius",
        "fan_double_upper_outer_radius",
        "fan_double_upper_inner_radius",
    }.issubset(config_keys)


def test_put_block_on_exposes_block_count_and_plate_constants():
    source = PUT_BLOCK_ON_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "put_block_on")
    assigned_names = set()
    for node in class_node.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assigned_names.add(target.id)

    assert "BLOCK_COUNT" in assigned_names
    assert {
        "PLATE_MODEL_ID",
        "PLATE_LAYER",
        "PLATE_LAYER_SPECS",
        "BLOCK_LAYER_SEQUENCE",
        "BLOCK_LAYER_SPECS",
        "PLACE_PLATE_UPPER_HEAD_JOINT2_TARGET",
        "PLACE_PLATE_LOWER_HEAD_JOINT2_TARGET",
        "REQUIRE_PLATE_VISIBLE_BEFORE_PLACE",
        "HEAD_RESET_SAVE_FREQ",
        "RETURN_TO_HOMESTATE_AFTER_PLACE",
    }.issubset(assigned_names)


def test_put_block_on_validates_explicit_block_layers():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_get_block_count",
            "_normalize_layer",
            "_get_block_layers",
        ],
    )

    class DummyTask(ExtractedTask):
        BLOCK_COUNT = 2
        BLOCK_LAYER_SEQUENCE = ("lower", "upper")

    assert DummyTask()._get_block_layers() == ("lower", "upper")

    class BadLengthTask(ExtractedTask):
        BLOCK_COUNT = 2
        BLOCK_LAYER_SEQUENCE = ("lower",)

    try:
        BadLengthTask()._get_block_layers()
    except ValueError as exc:
        assert "BLOCK_LAYER_SEQUENCE length" in str(exc)
    else:
        raise AssertionError("Expected length mismatch to raise ValueError")

    class BadLayerTask(ExtractedTask):
        BLOCK_COUNT = 1
        BLOCK_LAYER_SEQUENCE = ("middle",)

    try:
        BadLayerTask()._get_block_layers()
    except ValueError as exc:
        assert "lower" in str(exc) and "upper" in str(exc)
    else:
        raise AssertionError("Expected invalid layer to raise ValueError")


def test_put_block_on_layer_specs_drive_block_and_plate_heights():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_normalize_layer",
            "_normalize_closed_range",
            "_get_block_spawn_range_override",
            "_get_layer_spec",
            "_get_plate_layer",
            "_get_plate_layer_spec",
        ],
        extra_namespace={
            "rotate_theta_half": lambda self: np.deg2rad(90.0),
        },
    )

    class DummyTask(ExtractedTask):
        BLOCK_LAYER_SPECS = {
            "lower": {"inner_margin": 0.12, "outer_margin": 0.18, "max_cyl_r": 0.60, "theta_shrink": 0.92},
            "upper": {"inner_margin": 0.04, "outer_margin": 0.06, "max_cyl_r": 0.70, "theta_shrink": 0.92},
        }
        PLATE_LAYER = "upper"
        PLATE_LAYER_SPECS = {
            "lower": {"r": 0.55, "theta_deg": 0.0, "z_offset": 0.01, "qpos": [1, 0, 0, 0], "scale": [1, 1, 1]},
            "upper": {"r": 0.70, "theta_deg": 5.0, "z_offset": 0.02, "qpos": [0.5, 0.5, 0.5, 0.5], "scale": [2, 2, 2]},
        }

        def __init__(self):
            self.rotate_fan_double_lower_inner_radius = 0.30
            self.rotate_fan_double_lower_outer_radius = 0.90
            self.rotate_fan_double_upper_inner_radius = 0.60
            self.rotate_fan_double_upper_outer_radius = 0.80
            self.rotate_table_top_z = 0.74
            self.rotate_fan_double_layer_gap = 0.35

    task = DummyTask()
    lower = task._get_layer_spec("lower")
    upper = task._get_layer_spec("upper")
    plate_lower = task._get_plate_layer_spec("lower")
    plate_upper = task._get_plate_layer_spec("upper")

    assert np.isclose(lower["rlim"][1], 0.60)
    assert np.isclose(upper["rlim"][1], 0.70)
    assert np.isclose(plate_lower["z"], 0.75)
    assert np.isclose(plate_upper["z"], 1.11)
    assert plate_upper["scale"] == [2, 2, 2]


def test_put_block_on_runtime_spawn_range_overrides_are_applied_per_layer():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_apply_block_spawn_overrides_from_kwargs",
        ],
    )

    class DummyTask(ExtractedTask):
        pass

    task = DummyTask()
    task._apply_block_spawn_overrides_from_kwargs(
        {
            "block_spawn_lower_r_range": [0.42, 0.58],
            "block_spawn_lower_theta_deg_range": [-60.0, 20.0],
            "block_spawn_upper_r_range": [0.66, 0.69],
            "block_spawn_upper_theta_deg_range": [-10.0, 10.0],
        }
    )

    assert task.block_spawn_lower_r_range == [0.42, 0.58]
    assert task.block_spawn_lower_theta_deg_range == [-60.0, 20.0]
    assert task.block_spawn_upper_r_range == [0.66, 0.69]
    assert task.block_spawn_upper_theta_deg_range == [-10.0, 10.0]


def test_put_block_on_layer_specs_support_direct_r_and_theta_ranges():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_normalize_layer",
            "_normalize_closed_range",
            "_get_block_spawn_range_override",
            "_get_layer_spec",
        ],
        extra_namespace={
            "rotate_theta_half": lambda self: np.deg2rad(90.0),
        },
    )

    class DummyTask(ExtractedTask):
        BLOCK_LAYER_SPECS = {
            "lower": {"r_range": [0.46, 0.58], "theta_deg_range": [-35.0, 25.0]},
            "upper": {"inner_margin": 0.04, "outer_margin": 0.06, "max_cyl_r": 0.70, "theta_shrink": 0.92},
        }

        def __init__(self):
            self.rotate_fan_double_lower_inner_radius = 0.30
            self.rotate_fan_double_lower_outer_radius = 0.90
            self.rotate_fan_double_upper_inner_radius = 0.60
            self.rotate_fan_double_upper_outer_radius = 0.80
            self.rotate_table_top_z = 0.74
            self.rotate_fan_double_layer_gap = 0.35
            self.robot_yaw = 0.0
            self.block_spawn_upper_r_range = [0.66, 0.69]
            self.block_spawn_upper_theta_deg_range = [12.0, -8.0]

    task = DummyTask()
    lower = task._get_layer_spec("lower")
    upper = task._get_layer_spec("upper")

    assert np.allclose(lower["rlim"], [0.46, 0.58])
    assert np.allclose(lower["thetalim"], np.deg2rad([-35.0, 25.0]))
    assert np.allclose(upper["rlim"], [0.66, 0.69])
    assert np.allclose(upper["thetalim"], np.deg2rad([-8.0, 12.0]))


def test_put_block_on_same_layer_blocks_avoid_plate_anchor():
    class DummyPose:
        def __init__(self, p):
            self.p = np.array(p, dtype=np.float64)

    def fake_place_pose_cyl(pose_spec, robot_root_xy, robot_yaw_rad, ret="pose", quat_frame="world"):
        assert ret == "pose"
        r, theta_rad, z = [float(value) for value in pose_spec[:3]]
        return DummyPose(
            [
                float(robot_root_xy[0]) + r * math.cos(float(robot_yaw_rad) + theta_rad),
                float(robot_root_xy[1]) + r * math.sin(float(robot_yaw_rad) + theta_rad),
                z,
            ]
        )

    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_normalize_layer",
            "_normalize_closed_range",
            "_get_block_spawn_range_override",
            "_get_layer_spec",
            "_get_plate_layer",
            "_get_plate_layer_spec",
            "_get_plate_anchor_pose",
            "_get_block_spawn_avoid_pose_lst",
        ],
        extra_namespace={
            "rotate_theta_half": lambda self: np.deg2rad(90.0),
            "place_pose_cyl": fake_place_pose_cyl,
        },
    )

    class DummyTask(ExtractedTask):
        BLOCK_LAYER_SPECS = {
            "lower": {"inner_margin": 0.12, "outer_margin": 0.18, "max_cyl_r": 0.60, "theta_shrink": 0.92},
            "upper": {"inner_margin": 0.04, "outer_margin": 0.06, "max_cyl_r": 0.70, "theta_shrink": 0.92},
        }
        PLATE_LAYER = "upper"
        PLATE_LAYER_SPECS = {
            "lower": {"r": 0.55, "theta_deg": 20.0, "z_offset": 0.0, "qpos": [1, 0, 0, 0], "scale": [1, 1, 1]},
            "upper": {"r": 0.70, "theta_deg": 5.0, "z_offset": 0.0, "qpos": [1, 0, 0, 0], "scale": [1, 1, 1]},
        }

        def __init__(self):
            self.rotate_fan_double_lower_inner_radius = 0.30
            self.rotate_fan_double_lower_outer_radius = 0.90
            self.rotate_fan_double_upper_inner_radius = 0.60
            self.rotate_fan_double_upper_outer_radius = 0.80
            self.rotate_table_top_z = 0.74
            self.rotate_fan_double_layer_gap = 0.35
            self.robot_root_xy = [0.0, 0.0]
            self.robot_yaw = 0.0

    task = DummyTask()
    lower_avoid = task._get_block_spawn_avoid_pose_lst("lower")
    upper_avoid = task._get_block_spawn_avoid_pose_lst("upper")

    assert lower_avoid == []
    assert len(upper_avoid) == 1
    assert np.allclose(upper_avoid[0].p, [0.70 * math.cos(math.radians(5.0)), 0.70 * math.sin(math.radians(5.0)), 1.09])


def test_put_block_on_sample_block_pose_retries_when_plate_overlap_detected():
    class DummyPose:
        def __init__(self, p):
            self.p = np.array(p, dtype=np.float64)

    sampled_poses = [
        DummyPose([0.70, 0.0, 1.09]),
        DummyPose([0.62, 0.18, 1.09]),
    ]
    rand_calls = []

    def fake_rand_pose_cyl(*args, **kwargs):
        rand_calls.append((args, kwargs))
        return sampled_poses[len(rand_calls) - 1]

    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_valid_spacing",
            "_is_valid_block_spawn_pose",
            "_sample_block_pose",
        ],
        extra_namespace={
            "rand_pose_cyl": fake_rand_pose_cyl,
            "world_to_robot": lambda world_pt, robot_root_xy, robot_yaw_rad=0.0: [
                float(np.hypot(world_pt[0] - robot_root_xy[0], world_pt[1] - robot_root_xy[1])),
                float(np.arctan2(world_pt[1] - robot_root_xy[1], world_pt[0] - robot_root_xy[0]) - robot_yaw_rad),
                float(world_pt[2]),
            ],
        },
    )

    class DummyTask(ExtractedTask):
        BLOCK_SPAWN_MIN_DIST_SQ = 0.01
        PLATE_BLOCK_SPAWN_MIN_DIST_SQ = 0.0144
        _valid_spacing = staticmethod(ExtractedTask._valid_spacing)

        def __init__(self):
            self.robot_root_xy = [0.0, 0.0]
            self.robot_yaw = 0.0

        def _get_layer_spec(self, layer_name):
            assert layer_name == "upper"
            return {
                "rlim": [0.60, 0.70],
                "thetalim": [-0.1, 0.1],
                "top_z": 1.09,
            }

    task = DummyTask()
    plate_avoid = [DummyPose([0.70, 0.0, 1.09])]
    sampled = task._sample_block_pose(
        layer_name="upper",
        size=0.02,
        existing_pose_lst=[],
        avoid_pose_lst=plate_avoid,
        avoid_min_dist_sq=task.PLATE_BLOCK_SPAWN_MIN_DIST_SQ,
    )

    assert sampled is sampled_poses[1]
    assert len(rand_calls) == 2


def test_put_block_on_block_count_builds_registry_and_subtasks():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_get_block_count",
            "_configure_rotate_subtask_plan",
        ],
    )

    class DummyTask(ExtractedTask):
        BLOCK_COUNT = 3

        def __init__(self):
            self.blocks = ["block0", "block1", "block2"]
            self.block_keys = ["A0", "A1", "A2"]
            self.plate = "plate"
            self.configured = None

        def configure_rotate_subtask_plan(self, **kwargs):
            self.configured = kwargs

    task = DummyTask()
    assert task._get_block_count() == 3
    task._configure_rotate_subtask_plan()

    assert list(task.configured["object_registry"].keys()) == ["A0", "A1", "A2", "B"]
    subtask_defs = task.configured["subtask_defs"]
    assert len(subtask_defs) == 6
    assert [item["id"] for item in subtask_defs] == [1, 2, 3, 4, 5, 6]
    assert subtask_defs[0]["search_target_keys"] == ["A0", "A1", "A2"]
    assert subtask_defs[0]["action_target_keys"] == ["A0", "A1", "A2"]
    assert subtask_defs[1]["search_target_keys"] == ["B"]
    assert subtask_defs[1]["action_target_keys"] == ["B"]
    assert subtask_defs[0]["allow_stage2_from_memory"] is False
    assert subtask_defs[1]["allow_stage2_from_memory"] is True
    assert subtask_defs[2]["allow_stage2_from_memory"] is False
    assert subtask_defs[3]["allow_stage2_from_memory"] is True
    assert subtask_defs[-1]["next_subtask_id"] == -1


def test_put_block_on_dynamic_subtasks_use_remaining_blocks_and_actual_carried_block():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_prepare_dynamic_pick_subtask",
            "_prepare_dynamic_place_subtask",
        ],
    )

    class DummyTask(ExtractedTask):
        def __init__(self):
            self.subtask_def_map = {
                1: {
                    "search_target_keys": ["A0", "A1"],
                    "action_target_keys": ["A0", "A1"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                },
                2: {
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                },
            }

        def _get_rotate_subtask_def(self, subtask_idx):
            return self.subtask_def_map.get(int(subtask_idx))

    task = DummyTask()
    task._prepare_dynamic_pick_subtask(1, ["A1", "A0"])

    assert task.subtask_def_map[1]["search_target_keys"] == ["A1", "A0"]
    assert task.subtask_def_map[1]["action_target_keys"] == ["A1", "A0"]
    assert task.subtask_def_map[1]["allow_stage2_from_memory"] is False

    task._prepare_dynamic_pick_subtask(1, ["A0"])
    assert task.subtask_def_map[1]["search_target_keys"] == ["A0"]
    assert task.subtask_def_map[1]["allow_stage2_from_memory"] is True

    task._prepare_dynamic_place_subtask(2, "A1")
    assert task.subtask_def_map[2]["search_target_keys"] == ["B"]
    assert task.subtask_def_map[2]["action_target_keys"] == ["A1", "B"]
    assert task.subtask_def_map[2]["required_carried_keys"] == ["A1"]
    assert task.subtask_def_map[2]["allow_stage2_from_memory"] is True


def test_put_block_on_upper_plate_search_keeps_lower_phase_before_raising_head():
    SearchBase = _extract_base_methods(
        [
            "_normalize_rotate_search_layer",
            "_get_rotate_discrete_search_states",
            "_get_rotate_first_upper_search_state_index",
            "_set_rotate_search_cursor",
        ]
    )
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_normalize_layer",
            "_get_subtask_search_target_keys",
            "_get_subtask_upper_search_target_keys",
            "_should_search_lower_before_upper_for_subtask",
            "_has_unfinished_lower_search_phase",
            "_get_place_search_block_key",
            "_is_upper_plate_search_after_lower_pick",
            "_clear_rotate_target_search_history",
            "_prepare_subtask_rotate_search",
            "_prepare_plate_rotate_search",
            "_should_skip_rotate_head_home_reset",
            "_get_subtask_search_layers",
            "_subtask_requires_head_home_reset",
            "_maybe_reset_head_to_home_for_subtask",
        ],
    )

    class DummyTask(ExtractedTask, SearchBase):
        def __init__(self):
            self.block_keys = ["A0", "A1"]
            self.object_registry = {"A0": object(), "A1": object(), "B": object()}
            self.object_layers = {"A0": "lower", "A1": "lower", "B": "upper"}
            self.plate_layer = "upper"
            self.rotate_table_shape = "fan_double"
            self.carried_object_keys = ["A0"]
            self.discovered_objects = {
                "B": {
                    "discovered": True,
                    "visible_now": True,
                    "first_seen_frame": 8,
                    "last_seen_frame": 8,
                    "last_seen_subtask": 1,
                    "last_seen_stage": 2,
                    "last_uv_norm": [0.4, 0.5],
                    "last_world_point": [0.68, 0.02, 1.09],
                }
            }
            self.visible_objects = {"B": True}
            self.subtask_def_map = {
                1: {
                    "search_target_keys": ["A0", "A1"],
                    "action_target_keys": ["A0", "A1"],
                    "required_carried_keys": [],
                },
                2: {
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A0", "B"],
                    "required_carried_keys": ["A0"],
                },
            }
            self.search_cursor_state = None
            self.search_cursor_state_index = None
            self.search_cursor_theta = np.nan
            self.search_cursor_layer = None
            self.search_cursor_state_complete = False
            self.search_cursor_boundary_reached = False
            self.fixed_layer_head_joint2_only = True
            self.HEAD_RESET_SAVE_FREQ = -1
            self.reset_calls = []

        def _get_rotate_subtask_def(self, subtask_idx):
            return self.subtask_def_map.get(int(subtask_idx))

        def _move_head_to_rotate_search_layer(self, layer_name, head_joint2_name=None, settle_steps=None, save_freq=-1):
            self.reset_calls.append(str(layer_name))
            return True

        def _reset_head_to_home_pose(self, settle_steps=None, save_freq=-1):
            self.reset_calls.append("home")
            return True

        def _get_current_scan_camera_theta(self, camera_name=None):
            return 0.0

    DummyTask._normalize_rotate_search_layer = staticmethod(SearchBase._normalize_rotate_search_layer)
    task = DummyTask()
    task._set_rotate_search_cursor(state_idx=0, theta=0.0, layer_name="lower")

    assert task._should_skip_rotate_head_home_reset(2, prev_subtask_idx=1) is True
    assert task._maybe_reset_head_to_home_for_subtask(2, prev_subtask_idx=1) is True
    assert task.reset_calls == ["lower"]

    task._prepare_plate_rotate_search(2)
    assert task.search_cursor_state_index == 0
    assert task.visible_objects["B"] is False
    assert task.discovered_objects["B"]["discovered"] is False

    task.discovered_objects["B"]["discovered"] = True
    task.visible_objects["B"] = True
    task._set_rotate_search_cursor(state_idx=6, theta=0.0, layer_name="upper")

    assert task._should_skip_rotate_head_home_reset(2, prev_subtask_idx=1) is False
    assert task._maybe_reset_head_to_home_for_subtask(2, prev_subtask_idx=1) is True
    assert task.reset_calls == ["lower", "upper"]

    task._prepare_plate_rotate_search(2)
    assert task.search_cursor_state_index == task._get_rotate_first_upper_search_state_index()
    assert task.search_cursor_layer == "upper"


def test_put_block_on_upper_block_pick_search_starts_from_lower_phase():
    SearchBase = _extract_base_methods(
        [
            "_normalize_rotate_search_layer",
            "_get_rotate_discrete_search_states",
            "_get_rotate_first_upper_search_state_index",
            "_set_rotate_search_cursor",
        ]
    )
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_normalize_layer",
            "_get_subtask_search_target_keys",
            "_get_subtask_upper_search_target_keys",
            "_should_search_lower_before_upper_for_subtask",
            "_has_unfinished_lower_search_phase",
            "_clear_rotate_target_search_history",
            "_prepare_subtask_rotate_search",
            "_should_skip_rotate_head_home_reset",
            "_get_subtask_search_layers",
            "_subtask_requires_head_home_reset",
            "_maybe_reset_head_to_home_for_subtask",
        ],
    )

    class DummyTask(ExtractedTask, SearchBase):
        def __init__(self):
            self.block_keys = ["A0", "A1"]
            self.object_registry = {"A0": object(), "A1": object(), "B": object()}
            self.object_layers = {"A0": "upper", "A1": "upper", "B": "lower"}
            self.plate_layer = "lower"
            self.rotate_table_shape = "fan_double"
            self.carried_object_keys = []
            self.discovered_objects = {
                "A0": {
                    "discovered": True,
                    "visible_now": True,
                    "first_seen_frame": 8,
                    "last_seen_frame": 8,
                    "last_seen_subtask": 1,
                    "last_seen_stage": 1,
                    "last_uv_norm": [0.4, 0.5],
                    "last_world_point": [0.68, 0.02, 1.09],
                },
                "A1": {
                    "discovered": True,
                    "visible_now": True,
                    "first_seen_frame": 9,
                    "last_seen_frame": 9,
                    "last_seen_subtask": 1,
                    "last_seen_stage": 1,
                    "last_uv_norm": [0.6, 0.5],
                    "last_world_point": [0.72, -0.03, 1.09],
                },
            }
            self.visible_objects = {"A0": True, "A1": True}
            self.subtask_def_map = {
                1: {
                    "search_target_keys": ["A0", "A1"],
                    "action_target_keys": ["A0", "A1"],
                    "required_carried_keys": [],
                },
            }
            self.search_cursor_state = None
            self.search_cursor_state_index = None
            self.search_cursor_theta = np.nan
            self.search_cursor_layer = None
            self.search_cursor_state_complete = False
            self.search_cursor_boundary_reached = False
            self.fixed_layer_head_joint2_only = True
            self.HEAD_RESET_SAVE_FREQ = -1
            self.reset_calls = []

        def _get_rotate_subtask_def(self, subtask_idx):
            return self.subtask_def_map.get(int(subtask_idx))

        def _move_head_to_rotate_search_layer(self, layer_name, head_joint2_name=None, settle_steps=None, save_freq=-1):
            self.reset_calls.append(str(layer_name))
            return True

        def _reset_head_to_home_pose(self, settle_steps=None, save_freq=-1):
            self.reset_calls.append("home")
            return True

    DummyTask._normalize_rotate_search_layer = staticmethod(SearchBase._normalize_rotate_search_layer)
    task = DummyTask()

    assert task._should_search_lower_before_upper_for_subtask(1) is True
    assert task._should_skip_rotate_head_home_reset(1, prev_subtask_idx=None) is True

    task._prepare_subtask_rotate_search(1)
    assert task.search_cursor_state_index is None
    assert task.discovered_objects["A0"]["discovered"] is False
    assert task.discovered_objects["A1"]["discovered"] is False

    assert task._maybe_reset_head_to_home_for_subtask(1, prev_subtask_idx=None) is True
    assert task.reset_calls == ["lower"]


def test_put_block_on_masks_upper_plate_visibility_during_lower_stage1_search():
    SearchBase = _extract_base_methods(
        [
            "_normalize_rotate_search_layer",
            "_get_rotate_discrete_search_states",
            "_get_rotate_first_upper_search_state_index",
        ]
    )
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_normalize_layer",
            "_get_subtask_search_target_keys",
            "_get_subtask_upper_search_target_keys",
            "_should_search_lower_before_upper_for_subtask",
            "_get_place_search_block_key",
            "_is_upper_plate_search_after_lower_pick",
            "_clear_rotate_target_search_history",
            "_after_rotate_visibility_refresh",
        ],
    )

    class DummyTask(ExtractedTask, SearchBase):
        def __init__(self):
            self.block_keys = ["A0", "A1"]
            self.object_layers = {"A0": "lower", "A1": "lower", "B": "upper"}
            self.plate_layer = "upper"
            self.rotate_table_shape = "fan_double"
            self.carried_object_keys = ["A0"]
            self.current_stage = 1
            self.current_subtask_idx = 2
            self.search_cursor_layer = "lower"
            self.subtask_def_map = {
                2: {
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A0", "B"],
                    "required_carried_keys": ["A0"],
                }
            }
            self.discovered_objects = {
                "B": {
                    "discovered": True,
                    "visible_now": True,
                    "first_seen_frame": 8,
                    "last_seen_frame": 8,
                    "last_seen_subtask": 2,
                    "last_seen_stage": 1,
                    "last_uv_norm": [0.4, 0.5],
                    "last_world_point": [0.68, 0.02, 1.09],
                }
            }
            self.visible_objects = {"B": True}

        def _get_rotate_subtask_def(self, subtask_idx):
            return self.subtask_def_map.get(int(subtask_idx))

    DummyTask._normalize_rotate_search_layer = staticmethod(SearchBase._normalize_rotate_search_layer)
    task = DummyTask()
    visibility_map = {
        "B": {
            "visible": True,
            "u_norm": 0.41,
            "v_norm": 0.37,
            "world_point": [0.68, 0.02, 1.09],
        }
    }

    task._after_rotate_visibility_refresh(visibility_map)

    assert task.visible_objects["B"] is False
    assert task.discovered_objects["B"]["discovered"] is False
    assert visibility_map["B"] == {
        "visible": False,
        "u_norm": None,
        "v_norm": None,
        "world_point": None,
    }


def test_put_block_target_fan_double_upper_target_search_stays_lower_first():
    SearchBase = _extract_base_methods(
        [
            "_normalize_rotate_search_layer",
            "_get_rotate_discrete_search_states",
            "_get_rotate_first_upper_search_state_index",
            "_set_rotate_search_cursor",
        ]
    )
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_TARGET_FAN_DOUBLE_BASE_PATH,
        "PutBlockTargetFanDoubleBase",
        [
            "_normalize_layer",
            "_get_subtask_search_layers",
            "_subtask_requires_head_home_reset",
            "_get_subtask_search_target_keys",
            "_get_subtask_upper_search_target_keys",
            "_should_search_lower_before_upper_for_subtask",
            "_has_unfinished_lower_search_phase",
            "_clear_rotate_target_search_history",
            "_prepare_subtask_rotate_search",
            "_should_enforce_rotate_stage1_search_order",
            "_should_skip_rotate_head_home_reset",
            "_maybe_reset_head_to_home_for_subtask",
            "_after_rotate_visibility_refresh",
        ],
    )

    class DummyTask(ExtractedTask, SearchBase):
        def __init__(self):
            self.object_registry = {"A0": object(), "B": object()}
            self.object_layers = {"A0": "lower", "B": "upper"}
            self.rotate_table_shape = "fan_double"
            self.subtask_def_map = {
                1: {
                    "search_target_keys": ["A0"],
                    "action_target_keys": ["A0"],
                },
                2: {
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A0", "B"],
                },
            }
            self.discovered_objects = {
                "B": {
                    "discovered": True,
                    "visible_now": True,
                    "first_seen_frame": 5,
                    "last_seen_frame": 5,
                    "last_seen_subtask": 2,
                    "last_seen_stage": 1,
                    "last_uv_norm": [0.4, 0.4],
                    "last_world_point": [0.7, 0.0, 1.09],
                }
            }
            self.visible_objects = {"B": True}
            self.search_cursor_state = None
            self.search_cursor_state_index = None
            self.search_cursor_theta = np.nan
            self.search_cursor_layer = None
            self.search_cursor_state_complete = False
            self.search_cursor_boundary_reached = False
            self.current_stage = 1
            self.current_subtask_idx = 2
            self.fixed_layer_head_joint2_only = True
            self.HEAD_RESET_SAVE_FREQ = -1
            self.reset_calls = []

        def _get_rotate_subtask_def(self, subtask_idx):
            return self.subtask_def_map.get(int(subtask_idx))

        def _move_head_to_rotate_search_layer(self, layer_name, head_joint2_name=None, settle_steps=None, save_freq=-1):
            self.reset_calls.append((str(layer_name), save_freq))
            return True

        def _reset_head_to_home_pose(self, settle_steps=None, save_freq=-1):
            self.reset_calls.append(("home", save_freq))
            return True

        def _get_current_scan_camera_theta(self, camera_name=None):
            return 0.0

    DummyTask._normalize_rotate_search_layer = staticmethod(SearchBase._normalize_rotate_search_layer)
    task = DummyTask()

    assert task._should_enforce_rotate_stage1_search_order(2) is True
    assert task._should_skip_rotate_head_home_reset(2, prev_subtask_idx=1) is True

    task._prepare_subtask_rotate_search(2)
    assert task.search_cursor_state_index is None
    assert task.discovered_objects["B"]["discovered"] is False
    assert task.visible_objects["B"] is False

    visibility_map = {
        "B": {
            "visible": True,
            "u_norm": 0.41,
            "v_norm": 0.37,
            "world_point": [0.68, 0.02, 1.09],
        }
    }
    task.search_cursor_layer = "lower"
    task._after_rotate_visibility_refresh(visibility_map)
    assert visibility_map["B"] == {
        "visible": False,
        "u_norm": None,
        "v_norm": None,
        "world_point": None,
    }

    assert task._maybe_reset_head_to_home_for_subtask(2, prev_subtask_idx=1) is True
    assert task.reset_calls == [("lower", -1)]

    task.discovered_objects["B"]["discovered"] = True
    task.visible_objects["B"] = True
    task._set_rotate_search_cursor(state_idx=task._get_rotate_first_upper_search_state_index(), layer_name="upper")

    assert task._should_skip_rotate_head_home_reset(2, prev_subtask_idx=1) is False
    task._prepare_subtask_rotate_search(2)
    assert task.search_cursor_state_index == task._get_rotate_first_upper_search_state_index()
    assert task.search_cursor_layer == "upper"
    assert task._maybe_reset_head_to_home_for_subtask(2, prev_subtask_idx=1) is True
    assert task.reset_calls == [("lower", -1), ("upper", -1)]


def test_put_single_block_target_fan_double_defaults_enable_fixed_head_and_025_escape():
    assert (
        _extract_class_constant(
            PUT_BLOCK_TARGET_FAN_DOUBLE_BASE_PATH,
            "PutSingleBlockTargetFanDoubleBase",
            "FIXED_LAYER_HEAD_JOINT2_ONLY",
        )
        is True
    )
    assert _extract_class_constant(
        PUT_BLOCK_TARGET_FAN_DOUBLE_BASE_PATH,
        "PutSingleBlockTargetFanDoubleBase",
        "UPPER_PLACE_LATERAL_ESCAPE_DIS",
    ) == 0.25


def test_fan_double_utils_skip_reset_moves_head_to_lower_fixed_layer():
    namespace = _extract_module_functions(
        FAN_DOUBLE_UTILS_PATH,
        ["maybe_reset_head_for_subtask"],
    )
    maybe_reset_head_for_subtask = namespace["maybe_reset_head_for_subtask"]

    class DummyTask:
        def __init__(self):
            self.fixed_layer_head_joint2_only = True
            self.HEAD_RESET_SAVE_FREQ = -1
            self.reset_calls = []

        def _should_skip_rotate_head_home_reset(self, subtask_idx, prev_subtask_idx=None):
            assert subtask_idx == 3
            assert prev_subtask_idx == 2
            return True

        def _move_head_to_rotate_search_layer(self, layer_name, save_freq=-1):
            self.reset_calls.append((str(layer_name), save_freq))
            return True

        def _reset_head_to_home_pose(self, save_freq=-1):
            self.reset_calls.append(("home", save_freq))
            return True

    task = DummyTask()
    assert maybe_reset_head_for_subtask(task, 3, prev_subtask_idx=2) is True
    assert task.reset_calls == [("lower", -1)]


def test_blocks_ranking_rgb_fan_double_place_updates_object_layer_to_target_layer():
    fake_fd = types.SimpleNamespace(
        place_object=lambda *args, **kwargs: True,
        normalize_layer=lambda layer_name: str(layer_name).lower(),
    )
    ExtractedTask = _extract_class_methods(
        BLOCKS_RANKING_RGB_FAN_DOUBLE_PATH,
        "blocks_ranking_rgb_fan_double",
        ["_place"],
        extra_namespace={"fd": fake_fd},
    )

    class DummyTask(ExtractedTask):
        TARGET_LAYER = "lower"
        LOWER_PLACE_FUNCTIONAL_POINT_ID = 0
        LOWER_PLACE_PRE_DIS = 0.18
        LOWER_PLACE_DIS = 0.03
        LOWER_PLACE_CONSTRAIN = "free"
        LOWER_PLACE_PRE_DIS_AXIS = "fp"
        LOWER_PLACE_IS_OPEN = True

        def __init__(self):
            self.blocks = {"B": object()}
            self.block_layers = {"B": "upper"}
            self.object_layers = {"A": "lower", "B": "upper", "C": "lower"}
            self.target_poses = {"B": [0.5, 0.0, 0.74, 0.0, 1.0, 0.0, 0.0]}

    task = DummyTask()
    assert task._place(2, "B", "left", "A") is True
    assert task.block_layers["B"] == "lower"
    assert task.object_layers["B"] == "lower"


def test_blocks_ranking_size_fan_double_place_updates_object_layer_to_target_layer():
    fake_fd = types.SimpleNamespace(
        place_object=lambda *args, **kwargs: True,
        normalize_layer=lambda layer_name: str(layer_name).lower(),
    )
    ExtractedTask = _extract_class_methods(
        BLOCKS_RANKING_SIZE_FAN_DOUBLE_PATH,
        "blocks_ranking_size_fan_double",
        ["_place"],
        extra_namespace={"fd": fake_fd},
    )

    class DummyTask(ExtractedTask):
        TARGET_LAYER = "lower"
        LOWER_PLACE_FUNCTIONAL_POINT_ID = 0
        LOWER_PLACE_PRE_DIS = 0.18
        LOWER_PLACE_DIS = 0.03
        LOWER_PLACE_CONSTRAIN = "free"
        LOWER_PLACE_PRE_DIS_AXIS = "fp"
        LOWER_PLACE_IS_OPEN = True

        def __init__(self):
            self.blocks = {"C": object()}
            self.block_layers = {"C": "upper"}
            self.object_layers = {"A": "lower", "B": "lower", "C": "upper"}
            self.target_poses = {"C": [0.4, 0.1, 0.74, 0.0, 1.0, 0.0, 0.0]}

    task = DummyTask()
    assert task._place(4, "C", "left", "B") is True
    assert task.block_layers["C"] == "lower"
    assert task.object_layers["C"] == "lower"


def test_put_block_on_multi_block_success_requires_all_blocks_in_plate():
    ExtractedTask = _extract_class_methods(PUT_BLOCK_ON_PATH, "put_block_on", ["check_success"])

    class DummyPose:
        def __init__(self, p):
            self.p = np.array(p, dtype=np.float64)

    class DummyActor:
        def __init__(self, p):
            self._p = np.array(p, dtype=np.float64)

        def get_functional_point(self, idx, ret="list"):
            assert idx == 0
            assert ret == "pose"
            return DummyPose(self._p)

    class DummyTask(ExtractedTask):
        SUCCESS_EPS = np.array([0.08, 0.08, 0.08], dtype=np.float64)

        def __init__(self, block_points):
            self.plate = DummyActor([0.0, 0.0, 1.0])
            self.blocks = [DummyActor(point) for point in block_points]

        def is_left_gripper_open(self):
            return True

        def is_right_gripper_open(self):
            return True

    assert DummyTask([[0.01, 0.01, 1.02], [-0.02, 0.02, 1.03]]).check_success() is True
    assert DummyTask([[0.01, 0.01, 1.02], [0.20, 0.02, 1.03]]).check_success() is False


def _fake_place_pose_cyl(pose_cyl, robot_root_xy, robot_yaw_rad=0.0, ret="list", quat_frame="world"):
    assert quat_frame == "world"
    pose_cyl = np.array(pose_cyl, dtype=np.float64).reshape(-1)
    theta_world = float(robot_yaw_rad + pose_cyl[1])
    pose = [
        float(robot_root_xy[0] + pose_cyl[0] * math.cos(theta_world)),
        float(robot_root_xy[1] + pose_cyl[0] * math.sin(theta_world)),
        float(pose_cyl[2]),
    ] + pose_cyl[3:].tolist()
    return pose


def test_put_block_on_direct_release_candidates_use_plan_check_order():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_planner_pose_from_tcp_pose",
            "_get_direct_release_entry_r",
            "_build_horizontal_tcp_pose",
            "_build_direct_release_pose_candidates",
            "_select_direct_release_pose",
        ],
        extra_namespace={
            "place_pose_cyl": _fake_place_pose_cyl,
            "ArmTag": lambda arm_tag: str(arm_tag),
            "world_to_robot": lambda world_pt, robot_root_xy, robot_yaw_rad=0.0: [
                float(np.hypot(world_pt[0] - robot_root_xy[0], world_pt[1] - robot_root_xy[1])),
                float(np.arctan2(world_pt[1] - robot_root_xy[1], world_pt[0] - robot_root_xy[0]) - robot_yaw_rad),
                float(world_pt[2]),
            ],
        },
    )

    class DummyRobot:
        def __init__(self, tcp_pose):
            self._tcp_pose = list(tcp_pose)
            self.plan_calls = []

        def get_left_tcp_pose(self):
            return list(self._tcp_pose)

        def right_plan_path(self, planner_pose):
            raise AssertionError("Only left arm planning should be used in this test")

        def left_plan_path(self, planner_pose):
            pose = np.array(planner_pose, dtype=np.float64).reshape(-1)
            self.plan_calls.append(pose.tolist())
            status = "Success" if len(self.plan_calls) == 3 else "Fail"
            return {"status": status}

    class DummyTask(ExtractedTask):
        DIRECT_RELEASE_TCP_BACKOFF = 0.12
        DIRECT_RELEASE_ENTRY_TCP_CYL_R = None
        DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER = 0.08
        DIRECT_RELEASE_TCP_Z_OFFSET = 0.06
        DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET = 0.10
        DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET = 0.10
        DIRECT_RELEASE_R_OFFSETS = (0.0, -0.03, 0.03)
        DIRECT_RELEASE_THETA_OFFSETS_DEG = (0.0, -3.0, 3.0)
        DIRECT_RELEASE_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0)

        def __init__(self):
            self.block_size = 0.02
            self.robot_root_xy = [0.0, 0.0]
            self.robot_yaw = 0.0
            self.plate_layer = "upper"
            self.plate_cyl_r = 0.70
            self.plate_cyl_theta_deg = 0.0
            self.plate_z = 1.14
            self.plate_target_pose = [0.70, 0.0, 1.14, 1.0, 0.0, 0.0, 0.0]
            self.rotate_fan_double_upper_inner_radius = 0.60
            self.robot = DummyRobot([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        def _get_plate_place_target_pose(self, block_key=None):
            assert block_key is None
            return list(self.plate_target_pose)

    task = DummyTask()
    candidates = task._build_direct_release_pose_candidates("left")
    selected = task._select_direct_release_pose("left")

    assert len(task.robot.plan_calls) == 3
    assert np.allclose(task.robot.plan_calls[0], candidates[0]["planner_pose"])
    assert np.allclose(task.robot.plan_calls[1], candidates[1]["planner_pose"])
    assert np.allclose(task.robot.plan_calls[2], candidates[2]["planner_pose"])
    assert np.allclose(selected["planner_pose"], candidates[2]["planner_pose"])
    assert np.allclose(candidates[0]["tcp_pose"][3:], _DummyEuler.euler2quat(0.0, 0.0, 0.0))
    assert np.isclose(candidates[0]["tcp_pose"][2], 1.20)
    assert np.isclose(candidates[0]["entry_tcp_pose"][0], 0.52)
    assert np.isclose(candidates[0]["entry_tcp_pose"][2], 1.24)
    assert np.isclose(candidates[0]["approach_tcp_pose"][2], 1.24)


def test_put_block_on_direct_release_checks_full_pose_sequence_before_execution():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_expand_active_plan_qpos_to_entity_qpos",
            "_plan_path_for_sequence_check",
            "_select_pose_sequence_candidate",
            "_select_direct_release_pose_sequence_candidate",
        ],
        extra_namespace={
            "ArmTag": lambda arm_tag: str(arm_tag),
        },
    )

    class DummyRobot:
        def __init__(self):
            self.plan_calls = []
            self.left_entity = type("DummyEntity", (), {"get_qpos": lambda _self: np.zeros(9, dtype=np.float64)})()
            self.left_planner = type(
                "DummyPlanner",
                (),
                {
                    "all_joints": [f"j{i}" for i in range(9)],
                    "active_joints_name": [f"j{i}" for i in range(2, 9)],
                },
            )()

        def left_plan_path(self, planner_pose, last_qpos=None):
            self.plan_calls.append(
                {
                    "pose": list(planner_pose),
                    "last_qpos": None if last_qpos is None else np.array(last_qpos, dtype=np.float64).tolist(),
                }
            )
            call_idx = len(self.plan_calls)
            status = "Fail" if call_idx == 3 else "Success"
            return {"status": status, "position": np.full((1, 7), float(call_idx), dtype=np.float64)}

        def right_plan_path(self, planner_pose, last_qpos=None):
            raise AssertionError("Only left arm planning should be used")

    class DummyTask(ExtractedTask):
        def __init__(self):
            self.robot = DummyRobot()
            self.need_plan = True

    task = DummyTask()
    candidates = [
        {
            "entry_planner_pose": [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "approach_planner_pose": [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "planner_pose": [0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
        {
            "entry_planner_pose": [0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "approach_planner_pose": [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "planner_pose": [0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
    ]

    selected = task._select_direct_release_pose_sequence_candidate("left", candidates=candidates)

    assert selected is candidates[1]
    assert len(task.robot.plan_calls) == 6
    assert task.robot.plan_calls[0]["last_qpos"] is None
    assert len(task.robot.plan_calls[1]["last_qpos"]) == 9
    assert task.robot.plan_calls[1]["last_qpos"][:2] == [0.0, 0.0]
    assert task.robot.plan_calls[1]["last_qpos"][2:] == [1.0] * 7
    assert task.robot.plan_calls[2]["last_qpos"][2:] == [2.0] * 7
    assert task.robot.plan_calls[3]["last_qpos"] is None


def test_put_block_on_upper_pick_candidates_include_post_grasp_retreat_poses():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_planner_pose_from_tcp_pose",
            "_build_upper_pick_pose_candidates",
        ],
        extra_namespace={
            "ArmTag": lambda arm_tag: str(arm_tag),
        },
    )

    class DummyPose:
        def __init__(self, p):
            self.p = np.array(p, dtype=np.float64)

    class DummyBlock:
        def get_pose(self):
            return DummyPose([0.65, 0.0, 1.12])

    class DummyTask(ExtractedTask):
        DIRECT_RELEASE_TCP_BACKOFF = 0.12
        UPPER_PICK_ENTRY_Z_OFFSET = 0.08
        UPPER_PICK_PRE_GRASP_DIS = 0.10
        UPPER_PICK_GRASP_Z_BIAS = 0.0
        UPPER_PICK_YAW_OFFSETS_DEG = (0.0,)
        UPPER_PICK_POST_ENTRY_RETREAT_DIS = 0.05

        def __init__(self):
            self.robot_root_xy = [0.0, 0.0]
            self.robot_yaw = 0.0

    task = DummyTask()
    candidate = task._build_upper_pick_pose_candidates(DummyBlock(), "left")[0]

    assert "post_grasp_retreat_tcp_pose" in candidate
    assert "post_grasp_retreat_planner_pose" in candidate
    assert "post_entry_retreat_tcp_pose" in candidate
    assert "post_entry_retreat_planner_pose" in candidate
    assert np.allclose(
        candidate["post_grasp_retreat_tcp_pose"][:2],
        candidate["pre_grasp_tcp_pose"][:2],
    )
    assert np.isclose(candidate["post_grasp_retreat_tcp_pose"][2], candidate["pre_grasp_tcp_pose"][2])
    assert candidate["post_entry_retreat_tcp_pose"][0] < candidate["entry_tcp_pose"][0]
    assert np.isclose(candidate["post_entry_retreat_tcp_pose"][2], candidate["entry_tcp_pose"][2])


def test_put_block_on_upper_pick_checks_full_pose_sequence_before_execution():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_planner_pose_from_tcp_pose",
            "_expand_active_plan_qpos_to_entity_qpos",
            "_plan_path_for_sequence_check",
            "_build_upper_pick_pose_candidates",
            "_select_pose_sequence_candidate",
        ],
        extra_namespace={
            "ArmTag": lambda arm_tag: str(arm_tag),
        },
    )

    class DummyPose:
        def __init__(self, p):
            self.p = np.array(p, dtype=np.float64)

    class DummyBlock:
        def get_pose(self):
            return DummyPose([0.65, 0.0, 1.12])

    class DummyRobot:
        def __init__(self):
            self.plan_calls = []
            self.left_entity = type("DummyEntity", (), {"get_qpos": lambda _self: np.zeros(9, dtype=np.float64)})()
            self.left_planner = type(
                "DummyPlanner",
                (),
                {
                    "all_joints": [f"j{i}" for i in range(9)],
                    "active_joints_name": [f"j{i}" for i in range(2, 9)],
                },
            )()

        def left_plan_path(self, planner_pose, last_qpos=None):
            self.plan_calls.append(
                {
                    "pose": list(planner_pose),
                    "last_qpos": None if last_qpos is None else np.array(last_qpos, dtype=np.float64).tolist(),
                }
            )
            call_idx = len(self.plan_calls)
            status = "Fail" if call_idx == 6 else "Success"
            return {"status": status, "position": np.full((1, 7), float(call_idx), dtype=np.float64)}

        def right_plan_path(self, planner_pose, last_qpos=None):
            raise AssertionError("Only left arm planning should be used")

    class DummyTask(ExtractedTask):
        DIRECT_RELEASE_TCP_BACKOFF = 0.12
        UPPER_PICK_ENTRY_Z_OFFSET = 0.08
        UPPER_PICK_PRE_GRASP_DIS = 0.10
        UPPER_PICK_GRASP_Z_BIAS = 0.0
        UPPER_PICK_YAW_OFFSETS_DEG = (0.0, 15.0)
        UPPER_PICK_POST_ENTRY_RETREAT_DIS = 0.05

        def __init__(self):
            self.robot_root_xy = [0.0, 0.0]
            self.robot_yaw = 0.0
            self.robot = DummyRobot()
            self.need_plan = True

    task = DummyTask()
    candidates = task._build_upper_pick_pose_candidates(DummyBlock(), "left")
    selected = task._select_pose_sequence_candidate(
        "left",
        candidates,
        (
            "entry_planner_pose",
            "pre_grasp_planner_pose",
            "grasp_planner_pose",
            "post_grasp_retreat_planner_pose",
            "entry_planner_pose",
            "post_entry_retreat_planner_pose",
        ),
    )

    assert selected is candidates[1]
    assert len(task.robot.plan_calls) == 12
    assert task.robot.plan_calls[0]["last_qpos"] is None
    assert len(task.robot.plan_calls[1]["last_qpos"]) == 9
    assert task.robot.plan_calls[1]["last_qpos"][:2] == [0.0, 0.0]
    assert task.robot.plan_calls[1]["last_qpos"][2:] == [1.0] * 7
    assert task.robot.plan_calls[2]["last_qpos"][2:] == [2.0] * 7
    assert task.robot.plan_calls[3]["last_qpos"][2:] == [3.0] * 7
    assert task.robot.plan_calls[4]["last_qpos"][2:] == [4.0] * 7
    assert task.robot.plan_calls[5]["last_qpos"][2:] == [5.0] * 7
    assert task.robot.plan_calls[6]["last_qpos"] is None


def test_put_block_on_upper_pick_executes_post_grasp_retreat_and_syncs_collisions():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_pick_upper_block_with_direct_move",
        ],
    )

    class DummyTask(ExtractedTask):
        UPPER_PICK_GRIPPER_POS = -0.1

        def __init__(self):
            self.plan_success = True
            self.sync_calls = 0
            self.move_calls = []
            self.carried = None
            self.completed = None

        def _sync_curobo_tabletop_collisions(self):
            self.sync_calls += 1

        def _build_upper_pick_pose_candidates(self, block, arm_tag):
            assert arm_tag == "left"
            return [{
                "entry_planner_pose": "entry_pose",
                "pre_grasp_planner_pose": "pre_grasp_pose",
                "grasp_planner_pose": "grasp_pose",
                "post_grasp_retreat_planner_pose": "post_retreat_pose",
                "post_entry_retreat_planner_pose": "post_entry_retreat_pose",
            }]

        def _select_pose_sequence_candidate(self, arm_tag, candidates, pose_keys):
            assert arm_tag == "left"
            assert len(candidates) == 1
            assert pose_keys == (
                "entry_planner_pose",
                "pre_grasp_planner_pose",
                "grasp_planner_pose",
                "post_grasp_retreat_planner_pose",
                "entry_planner_pose",
                "post_entry_retreat_planner_pose",
            )
            return candidates[0]

        def open_gripper(self, arm_tag):
            return ("open", arm_tag)

        def close_gripper(self, arm_tag, pos):
            return ("close", arm_tag, pos)

        def move_to_pose(self, arm_tag, target_pose):
            return ("move_pose", arm_tag, target_pose)

        def move(self, action):
            self.move_calls.append(action)
            return True

        def _set_carried_object_keys(self, keys):
            self.carried = list(keys)

        def complete_rotate_subtask(self, subtask_idx, carried_after):
            self.completed = (subtask_idx, list(carried_after))

    task = DummyTask()
    result = task._pick_upper_block_with_direct_move(5, "A0", object(), "left")

    assert result == "left"
    assert task.sync_calls == 1
    assert task.carried == ["A0"]
    assert task.move_calls == [
        ("open", "left"),
        ("move_pose", "left", "entry_pose"),
        ("move_pose", "left", "pre_grasp_pose"),
        ("move_pose", "left", "grasp_pose"),
        ("close", "left", -0.1),
        ("move_pose", "left", "post_retreat_pose"),
        ("move_pose", "left", "entry_pose"),
        ("move_pose", "left", "post_entry_retreat_pose"),
    ]
    assert task.completed == (5, ["A0"])


def test_put_block_on_direct_release_place_reuses_single_selected_candidate():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_place_block_into_plate_with_direct_release",
        ],
    )

    class DummyTask(ExtractedTask):
        def __init__(self):
            self.plan_success = True
            self.candidates = [
                {"name": "first"},
                {"name": "second"},
            ]
            self.sequence_select_calls = 0
            self.pose_calls = []
            self.release_target_pose = [0.68, 0.04, 1.14, 1.0, 0.0, 0.0, 0.0]
            self.move_calls = []
            self.placed_block_keys = []
            self.carried = None
            self.completed = None
            self.retreat_arm = None

        def _get_plate_place_target_pose(self, block_key=None):
            assert block_key == "A0"
            return list(self.release_target_pose)

        def _build_direct_release_pose_candidates(self, arm_tag, target_pose=None):
            assert arm_tag == "left"
            assert target_pose == self.release_target_pose
            return self.candidates

        def _select_direct_release_pose_sequence_candidate(self, arm_tag, candidates=None):
            assert arm_tag == "left"
            assert candidates is self.candidates
            self.sequence_select_calls += 1
            return candidates[1]

        def _move_to_first_direct_release_pose(self, arm_tag, candidates, pose_key, selected_candidate=None):
            assert arm_tag == "left"
            assert candidates is self.candidates
            self.pose_calls.append((pose_key, selected_candidate))
            return True

        def open_gripper(self, arm_tag):
            return ("open", arm_tag)

        def move(self, action):
            self.move_calls.append(action)
            return True

        def _set_carried_object_keys(self, keys):
            self.carried = list(keys)

        def _retreat_then_return_both_arms_to_initial_pose(self, arm_tag):
            self.retreat_arm = arm_tag
            return True

        def complete_rotate_subtask(self, subtask_idx, carried_after):
            self.completed = (subtask_idx, list(carried_after))

    task = DummyTask()
    task._place_block_into_plate_with_direct_release("left", 8, block_key="A0")

    assert task.sequence_select_calls == 1
    assert [pose_key for pose_key, _ in task.pose_calls] == [
        "entry_planner_pose",
        "approach_planner_pose",
        "planner_pose",
    ]
    assert all(selected is task.candidates[1] for _, selected in task.pose_calls)
    assert task.carried == []
    assert task.placed_block_keys == ["A0"]
    assert task.retreat_arm == "left"
    assert task.completed == (8, [])
    assert task.move_calls == [("open", "left")]


def test_put_block_on_upper_to_lower_direct_release_98c4504_checks_each_stage_separately():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_place_upper_picked_block_into_lower_plate_with_direct_release_98c4504",
        ],
    )

    class DummyTask(ExtractedTask):
        def __init__(self):
            self.plan_success = True
            self.candidates = [
                {"name": "first"},
                {"name": "second"},
            ]
            self.select_calls = []
            self.execute_calls = []
            self.release_target_pose = [0.48, -0.12, 0.76, 1.0, 0.0, 0.0, 0.0]
            self.move_calls = []
            self.placed_block_keys = []
            self.carried = None
            self.completed = None
            self.retreat_arm = None

        def _get_plate_place_target_pose(self, block_key=None):
            assert block_key == "A0"
            return list(self.release_target_pose)

        def _build_direct_release_pose_candidates(self, arm_tag, target_pose=None):
            assert arm_tag == "left"
            assert target_pose == self.release_target_pose
            return self.candidates

        def _select_direct_release_pose_with_plan(self, arm_tag, candidates=None, pose_key="planner_pose"):
            assert arm_tag == "left"
            assert candidates is self.candidates
            self.select_calls.append(pose_key)
            candidate = {
                "entry_planner_pose": "entry_pose",
                "approach_planner_pose": "approach_pose",
                "planner_pose": "release_pose",
            }
            return candidate, {"position": np.zeros((2, 7), dtype=np.float64), "velocity": np.zeros((2, 7), dtype=np.float64)}

        def _execute_arm_plan_result(self, arm_tag, plan_result, fallback_pose=None):
            assert arm_tag == "left"
            self.execute_calls.append((plan_result, fallback_pose))
            return True

        def open_gripper(self, arm_tag):
            return ("open", arm_tag)

        def move(self, action):
            self.move_calls.append(action)
            return True

        def _set_carried_object_keys(self, keys):
            self.carried = list(keys)

        def _retreat_then_return_both_arms_to_initial_pose(self, arm_tag):
            self.retreat_arm = arm_tag
            return True

        def complete_rotate_subtask(self, subtask_idx, carried_after):
            self.completed = (subtask_idx, list(carried_after))

    task = DummyTask()
    task._place_upper_picked_block_into_lower_plate_with_direct_release_98c4504("left", 8, block_key="A0")

    assert task.select_calls == [
        "entry_planner_pose",
        "approach_planner_pose",
        "planner_pose",
    ]
    assert [fallback_pose for _, fallback_pose in task.execute_calls] == [
        "entry_pose",
        "approach_pose",
        "release_pose",
    ]
    assert task.carried == []
    assert task.retreat_arm == "left"
    assert task.placed_block_keys == ["A0"]
    assert task.completed == (8, [])
    assert task.move_calls == [("open", "left")]


def test_put_block_on_lower_place_uses_place_actor_then_arm_axis_retreat():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_retreat_after_lower_place",
            "_place_block_into_lower_plate_with_place_actor",
        ],
    )

    class DummyTask(ExtractedTask):
        PLACE_RETREAT_Z = 0.07
        RETURN_TO_HOMESTATE_AFTER_PLACE = True
        LOWER_PLACE_PRE_DIS = 0.12
        LOWER_PLACE_DIS = 0.01
        LOWER_PLACE_CONSTRAIN = "free"
        LOWER_PLACE_PRE_DIS_AXIS = "fp"

        def __init__(self):
            self.plan_success = True
            self.object_registry = {"A0": object()}
            self.place_target = [0.55, 0.03, 0.74, 0.0, 1.0, 0.0, 0.0]
            self.move_calls = []
            self.placed_block_keys = []
            self.carried = None
            self.completed = None

        def _get_plate_place_target_pose(self, block_key=None):
            assert block_key == "A0"
            return list(self.place_target)

        def place_actor(
            self,
            actor,
            target_pose,
            arm_tag,
            functional_point_id=None,
            pre_dis=None,
            dis=None,
            pre_dis_axis=None,
            constrain=None,
        ):
            assert actor is self.object_registry["A0"]
            assert target_pose == self.place_target
            assert arm_tag == "left"
            assert functional_point_id == 0
            assert pre_dis == 0.12
            assert dis == 0.01
            assert pre_dis_axis == "fp"
            assert constrain == "free"
            return ("place_actor", arm_tag)

        def move_by_displacement(self, arm_tag, x=0.0, y=0.0, z=0.0, quat=None, move_axis="world"):
            return ("disp", arm_tag, float(x), float(y), float(z), move_axis)

        def back_to_origin(self, arm_tag):
            return ("home", arm_tag)

        def move(self, *actions):
            self.move_calls.append(actions if len(actions) > 1 else actions[0])
            return True

        def _set_carried_object_keys(self, keys):
            self.carried = list(keys)

        def complete_rotate_subtask(self, subtask_idx, carried_after):
            self.completed = (subtask_idx, list(carried_after))

    task = DummyTask()
    task._place_block_into_lower_plate_with_place_actor("left", 6, "A0")

    assert task.move_calls == [
        ("place_actor", "left"),
        ("disp", "left", 0.0, 0.0, 0.07, "arm"),
        (("home", "left"), ("home", "right")),
    ]
    assert task.carried == []
    assert task.completed == (6, [])
    assert task.placed_block_keys == ["A0"]


def test_put_block_on_plate_place_slots_spread_three_blocks_without_overlap():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_get_plate_place_slot_offsets",
            "_get_plate_place_slot_index",
            "_get_plate_place_target_pose",
        ],
        extra_namespace={
            "world_to_robot": lambda world_pt, robot_root_xy, robot_yaw_rad=0.0: [
                float(np.hypot(world_pt[0] - robot_root_xy[0], world_pt[1] - robot_root_xy[1])),
                float(np.arctan2(world_pt[1] - robot_root_xy[1], world_pt[0] - robot_root_xy[0]) - robot_yaw_rad),
                float(world_pt[2]),
            ],
        },
    )

    class DummyTask(ExtractedTask):
        PLATE_PLACE_SLOT_OFFSETS = {
            3: ((-0.028, 0.0), (0.020, -0.032), (0.020, 0.032)),
        }

        def __init__(self):
            self.block_count = 3
            self.block_keys = ["A0", "A1", "A2"]
            self.robot_root_xy = [0.0, 0.0]
            self.robot_yaw = 0.0
            self.plate_place_slot_assignments = {}
            self.object_registry = {}
            self.plate_target_pose = [0.50, 0.00, 0.74, 1.0, 0.0, 0.0, 0.0]

    task = DummyTask()
    poses = [
        np.array(task._get_plate_place_target_pose(block_key), dtype=np.float64)
        for block_key in ("A0", "A1", "A2")
    ]

    xy_positions = [pose[:2] for pose in poses]
    pairwise_distances = [
        float(np.linalg.norm(xy_positions[i] - xy_positions[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    ]

    assert all(distance > 0.045 for distance in pairwise_distances)
    assert all(np.linalg.norm(pose[:2] - np.array([0.50, 0.00])) < 0.05 for pose in poses)


def test_put_block_on_pick_block_routes_lower_without_upper_pick_logic():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_pick_block",
        ],
        extra_namespace={
            "ArmTag": lambda arm_tag: str(arm_tag),
        },
    )

    class DummyTask(ExtractedTask):
        def __init__(self, layer_name):
            self.object_registry = {"A0": object()}
            self.object_layers = {"A0": layer_name}
            self.stage_calls = []
            self.lower_calls = []
            self.upper_calls = []

        def _get_object_arm_tag(self, obj):
            return "left"

        def enter_rotate_action_stage(self, subtask_idx, focus_object_key=None):
            self.stage_calls.append((subtask_idx, focus_object_key))

        def _pick_upper_block_with_direct_move(self, subtask_idx, block_key, block, arm_tag):
            self.upper_calls.append((subtask_idx, block_key, arm_tag))
            return "upper"

        def _pick_lower_block_with_grasp_actor(self, subtask_idx, block_key, block, arm_tag):
            self.lower_calls.append((subtask_idx, block_key, arm_tag))
            return "lower"

    lower_task = DummyTask("lower")
    upper_task = DummyTask("upper")

    assert lower_task._pick_block(3, "A0") == "lower"
    assert lower_task.lower_calls == [(3, "A0", "left")]
    assert lower_task.upper_calls == []
    assert upper_task._pick_block(4, "A0") == "upper"
    assert upper_task.upper_calls == [(4, "A0", "left")]
    assert upper_task.lower_calls == []


def test_put_block_on_upper_place_retreat_adds_lateral_escape_before_home():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_get_post_place_escape_theta",
            "_get_lateral_escape_displacement",
            "_retreat_then_return_both_arms_to_initial_pose",
        ],
        extra_namespace={
            "ArmTag": lambda arm_tag: str(arm_tag),
            "world_to_robot": lambda world_pt, robot_root_xy, robot_yaw_rad=0.0: [
                float(np.hypot(world_pt[0] - robot_root_xy[0], world_pt[1] - robot_root_xy[1])),
                float(np.arctan2(world_pt[1] - robot_root_xy[1], world_pt[0] - robot_root_xy[0]) - robot_yaw_rad),
                float(world_pt[2]),
            ],
        },
    )

    class DummyTask(ExtractedTask):
        DIRECT_RELEASE_RETREAT_Z = 0.06
        POST_PLACE_LATERAL_ESCAPE_DIS = 0.12
        RETURN_TO_HOMESTATE_AFTER_PLACE = True

        def __init__(self, plate_layer):
            self.robot_yaw = 0.0
            self.robot_root_xy = [0.0, 0.0]
            self.object_layers = {"B": plate_layer}
            self.move_calls = []

        def move_by_displacement(self, arm_tag, x=0.0, y=0.0, z=0.0, quat=None, move_axis="world"):
            return ("disp", arm_tag, float(x), float(y), float(z), move_axis)

        def back_to_origin(self, arm_tag):
            return ("home", arm_tag)

        def move(self, *actions):
            self.move_calls.append(actions if len(actions) > 1 else actions[0])
            return True

    upper_task = DummyTask("upper")
    lower_task = DummyTask("lower")

    assert upper_task._get_lateral_escape_displacement("left") == [0.0, 0.12]
    assert upper_task._get_lateral_escape_displacement("right") == [-0.0, -0.12]
    assert upper_task._retreat_then_return_both_arms_to_initial_pose("left") is True
    assert upper_task.move_calls == [
        ("disp", "left", 0.0, 0.0, 0.06, "world"),
        ("disp", "left", 0.0, 0.12, 0.0, "world"),
        (("home", "left"), ("home", "right")),
    ]

    assert lower_task._retreat_then_return_both_arms_to_initial_pose("left") is True
    assert lower_task.move_calls == [
        ("disp", "left", 0.0, 0.0, 0.06, "world"),
        (("home", "left"), ("home", "right")),
    ]


def test_put_block_on_lateral_escape_tracks_current_theta_instead_of_fixed_robot_side():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_get_post_place_escape_theta",
            "_get_lateral_escape_displacement",
        ],
        extra_namespace={
            "ArmTag": lambda arm_tag: str(arm_tag),
            "world_to_robot": lambda world_pt, robot_root_xy, robot_yaw_rad=0.0: [
                float(np.hypot(world_pt[0] - robot_root_xy[0], world_pt[1] - robot_root_xy[1])),
                float(np.arctan2(world_pt[1] - robot_root_xy[1], world_pt[0] - robot_root_xy[0]) - robot_yaw_rad),
                float(world_pt[2]),
            ],
        },
    )

    class DummyTask(ExtractedTask):
        POST_PLACE_LATERAL_ESCAPE_DIS = 0.2

        def __init__(self):
            self.robot_yaw = 0.0
            self.robot_root_xy = [0.0, 0.0]

        def get_arm_pose(self, arm_tag):
            assert arm_tag == "left"
            return [1.0, 1.0, 1.2, 1.0, 0.0, 0.0, 0.0]

    task = DummyTask()
    lateral_xy = np.array(task._get_lateral_escape_displacement("left"), dtype=np.float64)

    expected = np.array(
        [
            -np.sqrt(0.5),
            np.sqrt(0.5),
        ],
        dtype=np.float64,
    ) * 0.2
    assert np.allclose(lateral_xy, expected)


def test_put_block_on_upper_pick_plan_check_runtime_error_does_not_escape():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_plan_path_for_sequence_check",
        ],
    )

    class DummyTask(ExtractedTask):
        pass

    task = DummyTask()
    calls = []

    def flaky_plan_path(planner_pose, last_qpos=None):
        calls.append(last_qpos)
        if last_qpos is not None:
            raise RuntimeError("expected scalar type Float but found Double")
        return {"status": "Success"}

    result = task._plan_path_for_sequence_check(flaky_plan_path, [0, 0, 0, 1, 0, 0, 0], last_entity_qpos=[0.0] * 9)

    assert result["status"] == "Success"
    assert calls == [[0.0] * 9, None]


def test_put_block_on_replay_uses_cached_arm_and_skips_release_plan_check():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_get_object_arm_tag",
            "_select_direct_release_pose",
        ],
        extra_namespace={
            "ArmTag": lambda arm_tag: str(arm_tag),
        },
    )

    class DummyRobot:
        def left_plan_path(self, planner_pose):
            raise AssertionError("Replay mode should not run left plan checks")

        def right_plan_path(self, planner_pose):
            raise AssertionError("Replay mode should not run right plan checks")

    class DummyTask(ExtractedTask):
        def __init__(self):
            self.need_plan = False
            self.left_joint_path = []
            self.right_joint_path = [{"status": "Success"}]
            self.left_cnt = 0
            self.right_cnt = 0
            self.robot = DummyRobot()

    task = DummyTask()
    candidates = [
        {"planner_pose": [1, 0, 0, 1, 0, 0, 0]},
        {"planner_pose": [2, 0, 0, 1, 0, 0, 0]},
    ]

    assert task._get_object_arm_tag(None) == "right"
    assert task._select_direct_release_pose("right", candidates=candidates) is candidates[0]


def test_put_block_on_pick_no_longer_returns_arm_home():
    source = PUT_BLOCK_ON_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "put_block_on")
    pick_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "_pick_block")
    lower_pick_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_pick_lower_block_with_grasp_actor"
    )
    upper_pick_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_pick_upper_block_with_direct_move"
    )

    call_order = []
    for node in ast.walk(pick_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            call_order.append(func.attr)

    lower_call_order = []
    for node in ast.walk(lower_pick_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            lower_call_order.append(func.attr)

    upper_call_order = []
    for node in ast.walk(upper_pick_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            upper_call_order.append(func.attr)

    assert "_pick_upper_block_with_direct_move" in call_order
    assert "_pick_lower_block_with_grasp_actor" in call_order
    assert "grasp_actor" in lower_call_order
    assert "grasp_actor" not in upper_call_order
    assert "_build_upper_pick_pose_candidates" in upper_call_order
    assert "_select_pose_sequence_candidate" in upper_call_order
    assert "_lift_block_to_place_ready_pose" in lower_call_order
    assert "_lift_block_to_place_ready_pose" not in upper_call_order
    assert "_return_grasp_arm_to_home_pose" not in call_order


def test_put_block_on_focuses_plate_with_shared_visibility_check():
    source = PUT_BLOCK_ON_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "put_block_on")
    play_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "play_once")

    play_call_order = []
    for node in ast.walk(play_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            play_call_order.append(func.attr)

    assert play_call_order.count("search_and_focus_rotate_and_head_subtask") == 2
    assert "_focus_plate_before_place" in play_call_order
    assert "_place_block_into_plate" in play_call_order

    focus_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_focus_plate_before_place"
    )
    focus_call_order = []
    for node in ast.walk(focus_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            focus_call_order.append(func.attr)

    assert "_ensure_action_target_visible" in focus_call_order

    ensure_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_action_target_visible"
    )
    ensure_call_order = []
    for node in ast.walk(ensure_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            ensure_call_order.append(func.attr)

    assert "_align_rotate_registry_target_with_torso_and_head_joint2" in ensure_call_order


def test_put_block_on_play_once_uses_snapshot_restore_and_not_head_reset():
    source = PUT_BLOCK_ON_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "put_block_on")
    play_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "play_once")

    self_calls = []
    for node in ast.walk(play_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            self_calls.append(func.attr)

    assert "_reset_head_to_home_pose" not in self_calls
    assert "_restore_block_search_snapshot_or_default" in self_calls
    assert "_ensure_action_target_visible" in self_calls
    assert "_has_discovered_pending_block" in self_calls


def test_put_block_on_restore_prefers_snapshot_for_remaining_block():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_after_rotate_visibility_refresh",
            "_restore_block_search_snapshot_or_default",
        ],
    )

    class DummyTask(ExtractedTask):
        SCAN_R = 0.62
        SCAN_JOINT_NAME = "astribot_torso_joint_2"

        def __init__(self):
            self.object_registry = {"A0": object(), "A1": object()}
            self.visible_objects = {"A0": False, "A1": False}
            self.pending_block_keys_for_search_snapshot = ["A0", "A1"]
            self.pending_block_search_snapshots = {}
            self.pending_block_search_snapshot_seq = 0
            self.last_pending_block_search_snapshot = None
            self.plan_success = True
            self.restore_calls = []

        def _capture_rotate_search_snapshot(self):
            if self.visible_objects["A0"]:
                theta = 30.0
            elif self.visible_objects["A1"]:
                theta = 20.0
            else:
                theta = -1.0
            return {
                "search_cursor_state": "lower_left",
                "search_cursor_state_index": 1,
                "search_cursor_theta": theta,
                "search_cursor_layer": "lower",
            }

        def _restore_rotate_search_snapshot(self, snapshot, scan_r, scan_z, joint_name_prefer, head_joint2_name):
            self.restore_calls.append(
                {
                    "snapshot": dict(snapshot),
                    "scan_r": scan_r,
                    "scan_z": scan_z,
                    "joint_name_prefer": joint_name_prefer,
                    "head_joint2_name": head_joint2_name,
                }
            )
            return True

        def _get_default_rotate_search_snapshot(self):
            return {
                "search_cursor_state": "lower_center",
                "search_cursor_state_index": 0,
                "search_cursor_theta": 0.0,
                "search_cursor_layer": "lower",
            }

    task = DummyTask()
    task.visible_objects = {"A0": True, "A1": False}
    task._after_rotate_visibility_refresh({})
    task.visible_objects = {"A0": False, "A1": True}
    task._after_rotate_visibility_refresh({})
    task.visible_objects = {"A0": True, "A1": False}
    task._after_rotate_visibility_refresh({})

    task.pending_block_keys_for_search_snapshot = ["A1"]
    assert task._restore_block_search_snapshot_or_default(scan_z=0.91) is True
    assert task.restore_calls[-1]["snapshot"]["search_cursor_theta"] == 20.0
    assert task.restore_calls[-1]["snapshot"]["pending_block_keys"] == ["A1"]


def test_put_block_on_load_actors_passes_plate_avoid_pose_to_block_sampling():
    source = PUT_BLOCK_ON_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "put_block_on")
    load_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "load_actors")

    sample_call = next(
        node
        for node in ast.walk(load_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_sample_block_pose"
    )
    keyword_names = {keyword.arg for keyword in sample_call.keywords}

    assert "avoid_pose_lst" in keyword_names
    assert "avoid_min_dist_sq" in keyword_names


def test_put_block_on_place_routes_lower_to_place_actor_and_uses_98c4504_helper_for_upper_to_lower():
    source = PUT_BLOCK_ON_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "put_block_on")
    place_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "_place_block_into_plate")
    lower_place_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_place_block_into_lower_plate_with_place_actor"
    )
    direct_release_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_place_block_into_plate_with_direct_release"
    )
    upper_to_lower_direct_release_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_place_upper_picked_block_into_lower_plate_with_direct_release_98c4504"
    )
    call_order = []
    for node in ast.walk(place_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            call_order.append(func.attr)

    assert "_place_block_into_lower_plate_with_place_actor" in call_order
    assert "_place_block_into_plate_with_direct_release" in call_order
    assert "_place_upper_picked_block_into_lower_plate_with_direct_release_98c4504" in call_order

    lower_call_order = []
    for node in ast.walk(lower_place_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            lower_call_order.append(func.attr)

    direct_call_order = []
    for node in ast.walk(direct_release_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            direct_call_order.append(func.attr)

    upper_to_lower_direct_call_order = []
    for node in ast.walk(upper_to_lower_direct_release_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            upper_to_lower_direct_call_order.append(func.attr)

    assert "place_actor" in lower_call_order
    assert "_retreat_after_lower_place" in lower_call_order
    assert direct_call_order.count("_move_to_first_direct_release_pose") == 3
    assert "place_actor" not in direct_call_order
    assert "open_gripper" in direct_call_order
    assert "_retreat_then_return_both_arms_to_initial_pose" in direct_call_order
    assert upper_to_lower_direct_call_order.count("_select_direct_release_pose_with_plan") == 3
    assert upper_to_lower_direct_call_order.count("_execute_arm_plan_result") == 3
    assert "_move_to_first_direct_release_pose" not in upper_to_lower_direct_call_order
    assert "_select_direct_release_pose_sequence_candidate" not in upper_to_lower_direct_call_order
    assert "open_gripper" in upper_to_lower_direct_call_order
    assert "_retreat_then_return_both_arms_to_initial_pose" in upper_to_lower_direct_call_order

    helper_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_move_to_first_direct_release_pose"
    )
    helper_call_order = []
    for node in ast.walk(helper_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            helper_call_order.append(func.attr)

    assert "_select_direct_release_pose" in helper_call_order
    assert "move_to_pose" in helper_call_order

    retreat_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_retreat_then_return_both_arms_to_initial_pose"
    )
    retreat_call_order = []
    for node in ast.walk(retreat_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            retreat_call_order.append(func.attr)

    assert "move_by_displacement" in retreat_call_order
    assert retreat_call_order.count("back_to_origin") == 2
    assert "move" in retreat_call_order

    lower_retreat_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_retreat_after_lower_place"
    )
    lower_retreat_call_order = []
    for node in ast.walk(lower_retreat_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            lower_retreat_call_order.append(func.attr)

    assert "move_by_displacement" in lower_retreat_call_order
    assert lower_retreat_call_order.count("back_to_origin") == 2
    assert "move" in lower_retreat_call_order


def test_put_block_on_upper_to_lower_uses_98c4504_direct_release_while_lower_to_lower_keeps_place_actor():
    ExtractedTask = _extract_class_methods(
        PUT_BLOCK_ON_PATH,
        "put_block_on",
        [
            "_place_block_into_plate",
        ],
    )

    class DummyTask(ExtractedTask):
        def __init__(self, block_layer):
            self.plan_success = True
            self.object_layers = {"A0": block_layer, "B": "lower"}
            self.stage_calls = []
            self.sync_calls = 0
            self.lower_calls = []
            self.direct_release_calls = []

        def enter_rotate_action_stage(self, subtask_idx, focus_object_key=None):
            self.stage_calls.append((subtask_idx, focus_object_key))

        def _sync_curobo_tabletop_collisions(self):
            self.sync_calls += 1

        def _place_block_into_lower_plate_with_place_actor(self, arm_tag, subtask_idx, block_key):
            self.lower_calls.append((arm_tag, subtask_idx, block_key))

        def _place_upper_picked_block_into_lower_plate_with_direct_release_98c4504(self, arm_tag, subtask_idx, block_key=None):
            self.direct_release_calls.append((arm_tag, subtask_idx, block_key))

    upper_task = DummyTask("upper")
    upper_task._place_block_into_plate("left", 2, "A0", "B")
    assert upper_task.stage_calls == [(2, "B")]
    assert upper_task.sync_calls == 1
    assert upper_task.lower_calls == []
    assert upper_task.direct_release_calls == [("left", 2, "A0")]

    lower_task = DummyTask("lower")
    lower_task._place_block_into_plate("left", 2, "A0", "B")
    assert lower_task.stage_calls == [(2, "B")]
    assert lower_task.sync_calls == 1
    assert lower_task.lower_calls == [("left", 2, "A0")]
    assert lower_task.direct_release_calls == []


def test_put_block_on_does_not_prime_plate_memory():
    source = PUT_BLOCK_ON_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "put_block_on")
    assign = next(
        node
        for node in class_node.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "KNOWN_FIXED_TARGET_KEYS" for target in node.targets)
    )
    assert isinstance(assign.value, ast.Tuple)
    assert len(assign.value.elts) == 0


def test_put_bottle_rotate_and_head_override_delegates_to_base_search():
    source = PUT_BOTTLE_ROTATE_HEAD_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "put_bottle_on_cabinet_rotate_and_head"
    )
    search_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "search_and_focus_rotate_and_head_subtask"
    )

    super_calls = []
    for node in ast.walk(search_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        ):
            super_calls.append(func.attr)

    assert super_calls == ["search_and_focus_rotate_and_head_subtask"]
