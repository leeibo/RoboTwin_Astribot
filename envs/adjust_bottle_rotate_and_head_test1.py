from .adjust_bottle_rotate_view import adjust_bottle_rotate_view
from .utils import *
import json
import numpy as np
import sapien
from pathlib import Path
import transforms3d as t3d


class adjust_bottle_rotate_and_head_test1(adjust_bottle_rotate_view):
    HEAD_VERTICAL_CENTER_TOL = 0.08
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    SPAWN_CLEARANCE = 0.01
    PICK_CONFIRM_Z_DELTA = 0.02
    FRONT_RELEASE_RADIUS = 0.65
    FRONT_RELEASE_Z_ABOVE_UPPER_TOP = 0.06
    TRANSFER_APPROACH_Z_MARGIN = 0.08
    SUCCESS_XY_TOL = 0.14
    SUCCESS_LOWER_Z_TOL = 0.02
    SUCCESS_UPPER_Z_TOL = 0.08

    def setup_demo(self, **kwargs):
        kwargs.setdefault("table_shape", "fan_double")
        kwargs.setdefault("fan_center_on_robot", True)
        kwargs.setdefault("fan_outer_radius", 0.9)
        kwargs.setdefault("fan_inner_radius", 0.3)
        kwargs.setdefault("fan_double_lower_outer_radius", kwargs.get("fan_outer_radius", 0.9))
        kwargs.setdefault("fan_double_lower_inner_radius", kwargs.get("fan_inner_radius", 0.3))
        kwargs.setdefault(
            "fan_double_upper_outer_radius",
            kwargs.get("fan_double_lower_outer_radius", kwargs.get("fan_outer_radius", 0.9)),
        )
        kwargs.setdefault(
            "fan_double_upper_inner_radius",
            kwargs.get("fan_double_lower_inner_radius", kwargs.get("fan_inner_radius", 0.3)),
        )
        kwargs.setdefault("fan_double_layer_gap", 0.30)
        kwargs.setdefault("fan_angle_deg", 220)
        kwargs.setdefault("fan_center_deg", 90)
        self.fan_double_upper_outer_radius = float(
            kwargs.get(
                "fan_double_upper_outer_radius",
                kwargs.get("fan_double_lower_outer_radius", kwargs.get("fan_outer_radius", 0.9)),
            )
        )
        self.fan_double_upper_inner_radius = float(
            kwargs.get(
                "fan_double_upper_inner_radius",
                kwargs.get("fan_double_lower_inner_radius", kwargs.get("fan_inner_radius", 0.3)),
            )
        )
        self.fan_double_layer_gap = float(kwargs.get("fan_double_layer_gap", 0.30))
        kwargs = init_rotate_theta_bounds(self, kwargs)
        super(adjust_bottle_rotate_view, self).setup_demo(**kwargs)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.initial_bottle_z = None
        self.arm_tag = None
        self.qpose_tag = int(np.random.randint(0, 2))
        self.model_id = int(np.random.choice([13, 16]))

        qposes = [[0.707, 0.0, 0.0, -0.707], [0.707, 0.0, 0.0, 0.707]]
        xlims = [[-0.12, -0.08], [0.08, 0.12]]
        bottle_qpos = qposes[self.qpose_tag]
        bottle_spawn_z = self._get_sideways_bottle_support_z(
            qpos=bottle_qpos,
            model_id=self.model_id,
            table_top_z=float(getattr(self, "rotate_table_top_z", 0.74)),
            clearance=self.SPAWN_CLEARANCE,
        )

        self.bottle = rand_create_actor(
            scene=self,
            modelname="001_bottle",
            xlim=xlims[self.qpose_tag],
            ylim=[-0.13, -0.08],
            zlim=[bottle_spawn_z, bottle_spawn_z],
            qpos=bottle_qpos,
            rotate_rand=True,
            rotate_lim=(0.0, 0.0, 0.4),
            convex=True,
            model_id=self.model_id,
        )
        self.bottle.set_mass(0.01)
        self.delay(6)
        self.add_prohibit_area(self.bottle, padding=0.08)

    def _get_upper_layer_top_z(self):
        return float(getattr(self, "rotate_table_top_z", 0.74)) + float(self.fan_double_layer_gap)

    @staticmethod
    def _pose_to_matrix(pose_like):
        if isinstance(pose_like, sapien.Pose):
            return pose_like.to_transformation_matrix()

        pose_arr = np.array(pose_like, dtype=np.float64).reshape(-1)
        if pose_arr.shape[0] != 7:
            raise ValueError(f"pose_like must contain 7 values, got shape {pose_arr.shape}")
        return sapien.Pose(pose_arr[:3], pose_arr[3:]).to_transformation_matrix()

    @staticmethod
    def _matrix_to_pose_list(matrix):
        matrix = np.array(matrix, dtype=np.float64).reshape(4, 4)
        return matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(matrix[:3, :3]).tolist()

    def _get_bottle_min_relative_z(self, qpos, model_id):
        model_data_path = Path("assets/objects/001_bottle") / f"model_data{int(model_id)}.json"
        fallback_min_relative_z = -0.045
        if not model_data_path.exists():
            return fallback_min_relative_z

        try:
            with open(model_data_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
            scale = np.array(model_data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)
            center = np.array(model_data.get("center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3) * scale
            extents = np.array(model_data.get("extents", [0.06, 0.24, 0.06]), dtype=np.float64).reshape(3) * scale
            half_extents = 0.5 * np.abs(extents)
            rot = t3d.quaternions.quat2mat(np.array(qpos, dtype=np.float64).reshape(4))
            center_world = rot @ center
            support_half_height = float(np.sum(np.abs(rot[2, :]) * half_extents))
            return float(center_world[2] - support_half_height)
        except Exception:
            return fallback_min_relative_z

    def _get_sideways_bottle_support_z(self, qpos, model_id, table_top_z, clearance=0.004):
        min_relative_z = self._get_bottle_min_relative_z(qpos=qpos, model_id=model_id)
        return float(table_top_z) - float(min_relative_z) + float(clearance)

    def _get_front_release_xy(self):
        target_world = place_point_cyl(
            [
                float(self.FRONT_RELEASE_RADIUS),
                0.0,
                float(self._get_upper_layer_top_z() + self.FRONT_RELEASE_Z_ABOVE_UPPER_TOP),
            ],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="array",
        )
        return np.array(target_world[:2], dtype=np.float64).reshape(2)

    def _get_front_release_bottle_pose(self):
        target_xy = self._get_front_release_xy()
        bottle_quat = np.array(self.bottle.get_pose().q, dtype=np.float64).reshape(4)
        target_z = float(self._get_upper_layer_top_z() + self.FRONT_RELEASE_Z_ABOVE_UPPER_TOP)
        return sapien.Pose([float(target_xy[0]), float(target_xy[1]), float(target_z)], bottle_quat)

    def _get_transfer_pose_for_target_object_pose(self, arm_tag, target_object_pose):
        ee_pose = np.array(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(-1)
        ee_matrix = self._pose_to_matrix(ee_pose)
        bottle_matrix = self.bottle.get_pose().to_transformation_matrix()
        ee_to_bottle = np.linalg.inv(ee_matrix) @ bottle_matrix
        target_object_matrix = self._pose_to_matrix(target_object_pose)
        target_ee_matrix = target_object_matrix @ np.linalg.inv(ee_to_bottle)
        return self._matrix_to_pose_list(target_ee_matrix)

    def _get_front_release_transfer_waypoints(self, arm_tag):
        release_bottle_pose = self._get_front_release_bottle_pose()
        approach_bottle_pose = sapien.Pose(
            [
                float(release_bottle_pose.p[0]),
                float(release_bottle_pose.p[1]),
                float(release_bottle_pose.p[2]) + float(self.TRANSFER_APPROACH_Z_MARGIN),
            ],
            release_bottle_pose.q,
        )
        return {
            "target_point": release_bottle_pose.p.tolist(),
            "approach_pose": self._get_transfer_pose_for_target_object_pose(arm_tag, approach_bottle_pose),
            "release_pose": self._get_transfer_pose_for_target_object_pose(arm_tag, release_bottle_pose),
        }

    def _is_bottle_grasped_and_lifted(self):
        if self.arm_tag is None or self.initial_bottle_z is None:
            return False

        bottle_z = float(self.bottle.get_pose().p[2])
        gripper_closed = self.is_right_gripper_close() if self.arm_tag == "right" else self.is_left_gripper_close()
        return bool(gripper_closed and bottle_z > (float(self.initial_bottle_z) + float(self.PICK_CONFIRM_Z_DELTA)))

    def _transport_bottle_to_front_release(self):
        waypoints = self._get_front_release_transfer_waypoints(self.arm_tag)
        self.face_world_point_with_torso(
            waypoints["target_point"],
            joint_name_prefer=self.SCAN_JOINT_NAME,
            max_iter=35,
            tol_yaw_rad=2e-3,
        )
        self.move(self.move_to_pose(arm_tag=self.arm_tag, target_pose=waypoints["approach_pose"]))
        if not self.plan_success:
            return False

        self.move(self.move_to_pose(arm_tag=self.arm_tag, target_pose=waypoints["release_pose"]))
        if not self.plan_success:
            return False

        self.move(self.open_gripper(self.arm_tag))
        self.delay(6)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.08))
        return bool(self.plan_success)

    def _build_info(self):
        arm_tag = self.arm_tag
        if arm_tag is None:
            arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] >= 0 else "left")
        return {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(arm_tag),
        }

    def _get_head_joint2_index(self, head_joint2_name="astribot_head_joint_2"):
        for i, joint in enumerate(getattr(self.robot, "head_joints", [])):
            if joint is not None and joint.get_name() == str(head_joint2_name):
                return i
        head_now = self._get_head_joint_state_now()
        if head_now is None or head_now.shape[0] == 0:
            return None
        return min(1, head_now.shape[0] - 1)

    def _get_bottle_head_projection(self):
        camera_pose = self._get_scan_camera_pose("camera_head")
        camera_spec = self._get_scan_camera_runtime_spec("camera_head")
        if camera_pose is None or camera_spec is None:
            return None

        try:
            (u_norm, v_norm), debug = project_object_to_image_uv(
                obj=self.bottle,
                camera_pose=camera_pose,
                image_w=int(camera_spec["w"]),
                image_h=int(camera_spec["h"]),
                fovy_rad=float(camera_spec["fovy_rad"]),
                mode="aabb",
                far=camera_spec.get("far", None),
                ret_debug=True,
            )
        except Exception:
            return None

        return {
            "inside": bool(debug.get("inside", False)),
            "u_norm": None if not np.isfinite(u_norm) else float(u_norm),
            "v_norm": None if not np.isfinite(v_norm) else float(v_norm),
            "world_point": np.array(
                debug.get("world_point", self._resolve_object_world_point(self.bottle)),
                dtype=np.float64,
            ).reshape(-1).tolist(),
        }

    def _coarse_search_bottle_with_head_joint2(
        self,
        head_joint2_name="astribot_head_joint_2",
        step_rad=0.15,
        settle_steps=12,
    ):
        proj = self._get_bottle_head_projection()
        if proj is not None and proj["inside"]:
            return proj

        head_now = self._get_head_joint_state_now()
        head_joint2_idx = self._get_head_joint2_index(head_joint2_name=head_joint2_name)
        if head_now is None or head_joint2_idx is None:
            return None

        lower, upper = -1.22, 1.22
        try:
            limits = self.robot.head_joints[head_joint2_idx].get_limits()
            if limits is not None and len(limits) > 0:
                lower = float(limits[0][0])
                upper = float(limits[0][1])
        except Exception:
            pass

        current_joint2 = float(np.clip(head_now[head_joint2_idx], lower, upper))
        target_values = list(
            np.arange(current_joint2 - abs(step_rad), lower - 1e-9, -abs(step_rad), dtype=np.float64)
        )
        if len(target_values) == 0 or abs(target_values[-1] - lower) > 1e-6:
            target_values.append(lower)

        for target_joint2 in target_values:
            head_target = np.array(head_now, dtype=np.float64)
            head_target[head_joint2_idx] = float(np.clip(target_joint2, lower, upper))
            self.move_head_to(head_target, settle_steps=settle_steps)
            proj = self._get_bottle_head_projection()
            if proj is not None and proj["inside"]:
                return proj
        return None

    def _precisely_focus_bottle_with_head_joint2(
        self,
        head_joint2_name="astribot_head_joint_2",
        v_tol=None,
        settle_steps=12,
        max_refine_iter=2,
    ):
        if v_tol is None:
            v_tol = float(self.HEAD_VERTICAL_CENTER_TOL)
        head_joint2_idx = self._get_head_joint2_index(head_joint2_name=head_joint2_name)
        if head_joint2_idx is None:
            return False

        for _ in range(max(1, int(max_refine_iter))):
            proj = self._get_bottle_head_projection()
            if proj is None or (not proj["inside"]):
                return False
            if proj["v_norm"] is not None and abs(float(proj["v_norm"]) - 0.5) <= float(v_tol):
                return True

            solve_res = self.solve_head_lookat_joint_target(world_point=proj["world_point"])
            if solve_res is None:
                return False

            head_now = self._get_head_joint_state_now()
            if head_now is None:
                return False

            solved_head_target = np.array(solve_res["target"], dtype=np.float64).reshape(-1)
            if solved_head_target.shape[0] <= head_joint2_idx:
                return False

            head_target = np.array(head_now, dtype=np.float64)
            head_target[head_joint2_idx] = solved_head_target[head_joint2_idx]
            self.move_head_to(head_target, settle_steps=settle_steps)

        proj = self._get_bottle_head_projection()
        return bool(proj is not None and proj["inside"])

    def play_once(self):
        self.face_object_with_torso(
            self.bottle,
            joint_name_prefer=self.SCAN_JOINT_NAME,
            max_iter=35,
            tol_yaw_rad=2e-3,
        )

        proj = self._coarse_search_bottle_with_head_joint2()
        if proj is None or (not proj["inside"]):
            self.plan_success = False
            self.arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] >= 0 else "left")
            self.info["info"] = self._build_info()
            return self.info

        self._precisely_focus_bottle_with_head_joint2()

        self.arm_tag = ArmTag("right" if self.qpose_tag == 1 else "left")
        self.initial_bottle_z = float(self.bottle.get_pose().p[2])

        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.08,
                grasp_dis=-0.01,
                gripper_pos=0.2,
            )
        )
        self.delay(3)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.06))
        self.delay(2)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.06))
        self.delay(2)
        if not self._is_bottle_grasped_and_lifted():
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info

        if not self._transport_bottle_to_front_release():
            self.info["info"] = self._build_info()
            return self.info

        self.info["info"] = self._build_info()
        return self.info

    def check_success(self):
        if self.arm_tag is None:
            return False

        bottle_pose = self.bottle.get_pose()
        target_xy = self._get_front_release_xy()
        xy_dist = float(np.linalg.norm(np.array(bottle_pose.p[:2], dtype=np.float64) - target_xy))
        upper_top_z = self._get_upper_layer_top_z()
        bottle_lowest_z = float(bottle_pose.p[2]) + float(self._get_bottle_min_relative_z(bottle_pose.q, self.model_id))
        gripper_open = self.is_right_gripper_open() if self.arm_tag == "right" else self.is_left_gripper_open()
        return bool(
            gripper_open
            and xy_dist <= float(self.SUCCESS_XY_TOL)
            and bottle_lowest_z >= (upper_top_z - float(self.SUCCESS_LOWER_Z_TOL))
            and bottle_lowest_z <= (upper_top_z + float(self.SUCCESS_UPPER_Z_TOL))
        )
