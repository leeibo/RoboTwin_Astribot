from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d


class put_bottle_on_cabinet_rotate_and_head(Base_Task):
    CABINET_MODEL_ID = 46653
    BOTTLE_MODEL_ID = 13

    CABINET_CYL_R = 0.62
    CABINET_CYL_THETA_DEG = 5.0
    CABINET_CYL_Z = 0.741
    CABINET_CYL_SPIN_DEG = 0.0

    BOTTLE_CYL_R = 0.56
    BOTTLE_CYL_THETA_DEG = -25.0
    BOTTLE_CYL_Z = 0.753
    BOTTLE_CYL_SPIN_DEG = 0.0

    CABINET_TOP_LOCAL = np.array([-0.10, 0.00, 0.471], dtype=np.float64)
    CABINET_TOP_SUCCESS_Z_MIN = 0.40
    CABINET_TOP_SUCCESS_XY_TOL = 0.14

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    HEAD_VERTICAL_CENTER_TOL = 0.08

    def setup_demo(self, **kwargs):
        kwargs.setdefault("table_shape", "fan")
        kwargs.setdefault("fan_center_on_robot", True)
        kwargs.setdefault("fan_outer_radius", 0.9)
        kwargs.setdefault("fan_inner_radius", 0.3)
        kwargs.setdefault("fan_angle_deg", 220)
        kwargs.setdefault("fan_center_deg", 90)
        kwargs = init_rotate_theta_bounds(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.bottle,
                "B": self.cabinet,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_bottle",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "bottle_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_bottle_on_cabinet_top",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "bottle_on_cabinet_top",
                    "next_subtask_id": -1,
                },
            ],
            subtask_instruction_map={
                1: "Find {A} and pick it up.",
                2: "Find {B} and place {A} on top of it.",
            },
            task_instruction="Put {A} on top of {B}.",
        )

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    @staticmethod
    def _quat_from_yaw(yaw_rad):
        return t3d.euler.euler2quat(0.0, 0.0, float(yaw_rad))

    def _world_xy_from_cyl(self, r, theta_deg):
        theta_rad = float(np.deg2rad(theta_deg))
        phi_world = float(self.robot_yaw + theta_rad)
        x = float(self.robot_root_xy[0] + float(r) * np.cos(phi_world))
        y = float(self.robot_root_xy[1] + float(r) * np.sin(phi_world))
        return x, y

    def _cabinet_pose_from_cyl(self, r, theta_deg, z, spin_deg):
        x, y = self._world_xy_from_cyl(r=r, theta_deg=theta_deg)
        radial_out_yaw = float(np.arctan2(y - self.robot_root_xy[1], x - self.robot_root_xy[0]))
        yaw = float(radial_out_yaw + np.deg2rad(spin_deg))
        return sapien.Pose([x, y, float(z)], self._quat_from_yaw(yaw))

    def _bottle_pose_from_cyl(self, r, theta_deg, z, spin_deg):
        x, y = self._world_xy_from_cyl(r=r, theta_deg=theta_deg)
        base_upright_quat = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        spin_quat = np.array(self._quat_from_yaw(np.deg2rad(spin_deg)), dtype=np.float64)
        bottle_quat = t3d.quaternions.qmult(spin_quat, base_upright_quat)
        return sapien.Pose([x, y, float(z)], bottle_quat)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.arm_tag = None
        self.default_left_homestate = list(getattr(self.robot, "left_homestate", []))
        self.default_right_homestate = list(getattr(self.robot, "right_homestate", []))

        cabinet_pose = self._cabinet_pose_from_cyl(
            r=self.CABINET_CYL_R,
            theta_deg=self.CABINET_CYL_THETA_DEG,
            z=self.CABINET_CYL_Z,
            spin_deg=self.CABINET_CYL_SPIN_DEG,
        )
        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            pose=cabinet_pose,
            modelname="036_cabinet",
            modelid=self.CABINET_MODEL_ID,
            fix_root_link=True,
        )

        bottle_pose = self._bottle_pose_from_cyl(
            r=self.BOTTLE_CYL_R,
            theta_deg=self.BOTTLE_CYL_THETA_DEG,
            z=self.BOTTLE_CYL_Z,
            spin_deg=self.BOTTLE_CYL_SPIN_DEG,
        )
        self.bottle = create_actor(
            scene=self,
            pose=bottle_pose,
            modelname="001_bottle",
            convex=True,
            model_id=self.BOTTLE_MODEL_ID,
        )
        self.bottle.set_mass(0.01)

        self.add_prohibit_area(self.bottle, padding=0.03)
        self.add_prohibit_area(self.cabinet, padding=0.03)
        self._configure_rotate_subtask_plan()

    def _apply_initial_safe_arm_posture(self):
        left_default = list(self.default_left_homestate)
        right_default = list(self.default_right_homestate)
        if len(left_default) == 0 or len(right_default) == 0:
            return

        safe_left = list(left_default)
        safe_right = list(right_default)
        safe_left[0] = -1.500
        safe_right[0] = 1.500

        self.robot.left_homestate = safe_left
        self.robot.right_homestate = safe_right
        self.robot.move_to_homestate()
        self.robot.left_homestate = left_default
        self.robot.right_homestate = right_default

    def _restore_arm_posture_for_action(self, arm_tags):
        self.robot.left_homestate = list(self.default_left_homestate)
        self.robot.right_homestate = list(self.default_right_homestate)
        seen = []
        for arm_tag in arm_tags:
            if arm_tag is None:
                continue
            arm_tag = ArmTag(arm_tag)
            if arm_tag in seen:
                continue
            seen.append(arm_tag)

        for arm_tag in seen:
            self.move(self.back_to_origin(arm_tag))

    def _reset_head_home(self):
        head_home = np.array(getattr(self.robot, "head_homestate", []), dtype=np.float64).reshape(-1)
        if head_home.shape[0] == 0:
            return False
        return self.move_head_to(head_home, settle_steps=12, save_freq=None)

    def _get_head_joint2_index(self, head_joint2_name="astribot_head_joint_2"):
        for i, joint in enumerate(getattr(self.robot, "head_joints", [])):
            if joint is not None and joint.get_name() == str(head_joint2_name):
                return i
        head_now = self._get_head_joint_state_now()
        if head_now is None or head_now.shape[0] == 0:
            return None
        return min(1, head_now.shape[0] - 1)

    def _get_head_projection_for_registry_target(self, object_key):
        camera_pose = self._get_scan_camera_pose("camera_head")
        camera_spec = self._get_scan_camera_runtime_spec("camera_head")
        if camera_pose is None or camera_spec is None:
            return None
        return self._project_rotate_registry_object(
            object_key,
            camera_pose=camera_pose,
            camera_spec=camera_spec,
        )

    def _refine_registry_target_with_head(
        self,
        object_key,
        subtask_idx,
        max_refine_iter=2,
        v_tol=0.08,
        head_joint2_name="astribot_head_joint_2",
    ):
        focus_key = None if object_key is None else str(object_key)
        if focus_key is None:
            return None

        head_joint2_idx = self._get_head_joint2_index(head_joint2_name=head_joint2_name)
        if head_joint2_idx is None:
            return focus_key

        subtask_def = self._get_rotate_subtask_def(subtask_idx)
        target_keys = [] if subtask_def is None else subtask_def.get("search_target_keys", [])
        action_target_keys = [] if subtask_def is None else subtask_def.get("action_target_keys", [])

        for _ in range(max(1, int(max_refine_iter))):
            proj = self._get_head_projection_for_registry_target(focus_key)
            current_theta = self._get_current_scan_camera_theta()
            self._set_rotate_subtask_state(
                subtask_idx=subtask_idx,
                stage=2,
                focus_object_key=focus_key,
                search_target_keys=target_keys,
                action_target_keys=action_target_keys,
                info_complete=1,
                camera_mode=2,
                camera_target_theta=(np.nan if current_theta is None else float(current_theta)),
            )

            if (
                proj is not None
                and bool(proj["inside"])
                and proj["v_norm"] is not None
                and abs(float(proj["v_norm"]) - 0.5) <= float(v_tol)
            ):
                self._refresh_rotate_discovery_from_current_view()
                return focus_key

            world_point = None
            if proj is not None and proj.get("world_point", None) is not None:
                world_point = np.array(proj["world_point"], dtype=np.float64).reshape(-1)
            else:
                obj = self._resolve_rotate_registry_object(focus_key)
                if obj is not None:
                    world_point = self._resolve_object_world_point(obj=obj)
            if world_point is None:
                break

            solve_res = self.solve_head_lookat_joint_target(world_point=world_point)
            if solve_res is None:
                break
            head_now = self._get_head_joint_state_now()
            if head_now is None:
                break
            head_target = np.array(head_now, dtype=np.float64).reshape(-1)
            solved_head_target = np.array(solve_res["target"], dtype=np.float64).reshape(-1)
            if solved_head_target.shape[0] <= head_joint2_idx or head_target.shape[0] <= head_joint2_idx:
                break
            # Keep head_joint_1 fixed so horizontal search/alignment is handled by the torso only.
            head_target[head_joint2_idx] = solved_head_target[head_joint2_idx]
            self.move_head_to(head_target, settle_steps=12)
            self._refresh_rotate_discovery_from_current_view()

        return focus_key

    def search_and_focus_rotate_and_head_subtask(
        self,
        subtask_idx,
        scan_r,
        scan_z,
        joint_name_prefer="astribot_torso_joint_2",
        max_iter=35,
        tol_yaw_rad=2e-3,
        head_joint2_name=None,
    ):
        return super().search_and_focus_rotate_and_head_subtask(
            subtask_idx,
            scan_r=scan_r,
            scan_z=scan_z,
            joint_name_prefer=joint_name_prefer,
            max_iter=max_iter,
            tol_yaw_rad=tol_yaw_rad,
            head_joint2_name=head_joint2_name,
        )

    def _get_cabinet_top_release_pose(self):
        cabinet_matrix = self.cabinet.get_pose().to_transformation_matrix()
        target_local_h = np.ones(4, dtype=np.float64)
        target_local_h[:3] = self.CABINET_TOP_LOCAL
        target_world = cabinet_matrix @ target_local_h
        bottle_quat = np.array(self.bottle.get_pose().q, dtype=np.float64).reshape(4)
        return target_world[:3].tolist() + bottle_quat.tolist()

    def _build_info(self):
        arm_tag = self.arm_tag
        if arm_tag is None:
            arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] >= 0 else "left")
        return {
            "{A}": f"001_bottle/base{self.BOTTLE_MODEL_ID}",
            "{B}": "036_cabinet/base0",
            "{a}": str(arm_tag),
        }

    def play_once(self):
        self._apply_initial_safe_arm_posture()
        self._reset_head_home()

        scan_z = float(self.SCAN_Z_BIAS + self.table_z_bias)
        bottle_key = self.search_and_focus_rotate_and_head_subtask(
            1,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] >= 0 else "left")
        if bottle_key is None:
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info

        self._restore_arm_posture_for_action([self.arm_tag, self.arm_tag.opposite])
        self.enter_rotate_action_stage(1, focus_object_key=(bottle_key or "A"))
        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.08,
                grasp_dis=-0.01,
                gripper_pos=0.2,
            )
        )
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
        self.complete_rotate_subtask(1, carried_after=["A"])

        self._reset_head_home()
        cabinet_key = self.search_and_focus_rotate_and_head_subtask(
            2,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer="astribot_torso_joint_2",
        )
        if cabinet_key is None:
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info

        target_pose = self._get_cabinet_top_release_pose()
        self.enter_rotate_action_stage(2, focus_object_key=(cabinet_key or "B"))
        self.move(
            self.place_actor(
                self.bottle,
                arm_tag=self.arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0.02,
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.06))
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = self._build_info()
        return self.info

    def check_success(self):
        if self.arm_tag is None:
            return False

        bottle_pose = self.bottle.get_functional_point(0, "pose")
        bottle_world_h = np.ones(4, dtype=np.float64)
        bottle_world_h[:3] = np.array(bottle_pose.p, dtype=np.float64).reshape(3)

        cabinet_matrix = self.cabinet.get_pose().to_transformation_matrix()
        bottle_local = np.linalg.inv(cabinet_matrix) @ bottle_world_h
        local_xy_dist = float(np.linalg.norm(bottle_local[:2] - self.CABINET_TOP_LOCAL[:2]))
        bottle_on_top = (
            float(bottle_local[2]) >= float(self.CABINET_TOP_SUCCESS_Z_MIN)
            and local_xy_dist <= float(self.CABINET_TOP_SUCCESS_XY_TOL)
        )
        gripper_open = (
            self.robot.is_left_gripper_open()
            if self.arm_tag == "left"
            else self.robot.is_right_gripper_open()
        )
        return bool(bottle_on_top and gripper_open)
