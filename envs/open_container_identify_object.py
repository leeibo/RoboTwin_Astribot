import numpy as np
import sapien.core as sapien

from .search_object import search_object
from .utils import *


class open_container_identify_object(search_object):
    """Open a closed container and identify which object is inside."""

    # Use a slightly larger colored block than the generic search_object demo so
    # the extracted object is visible in the observer/head videos.
    OBJECT_HALF_SIZE = 0.024
    OBJECT_Z_BIAS = OBJECT_HALF_SIZE + 0.002
    CONFIRM_CYL_R = 0.50
    CONFIRM_CYL_THETA_ABS = 0.42
    CONFIRM_Z = 1.02
    CONFIRM_HOLD_STEPS = 12

    OBJECT_VARIANTS = (
        {
            "kind": "block",
            "label": "red block",
            "color": (0.90, 0.20, 0.20),
            "outward_offset": 0.015,
            "surface_z_offset": OBJECT_Z_BIAS,
            "mass": search_object.OBJECT_MASS,
        },
        {
            "kind": "block",
            "label": "blue block",
            "color": (0.20, 0.45, 0.92),
            "outward_offset": 0.015,
            "surface_z_offset": OBJECT_Z_BIAS,
            "mass": search_object.OBJECT_MASS,
        },
        {
            "kind": "block",
            "label": "yellow block",
            "color": (0.92, 0.74, 0.18),
            "outward_offset": 0.015,
            "surface_z_offset": OBJECT_Z_BIAS,
            "mass": search_object.OBJECT_MASS,
        },
    )

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object,
                "B": self.cabinet,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "confirm_contents_hidden",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": [],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "inside_object_not_visible",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "open_container",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "container_opened",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "identify_inside_object",
                    "instruction_idx": 3,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": False,
                    "done_when": "inside_object_extracted_for_identification",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Open {B} and identify what object is inside.",
        )

    def _create_search_target_object(self, drawer_pose, drawer_outward_dir):
        variant = dict(self.OBJECT_VARIANTS[int(np.random.randint(len(self.OBJECT_VARIANTS)))])
        block_pose = self._build_drawer_object_pose(
            drawer_pose=drawer_pose,
            drawer_outward_dir=drawer_outward_dir,
            outward_offset=float(variant.get("outward_offset", self.OBJECT_OUTER_EDGE_OFFSET)),
            surface_z_offset=float(variant.get("surface_z_offset", self.OBJECT_Z_BIAS)),
            quat=self._compose_object_quat(drawer_pose),
        )
        block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(self.OBJECT_HALF_SIZE, self.OBJECT_HALF_SIZE, self.OBJECT_HALF_SIZE),
            color=tuple(float(channel) for channel in variant["color"]),
            name=f"hidden_{str(variant['label']).replace(' ', '_')}",
        )
        block.set_mass(float(variant.get("mass", self.OBJECT_MASS)))
        self.selected_modelname = None
        self.selected_model_id = None
        return block, str(variant["label"])

    def load_actors(self):
        self.identified_object_label = None
        super().load_actors()

    def _build_info(self):
        if self.cabinet_arm_tag is None:
            self.cabinet_arm_tag = self._get_cabinet_arm_tag()
        return {
            "{A}": str(getattr(self, "object_label", self.OBJECT_LABEL)),
            "{B}": "036_cabinet/base0",
            "{a}": str(self.object_arm_tag) if self.object_arm_tag is not None else "opposite arm",
            "{b}": str(self.cabinet_arm_tag),
        }

    def _point_from_cyl(self, r, theta, z):
        return np.array(
            place_point_cyl(
                [float(r), float(theta), float(z)],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="list",
            ),
            dtype=np.float64,
        ).reshape(3)

    def _get_confirmation_point(self):
        # Present the extracted object on the side of the arm that is holding it
        # rather than directly above/behind the cabinet.  This makes the color
        # visible in the external demo camera instead of being hidden by the
        # cabinet top or black gripper.
        theta_sign = -1.0 if str(self.object_arm_tag) == "right" else 1.0
        return self._point_from_cyl(
            float(self.CONFIRM_CYL_R),
            theta_sign * float(self.CONFIRM_CYL_THETA_ABS),
            float(self.CONFIRM_Z),
        )

    def _target_theta(self, point):
        local = world_to_robot(
            np.array(point, dtype=np.float64).reshape(3).tolist(),
            self.robot_root_xy,
            self.robot_yaw,
        )
        return float(local[1])

    def _move_carried_object_center_to(self, target_center):
        if self.object_arm_tag is None:
            return False
        target_center = np.array(target_center, dtype=np.float64).reshape(3)
        ee_pose = self._get_arm_ee_pose(self.object_arm_tag)
        object_center = np.array(self.object.get_pose().p, dtype=np.float64).reshape(3)
        target_ee_pose = np.array(ee_pose, dtype=np.float64).reshape(7).copy()
        target_ee_pose[:3] += target_center - object_center
        if not self.move(self.move_to_pose(arm_tag=self.object_arm_tag, target_pose=target_ee_pose.tolist())):
            return False
        # Keep the rendered object exactly at the presentation point during the
        # confirmation hold.  The arm has already moved there; this removes small
        # grasp/physics offsets that can otherwise hide the colored block inside
        # the fingers in the demo frame.
        self.object.actor.set_pose(sapien.Pose(target_center.tolist(), list(self.object.get_pose().q)))
        self.delay(int(self.CONFIRM_HOLD_STEPS), save_freq=1)
        return True

    def _present_extracted_object_for_confirmation(self):
        target_center = self._get_confirmation_point()
        self._set_rotate_subtask_state(
            subtask_idx=3,
            stage=2,
            focus_object_key="A",
            search_target_keys=["A"],
            action_target_keys=["A"],
            info_complete=1,
            camera_mode=2,
            camera_target_theta=self._target_theta(target_center),
        )
        self.face_world_point_with_torso(
            target_center,
            max_iter=35,
            tol_yaw_rad=2e-3,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        return self._move_carried_object_center_to(target_center)

    def play_once(self):
        scan_z = float(self.SCAN_Z_BIAS + self.table_z_bias)
        self._reset_head_to_home_pose(save_freq=None)

        object_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is not None:
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(1, carried_after=[])

        self._reset_head_to_home_pose(save_freq=None)
        cabinet_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if cabinet_key is None or not self._open_cabinet_drawer(cabinet_key):
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(2, carried_after=[])

        self._reset_head_to_home_pose(save_freq=None)
        object_key = self.search_and_focus_rotate_subtask(
            3,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is None:
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.identified_object_label = str(self.object_label)
        if not self._grasp_and_lift_object(object_key):
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        if not self._present_extracted_object_for_confirmation():
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        # The object is now outside the container and held up for confirmation.
        self.identified_object_label = str(self.object_label)
        self.complete_rotate_subtask(3, carried_after=["A"])

        self.info["info"] = self._build_info()
        return self.info

    def check_success(self):
        return bool(
            getattr(self, "cabinet_opened", False)
            and getattr(self, "identified_object_label", None) == str(getattr(self, "object_label", ""))
            and super().check_success()
        )
