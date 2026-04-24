from ._base_task import Base_Task
from .utils import *
from . import _fan_double_task_utils as fd
import numpy as np


class place_can_basket_fan_double(Base_Task):
    LAYER_SPECS = {
        "lower": {
            "inner_margin": 0.12,
            "outer_margin": 0.16,
            "max_cyl_r": 0.58,
            "theta_shrink": 0.90,
        },
        "upper": {
            "inner_margin": 0.05,
            "outer_margin": 0.07,
            "max_cyl_r": 0.72,
            "theta_shrink": 0.96,
        },
    }

    CAN_LAYER = "lower"
    BASKET_LAYER = "upper"
    CAN_MODEL_IDS = [0, 1, 2, 3, 5, 6]
    BASKET_MODEL_IDS = [0, 1]
    CAN_R_RANGE = [0.35, 0.50]
    CAN_QPOS = [0.707225, 0.706849, -0.0100455, -0.00982061]
    CAN_PRE_GRASP_DIS = 0.12
    CAN_POSE_SPECS = {
        "lower": {
            "r": 0.52,
            "theta_deg": -42.0,
            "z_offset": 0.0,
            "qpos": [0.707225, 0.706849, -0.0100455, -0.00982061],
        },
        "upper": {
            "r": 0.69,
            "theta_deg": -12.0,
            "z_offset": 0.0,
            "qpos": [0.707225, 0.706849, -0.0100455, -0.00982061],
        },
    }
    BASKET_POSE_SPECS = {
        "lower": {"r": 0.52, "theta_deg": 38.0, "z_offset": 0.0, "qpos": [0.5, 0.5, 0.5, 0.5]},
        "upper": {"r": 0.70, "theta_deg": 0.0, "z_offset": 0.0, "qpos": [0.5, 0.5, 0.5, 0.5]},
    }
    LIFT_BASKET_AFTER_PLACE = False

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    HEAD_RESET_SAVE_FREQ = -1

    PICK_LIFT_Z = 0.14
    POST_GRASP_EXTRA_LIFT_Z = 0.02
    PLACE_RETREAT_Z = 0.10
    LOWER_PLACE_WITH_PLACE_ACTOR = True
    RETURN_TO_HOMESTATE_AFTER_PLACE = True

    DIRECT_RELEASE_TCP_BACKOFF = 0.12
    DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER = 0.08
    DIRECT_RELEASE_TCP_Z_OFFSET = 0.09
    DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET = 0.15
    DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET = 0.14
    DIRECT_RELEASE_RETREAT_Z = 0.08
    DIRECT_RELEASE_R_OFFSETS = (0.0, -0.03, 0.03)
    DIRECT_RELEASE_THETA_OFFSETS_DEG = (0.0, -3.0, 3.0)
    DIRECT_RELEASE_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0)
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.18
    UPPER_PLACE_BODY_JOINT_NAME = "astribot_torso_joint_2"

    UPPER_PICK_ENTRY_Z_OFFSET = 0.09
    UPPER_PICK_PRE_GRASP_DIS = 0.11
    UPPER_PICK_GRASP_Z_BIAS = 0.0
    UPPER_PICK_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
    UPPER_PICK_GRIPPER_POS = 0.0

    SUCCESS_DIST = 0.18
    SUCCESS_Z_MIN_DELTA = 0.02

    def setup_demo(self, **kwargs):
        kwargs = fd.setup_fan_double_defaults(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.can,
                "B": self.basket,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_can",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "can_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_can_into_basket",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "can_in_basket",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Put {A} into {B}.",
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = fd.get_robot_root_xy_yaw(self)
        self.can_layer = fd.normalize_layer(self.CAN_LAYER)
        self.basket_layer = fd.normalize_layer(self.BASKET_LAYER)
        self.can_name = "071_can"
        self.basket_name = "110_basket"
        self.can_id = int(np.random.choice(self.CAN_MODEL_IDS))
        self.basket_id = int(np.random.choice(self.BASKET_MODEL_IDS))
        self.arm_tag = ArmTag({0: "left", 1: "right"}[int(np.random.randint(0, 2))])

        if self.can_layer == "lower":
            # Follow the existing place_can_basket_rotate_view pattern: put the can
            # on the side assigned to the grasping arm, away from the center band.
            can_pose = rand_pose_cyl(
                rlim=self.CAN_R_RANGE,
                thetalim=rotate_theta_side(self, side=1 if self.arm_tag == "left" else -1),
                zlim=[
                    fd.get_layer_top_z(self, self.can_layer)
                    + float(self.CAN_POSE_SPECS[self.can_layer].get("z_offset", 0.0)),
                    fd.get_layer_top_z(self, self.can_layer)
                    + float(self.CAN_POSE_SPECS[self.can_layer].get("z_offset", 0.0)),
                ],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=self.CAN_QPOS,
                rotate_rand=False,
            )
        else:
            can_pose = fd.pose_from_cyl(
                self,
                self.can_layer,
                self.CAN_POSE_SPECS[self.can_layer],
                default_qpos=self.CAN_QPOS,
                ret="pose",
            )
        self.can = create_actor(
            self,
            pose=can_pose,
            modelname=self.can_name,
            model_id=self.can_id,
            convex=True,
        )
        self.can.set_mass(0.01)

        basket_pose = fd.pose_from_cyl(
            self,
            self.basket_layer,
            self.BASKET_POSE_SPECS[self.basket_layer],
            default_qpos=[0.5, 0.5, 0.5, 0.5],
            ret="pose",
        )
        self.basket = create_actor(
            self,
            pose=basket_pose,
            modelname=self.basket_name,
            model_id=self.basket_id,
            convex=True,
            is_static=not bool(self.LIFT_BASKET_AFTER_PLACE),
        )
        self.basket.set_mass(0.5)

        self.object_start_height = float(self.can.get_pose().p[2])
        self.basket_start_height = float(self.basket.get_pose().p[2])
        self.add_prohibit_area(self.can, padding=0.08)
        self.add_prohibit_area(self.basket, padding=0.08)
        self.object_layers = {"A": self.can_layer, "B": self.basket_layer}
        self._configure_rotate_subtask_plan()

    def _basket_target_pose(self, arm_tag):
        candidates = []
        for idx in (0, 1):
            pose = self.basket.get_functional_point(idx)
            if pose is not None:
                candidates.append(np.array(pose, dtype=np.float64).reshape(-1))
        if len(candidates) == 0:
            candidates = [np.array(self.basket.get_pose().p.tolist() + [1, 0, 0, 0], dtype=np.float64)]

        can_xy = np.array(self.can.get_pose().p[:2], dtype=np.float64)
        target = min(candidates, key=lambda item: float(np.linalg.norm(item[:2] - can_xy))).copy()
        target[3:] = [-1, 0, 0, 0] if ArmTag(arm_tag) == "left" else [0.05, 0, 0, 0.99]
        return target.tolist()

    def _lift_basket_if_requested(self, arm_tag):
        if not bool(self.LIFT_BASKET_AFTER_PLACE) or not self.plan_success:
            return
        lift_arm = ArmTag(arm_tag).opposite
        self.face_object_with_torso(self.basket, joint_name_prefer=self.SCAN_JOINT_NAME)
        self.move(
            self.back_to_origin(arm_tag),
            self.grasp_actor(self.basket, arm_tag=lift_arm, pre_grasp_dis=0.08),
        )
        self.move(self.move_by_displacement(arm_tag=lift_arm, z=0.05))

    def play_once(self):
        prev_subtask_idx = None
        fd.maybe_reset_head_for_subtask(self, 1, prev_subtask_idx=prev_subtask_idx)
        can_key = fd.search_focus(self, 1)
        if can_key is None:
            self.plan_success = False
            arm_tag = self.arm_tag
        else:
            arm_tag = fd.pick_object(
                self,
                1,
                "A",
                self.can,
                self.can_layer,
                arm_tag=self.arm_tag,
                lower_grasp_kwargs={"pre_grasp_dis": self.CAN_PRE_GRASP_DIS},
            )
            if self.plan_success:
                prev_subtask_idx = 1

        if self.plan_success:
            fd.maybe_reset_head_for_subtask(self, 2, prev_subtask_idx=prev_subtask_idx)
            basket_key = fd.search_focus(self, 2)
            if basket_key is None:
                self.plan_success = False
            else:
                place_ok = fd.place_object(
                    self,
                    2,
                    "A",
                    self.can,
                    arm_tag,
                    self._basket_target_pose(arm_tag),
                    self.basket_layer,
                    place_kwargs={
                        "dis": 0.02,
                        "is_open": True,
                        "constrain": "free",
                    },
                    focus_object_key=basket_key,
                )
                if place_ok:
                    prev_subtask_idx = 2
                self._lift_basket_if_requested(arm_tag)

        self.info["info"] = {
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{B}": f"{self.basket_name}/base{self.basket_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        can_p = np.array(self.can.get_pose().p, dtype=np.float64).reshape(3)
        basket_p = np.array(self.basket.get_pose().p, dtype=np.float64).reshape(3)
        near_basket = np.linalg.norm(can_p - basket_p) < self.SUCCESS_DIST
        lifted_from_start = can_p[2] - self.object_start_height > self.SUCCESS_Z_MIN_DELTA
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return bool(near_basket and lifted_from_start and gripper_open)
