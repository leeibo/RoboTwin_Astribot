from ._base_task import Base_Task
from .utils import *
from . import _fan_double_task_utils as fd
import numpy as np


class place_object_basket_fan_double(Base_Task):
    LAYER_SPECS = {
        "lower": {
            "inner_margin": 0.12,
            "outer_margin": 0.16,
            "max_cyl_r": 0.5,
            "theta_shrink": 0.90,
        },
        "upper": {
            "inner_margin": 0.05,
            "outer_margin": 0.07,
            "max_cyl_r": 0.68,
            "theta_shrink": 0.96,
        },
    }

    OBJECT_LAYER = "lower"
    BASKET_LAYER = "upper"
    OBJECT_CANDIDATES = {
        "081_playingcards": [0, 1, 2],
        "057_toycar": [0, 1, 2, 3, 4, 5],
    }
    BASKET_MODEL_IDS = [0, 1]
    OBJECT_R_RANGE = [0.40, 0.50]
    OBJECT_QPOS = [0.707225, 0.706849, -0.0100455, -0.00982061]
    OBJECT_ROTATE_RAND = True
    OBJECT_ROTATE_LIM = [0.0, np.pi / 6, 0.0]
    OBJECT_PRE_GRASP_DIS = 0.10
    PICK_SUCCESS_Z_DELTA = 0.03
    OBJECT_POSE_SPECS = {
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

    PICK_LIFT_Z = 0.20
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
    SUCCESS_Z_MIN_DELTA = 0.015

    def setup_demo(self, **kwargs):
        kwargs = fd.setup_fan_double_defaults(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object,
                "B": self.basket,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_object",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "object_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_object_into_basket",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "object_in_basket",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Put {A} into {B}.",
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = fd.get_robot_root_xy_yaw(self)
        self.object_layer = fd.normalize_layer(self.OBJECT_LAYER)
        self.basket_layer = fd.normalize_layer(self.BASKET_LAYER)
        self.object_name = str(np.random.choice(list(self.OBJECT_CANDIDATES.keys())))
        self.object_id = int(np.random.choice(self.OBJECT_CANDIDATES[self.object_name]))
        self.basket_name = "110_basket"
        self.basket_id = int(np.random.choice(self.BASKET_MODEL_IDS))
        self.arm_tag = ArmTag({0: "left", 1: "right"}[int(np.random.randint(0, 2))])

        if self.object_layer == "lower":
            # Follow place_object_basket_rotate_view: spawn the object on the
            # side assigned to the grasping arm, not near the center band.
            object_z = (
                fd.get_layer_top_z(self, self.object_layer)
                + float(self.OBJECT_POSE_SPECS[self.object_layer].get("z_offset", 0.0))
            )
            object_pose = rand_pose_cyl(
                rlim=self.OBJECT_R_RANGE,
                thetalim=rotate_theta_side(self, side=1 if self.arm_tag == "left" else -1),
                zlim=[object_z, object_z],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=self.OBJECT_QPOS,
                rotate_rand=self.OBJECT_ROTATE_RAND,
                rotate_lim=self.OBJECT_ROTATE_LIM,
            )
        else:
            object_pose = fd.pose_from_cyl(
                self,
                self.object_layer,
                self.OBJECT_POSE_SPECS[self.object_layer],
                default_qpos=self.OBJECT_QPOS,
                ret="pose",
            )
        self.object = create_actor(
            self,
            pose=object_pose,
            modelname=self.object_name,
            model_id=self.object_id,
            convex=True,
        )
        self.object.set_mass(0.01)

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

        self.object_start_height = float(self.object.get_pose().p[2])
        self.basket_start_height = float(self.basket.get_pose().p[2])
        self.add_prohibit_area(self.object, padding=0.08)
        self.add_prohibit_area(self.basket, padding=0.08)
        self.object_layers = {"A": self.object_layer, "B": self.basket_layer}
        self._configure_rotate_subtask_plan()

    def _basket_target_pose(self, arm_tag):
        candidates = []
        for idx in (0, 1):
            pose = self.basket.get_functional_point(idx)
            if pose is not None:
                candidates.append(np.array(pose, dtype=np.float64).reshape(-1))
        if len(candidates) == 0:
            candidates = [np.array(self.basket.get_pose().p.tolist() + [1, 0, 0, 0], dtype=np.float64)]

        obj_xy = np.array(self.object.get_pose().p[:2], dtype=np.float64)
        target = min(candidates, key=lambda item: float(np.linalg.norm(item[:2] - obj_xy))).copy()
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

    def _object_lifted_after_pick(self):
        object_z = float(self.object.get_pose().p[2])
        return bool(object_z - float(self.object_start_height) > float(self.PICK_SUCCESS_Z_DELTA))

    def _should_enforce_rotate_stage1_search_order(self, subtask_idx, subtask_def=None):
        if int(subtask_idx) != 2:
            return False
        if self.object_layer != "lower" or self.basket_layer != "upper":
            return False
        return bool(self._has_pending_lower_rotate_search_states())

    def _should_skip_rotate_head_home_reset(self, subtask_idx, prev_subtask_idx=None):
        if prev_subtask_idx is None or int(prev_subtask_idx) != 1:
            return False
        return bool(self._should_enforce_rotate_stage1_search_order(subtask_idx))

    def play_once(self):
        prev_subtask_idx = None
        fd.maybe_reset_head_for_subtask(self, 1, prev_subtask_idx=prev_subtask_idx)
        object_key = fd.search_focus(self, 1)
        if object_key is None:
            self.plan_success = False
            arm_tag = self.arm_tag
        else:
            arm_tag = fd.pick_object(
                self,
                1,
                "A",
                self.object,
                self.object_layer,
                arm_tag=self.arm_tag,
                lower_grasp_kwargs={"pre_grasp_dis": self.OBJECT_PRE_GRASP_DIS},
            )
            if self.plan_success and not self._object_lifted_after_pick():
                self.plan_success = False
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
                    self.object,
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
            "{A}": f"{self.object_name}/base{self.object_id}",
            "{B}": f"{self.basket_name}/base{self.basket_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        obj_p = np.array(self.object.get_pose().p, dtype=np.float64).reshape(3)
        basket_p = np.array(self.basket.get_pose().p, dtype=np.float64).reshape(3)
        near_basket = np.linalg.norm(obj_p - basket_p) < self.SUCCESS_DIST
        lifted_from_start = obj_p[2] - self.object_start_height > self.SUCCESS_Z_MIN_DELTA
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return bool(near_basket and lifted_from_start and gripper_open)
