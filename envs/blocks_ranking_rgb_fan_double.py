from ._base_task import Base_Task
from .utils import *
from . import _fan_double_task_utils as fd
import numpy as np
import sapien


class blocks_ranking_rgb_fan_double(Base_Task):
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
    BLOCK_SIZE_RANGE = (0.020, 0.026)
    BLOCK_DEFS = (
        {"key": "A", "label": "red block", "color": (1.0, 0.0, 0.0)},
        {"key": "B", "label": "green block", "color": (0.0, 0.85, 0.05)},
        {"key": "C", "label": "blue block", "color": (0.0, 0.15, 1.0)},
    )
    BLOCK_SPAWN_MIN_DIST_SQ = 0.014

    TARGET_LAYER = "lower"
    TARGET_ROW_SPEC = {
        "r": 0.52,
        "theta_deg": 34.0,
        "gap_theta_deg": 13.0,
        "z_offset": 0.0,
    }

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    HEAD_RESET_SAVE_FREQ = -1

    PICK_PRE_GRASP_DIS = 0.09
    PICK_GRASP_DIS = 0.01
    PICK_LIFT_Z = 0.10
    POST_GRASP_EXTRA_LIFT_Z = 0.04
    PLACE_RETREAT_Z = 0.08
    LOWER_PLACE_WITH_PLACE_ACTOR = True
    LOWER_PLACE_FUNCTIONAL_POINT_ID = 0
    LOWER_PLACE_PRE_DIS = 0.18
    LOWER_PLACE_DIS = 0.03
    LOWER_PLACE_CONSTRAIN = "free"
    LOWER_PLACE_PRE_DIS_AXIS = "fp"
    LOWER_PLACE_IS_OPEN = True
    LOWER_PLACE_RETREAT_Z = 0.12
    LOWER_PLACE_RETREAT_MOVE_AXIS = "arm"
    RETURN_TO_HOMESTATE_AFTER_PLACE = True

    DIRECT_RELEASE_TCP_BACKOFF = 0.12
    DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER = 0.08
    DIRECT_RELEASE_TCP_Z_OFFSET = 0.06
    DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET = 0.10
    DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET = 0.10
    DIRECT_RELEASE_RETREAT_Z = 0.06
    DIRECT_RELEASE_R_OFFSETS = (0.0, -0.03, 0.03)
    DIRECT_RELEASE_THETA_OFFSETS_DEG = (0.0, -3.0, 3.0)
    DIRECT_RELEASE_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0)

    UPPER_TO_LOWER_USE_HOVER_DROP = True
    UPPER_TO_LOWER_HOVER_Z_OFFSETS = (0.06, 0.08, 0.10)
    UPPER_TO_LOWER_DROP_YAW_OFFSETS_DEG = (0.0, 90.0, -90.0, 180.0)
    UPPER_TO_LOWER_RELEASE_DELAY_STEPS = 15
    UPPER_TO_LOWER_RELEASE_RETREAT_Z = 0.08

    UPPER_PICK_ENTRY_Z_OFFSET = 0.10
    UPPER_PICK_PRE_GRASP_DIS = 0.10
    UPPER_PICK_GRASP_Z_BIAS = 0.02
    UPPER_PICK_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
    UPPER_PICK_GRIPPER_POS = -0.01

    SUCCESS_XY_TOL = 0.09
    SUCCESS_Z_TOL = 0.08

    def setup_demo(self, **kwargs):
        kwargs = fd.setup_fan_double_defaults(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _sample_block_layers(self):
        non_anchor_layers = ("lower", "upper") if int(np.random.randint(0, 2)) == 0 else ("upper", "lower")
        return {
            "A": "lower",
            "B": non_anchor_layers[0],
            "C": non_anchor_layers[1],
        }

    def _target_point(self, target_idx, z_offset=0.0):
        spec = dict(self.TARGET_ROW_SPEC)
        theta_deg = float(spec["theta_deg"]) - float(spec["gap_theta_deg"]) * float(target_idx)
        return place_point_cyl(
            [
                float(spec["r"]),
                float(np.deg2rad(theta_deg)),
                fd.get_layer_top_z(self, self.TARGET_LAYER) + float(spec.get("z_offset", 0.0)) + float(z_offset),
            ],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="array",
        )

    def _target_pose(self, target_idx):
        return fd.pose_list_from_point(self._target_point(target_idx), quat=[0, 1, 0, 0])

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.blocks["A"],
                "B": self.blocks["B"],
                "C": self.blocks["C"],
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_green_block",
                    "instruction_idx": 1,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["B"],
                    "allow_stage2_from_memory": True,
                    "done_when": "green_block_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_green_block_right_of_red",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["B"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "green_block_placed",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "pick_blue_block",
                    "instruction_idx": 3,
                    "search_target_keys": ["C"],
                    "action_target_keys": ["C"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["C"],
                    "allow_stage2_from_memory": True,
                    "done_when": "blue_block_grasped",
                    "next_subtask_id": 4,
                },
                {
                    "id": 4,
                    "name": "place_blue_block_right_of_green",
                    "instruction_idx": 4,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B", "C"],
                    "required_carried_keys": ["C"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "blue_block_placed",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Arrange {A}, {B}, and {C} from left to right by color.",
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = fd.get_robot_root_xy_yaw(self)
        sampled_layers = self._sample_block_layers()
        block_size = float(np.random.uniform(*self.BLOCK_SIZE_RANGE))
        self.blocks = {}
        self.block_layers = {}
        self.block_size = block_size
        self.target_poses = {
            "A": self._target_pose(0),
            "B": self._target_pose(1),
            "C": self._target_pose(2),
        }
        target_xy_lst = [np.array(self.target_poses[key][:2], dtype=np.float64) for key in ("A", "B", "C")]

        existing_pose_lst = []
        for idx, block_def in enumerate(self.BLOCK_DEFS):
            key = block_def["key"]
            layer_name = fd.normalize_layer(sampled_layers[key])
            if key == "A":
                pose_point = self._target_point(0, z_offset=block_size)
                block_pose = sapien.Pose(pose_point.tolist(), [1, 0, 0, 0])
            else:
                block_pose = fd.sample_pose_on_layer(
                    self,
                    layer_name,
                    z_offset=block_size,
                    existing_pose_lst=existing_pose_lst,
                    avoid_xy_lst=target_xy_lst,
                    min_dist_sq=self.BLOCK_SPAWN_MIN_DIST_SQ,
                )
            block = create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_size, block_size, block_size),
                color=block_def["color"],
                name=f"{key}_block",
            )
            block.set_mass(0.03)
            self.blocks[key] = block
            self.block_layers[key] = layer_name
            existing_pose_lst.append(block_pose)
            self.add_prohibit_area(block, padding=0.06)

        self.block1 = self.blocks["A"]
        self.block2 = self.blocks["B"]
        self.block3 = self.blocks["C"]
        self.object_layers = dict(self.block_layers)
        self._configure_rotate_subtask_plan()

    def _pick(self, subtask_idx, key):
        return fd.pick_object(
            self,
            subtask_idx,
            key,
            self.blocks[key],
            self.block_layers[key],
            lower_grasp_kwargs={
                "pre_grasp_dis": self.PICK_PRE_GRASP_DIS,
                "grasp_dis": self.PICK_GRASP_DIS,
            },
        )

    def _place(self, subtask_idx, key, arm_tag, focus_key):
        return fd.place_object(
            self,
            subtask_idx,
            key,
            self.blocks[key],
            arm_tag,
            self.target_poses[key],
            self.TARGET_LAYER,
            place_kwargs={
                "functional_point_id": self.LOWER_PLACE_FUNCTIONAL_POINT_ID,
                "pre_dis": self.LOWER_PLACE_PRE_DIS,
                "dis": self.LOWER_PLACE_DIS,
                "constrain": self.LOWER_PLACE_CONSTRAIN,
                "pre_dis_axis": self.LOWER_PLACE_PRE_DIS_AXIS,
                "is_open": bool(self.LOWER_PLACE_IS_OPEN),
            },
            focus_object_key=focus_key,
        )

    def play_once(self):
        arm_tag_b = ArmTag("left")
        arm_tag_c = ArmTag("left")
        prev_subtask_idx = None
        for pick_idx, place_idx, key, focus_key in [(1, 2, "B", "A"), (3, 4, "C", "B")]:
            fd.maybe_reset_head_for_subtask(self, pick_idx, prev_subtask_idx=prev_subtask_idx)
            found_key = fd.search_focus(self, pick_idx)
            if found_key is None:
                self.plan_success = False
                break
            arm_tag = self._pick(pick_idx, key)
            if not self.plan_success:
                break
            prev_subtask_idx = pick_idx

            fd.maybe_reset_head_for_subtask(self, place_idx, prev_subtask_idx=prev_subtask_idx)
            found_focus = fd.search_focus(self, place_idx)
            if found_focus is None:
                self.plan_success = False
                break
            if not self._place(place_idx, key, arm_tag, found_focus or focus_key):
                break
            prev_subtask_idx = place_idx
            if key == "B":
                arm_tag_b = arm_tag
            else:
                arm_tag_c = arm_tag

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": str(fd.get_object_arm_tag(self, self.blocks["A"])),
            "{b}": str(arm_tag_b),
            "{c}": str(arm_tag_c),
        }
        return self.info

    def check_success(self):
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        xy_ok = True
        z_ok = True
        for key in ("A", "B", "C"):
            pose = np.array(self.blocks[key].get_pose().p, dtype=np.float64).reshape(3)
            target = np.array(self.target_poses[key][:3], dtype=np.float64).reshape(3)
            xy_ok = xy_ok and bool(np.linalg.norm(pose[:2] - target[:2]) < self.SUCCESS_XY_TOL)
            z_ok = z_ok and bool(abs(pose[2] - target[2]) < self.SUCCESS_Z_TOL)

        c1 = world_to_robot(self.blocks["A"].get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        c2 = world_to_robot(self.blocks["B"].get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        c3 = world_to_robot(self.blocks["C"].get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        ordered = c1[1] > c2[1] > c3[1]
        same_arc = abs(c1[0] - c2[0]) < 0.16 and abs(c2[0] - c3[0]) < 0.16
        return bool(gripper_open and xy_ok and z_ok and ordered and same_arc)
