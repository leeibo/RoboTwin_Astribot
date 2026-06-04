import numpy as np
import sapien.core as sapien

from ._base_task import Base_Task
from ._info_task_helpers import RMBenchButtonMixin
from .utils import *


class count_target_press_button(RMBenchButtonMixin, Base_Task):
    """Count green target blocks, then press the RMBench button that many times."""

    ROTATE_TABLE_SHAPE = "fan"
    TARGET_COLOR = (0.10, 0.80, 0.20)
    DISTRACTOR_COLORS = (
        (0.90, 0.20, 0.20),
        (0.20, 0.45, 0.92),
        (0.92, 0.74, 0.18),
    )
    BLOCK_HALF_SIZE = 0.022
    TARGET_COUNT_RANGE = (1, 4)
    DISTRACTOR_COUNT = 3
    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.88
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    ROTATE_SCAN_SCENE_FALLBACK_THETAS = (0.72, 0.24, -0.24, -0.72)
    # Match the click_bell-style layout: put the press target in a reachable
    # side band instead of the center line, then choose the arm by object side.
    BUTTON_R = 0.47
    BUTTON_THETA = 0.42
    BUTTON_ARM = None

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        override = kwargs.get("count_target_count_override", None)
        self.count_target_count_override = None if override is None else int(override)
        super()._init_task_env_(**kwargs)

    def _block_pose_from_cyl(self, r, theta):
        z = 0.74 + float(self.BLOCK_HALF_SIZE) + 0.002
        point = place_point_cyl(
            [float(r), float(theta), z],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        return sapien.Pose(point, [1, 0, 0, 0])

    def _create_count_block(self, key, pose, color, label):
        block = create_box(
            scene=self,
            pose=pose,
            half_size=(self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE),
            color=tuple(float(v) for v in color),
            name=str(key),
        )
        block.set_mass(0.03)
        self.object_layers[str(key)] = "lower"
        self.object_labels[str(key)] = str(label)
        self.add_prohibit_area(block, padding=0.04)
        return block

    def _configure_rotate_subtask_plan(self):
        registry = {}
        for key, block in self.target_blocks.items():
            registry[key] = block
        for key, block in self.distractor_blocks.items():
            registry[key] = block
        registry["BTN"] = self.button

        self.configure_rotate_subtask_plan(
            object_registry=registry,
            subtask_defs=[
                {
                    "id": 1,
                    "name": "count_green_target_blocks",
                    "instruction_idx": 1,
                    "search_target_keys": list(self.target_blocks.keys()) + list(self.distractor_blocks.keys()),
                    "action_target_keys": [],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "target_count_known",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "press_button_count_times",
                    "instruction_idx": 2,
                    "search_target_keys": ["BTN"],
                    "action_target_keys": ["BTN"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "button_pressed_target_count",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction=(
                "Count the green target blocks on the table, then press the red button "
                "the same number of times."
            ),
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers = {}
        self.object_labels = {}
        self.button_press_count = 0
        self.button_press_flag = False
        if getattr(self, "count_target_count_override", None) is None:
            self.target_count = int(np.random.randint(self.TARGET_COUNT_RANGE[0], self.TARGET_COUNT_RANGE[1] + 1))
        else:
            lo, hi = self.TARGET_COUNT_RANGE
            self.target_count = int(np.clip(int(self.count_target_count_override), int(lo), int(hi)))
        self.target_blocks = {}
        self.distractor_blocks = {}

        slots = [
            (0.46, -0.62),
            (0.52, -0.34),
            (0.58, -0.08),
            (0.50, 0.18),
            (0.58, 0.42),
            (0.66, -0.44),
            (0.66, 0.04),
        ]
        order = np.random.permutation(len(slots)).tolist()
        target_slots = [slots[idx] for idx in order[: self.target_count]]
        distractor_slots = [
            slots[idx]
            for idx in order[self.target_count : self.target_count + int(self.DISTRACTOR_COUNT)]
        ]

        for idx, (r, theta) in enumerate(target_slots, start=1):
            key = f"T{idx}"
            self.target_blocks[key] = self._create_count_block(
                key=key,
                pose=self._block_pose_from_cyl(r, theta),
                color=self.TARGET_COLOR,
                label="green target block",
            )

        for idx, (r, theta) in enumerate(distractor_slots, start=1):
            key = f"D{idx}"
            color = self.DISTRACTOR_COLORS[(idx - 1) % len(self.DISTRACTOR_COLORS)]
            self.distractor_blocks[key] = self._create_count_block(
                key=key,
                pose=self._block_pose_from_cyl(r, theta),
                color=color,
                label="distractor block",
            )

        self.button = self._create_rmbench_button(
            r=float(self.BUTTON_R),
            theta=float(self.BUTTON_THETA),
            name="count_confirm_button",
        )
        self.object_layers["BTN"] = "lower"
        self.object_labels["BTN"] = "red count button"
        self.add_prohibit_area(self.button, padding=0.06)
        self._configure_rotate_subtask_plan()

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    def _scan_all_blocks_for_count(self):
        self.begin_rotate_subtask(1)
        self._reset_head_to_home_pose(save_freq=None)
        self._move_head_to_rotate_search_layer("lower")
        scene_objects = list(self.target_blocks.values()) + list(self.distractor_blocks.values())
        scan_thetas = self._get_scan_thetas_from_object_list(
            scene_objects,
            fallback_thetas=self.ROTATE_SCAN_SCENE_FALLBACK_THETAS,
        )
        for theta in scan_thetas:
            scan_point = place_point_cyl(
                [float(self.SCAN_R), float(theta), float(self.SCAN_Z_BIAS) + float(self.table_z_bias)],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="list",
            )
            self._set_rotate_subtask_state(
                subtask_idx=1,
                stage=1,
                focus_object_key=None,
                search_target_keys=list(self.target_blocks.keys()) + list(self.distractor_blocks.keys()),
                action_target_keys=[],
                info_complete=0,
                camera_mode=1,
                camera_target_theta=float(theta),
            )
            self.face_world_point_with_torso(
                scan_point,
                max_iter=35,
                tol_yaw_rad=2e-3,
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            self._refresh_rotate_discovery_from_current_view()
            self.delay(2)

        self.counted_target_count = int(self.target_count)
        self._set_rotate_subtask_state(
            subtask_idx=1,
            stage=2,
            focus_object_key=None,
            search_target_keys=list(self.target_blocks.keys()),
            action_target_keys=[],
            info_complete=1,
            camera_mode=2,
            camera_target_theta=np.nan,
        )
        self.complete_rotate_subtask(1, carried_after=[])

    def _press_count_button(self):
        self._reset_head_to_home_pose(save_freq=None)
        button_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=float(self.SCAN_R),
            scan_z=float(self.SCAN_Z_BIAS) + float(self.table_z_bias),
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        button_cyl = world_to_robot(self.button.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if float(button_cyl[1]) >= 0.0 else "right")
        self.button_arm_tag = arm_tag

        self.enter_rotate_action_stage(2, focus_object_key=(button_key or "BTN"))
        self.face_object_with_torso(self.button, joint_name_prefer=self.SCAN_JOINT_NAME)
        if not self._grasp_button_for_press(
            self.button,
            arm_tag=arm_tag,
            language_annotation="Move to the red count button.",
        ):
            self.plan_success = False
            return
        for press_idx in range(int(self.counted_target_count)):
            if not self._press_button_cycle_after_grasp(
                self.button,
                arm_tag=arm_tag,
                flag_attr="button_press_flag",
                count_attr="button_press_count",
                language_annotation=f"Press the red button for count {press_idx + 1}.",
            ):
                self.plan_success = False
                return
        self.move(self.open_gripper(arm_tag))
        self.move(self.back_to_origin(arm_tag))
        self.complete_rotate_subtask(2, carried_after=[])

    def play_once(self):
        self._scan_all_blocks_for_count()
        if not self.plan_success:
            self.info["info"] = self._build_info()
            return self.info
        self._press_count_button()
        self.info["info"] = self._build_info()
        return self.info

    def _build_info(self):
        return {
            "{A}": "green target blocks",
            "{B}": "red button",
            "{a}": str(getattr(self, "button_arm_tag", None) or self.BUTTON_ARM or "left"),
            "{x}": int(getattr(self, "target_count", 0)),
        }

    def check_success(self):
        self._update_button_reset_flag(self.button, "button_press_flag")
        self._update_button_press_count(self.button, "button_press_flag", "button_press_count")
        self._soft_reset_button_for_success_check(self.button)
        return bool(int(self.button_press_count) == int(self.target_count))
