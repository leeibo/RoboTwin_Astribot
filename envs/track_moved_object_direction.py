import numpy as np
import sapien.core as sapien

from ._base_task import Base_Task
from .utils import *


class track_moved_object_direction(Base_Task):
    """Observe a target's motion direction, then search along that direction."""

    ROTATE_TABLE_SHAPE = "fan"

    TARGET_COLOR = (0.10, 0.80, 0.20)
    DISTRACTOR_COLORS = (
        (0.90, 0.20, 0.20),
        (0.20, 0.45, 0.92),
    )
    BLOCK_HALF_SIZE = 0.023

    # The target starts near the center-right of the fan table and moves
    # leftward, out of the currently focused camera view.  The expert policy
    # then turns the head/torso in the same direction before grasping it.
    TARGET_INITIAL_CYL = (0.55, -0.20)
    TARGET_FINAL_CYL = (0.55, 0.40)
    DISTRACTOR_SLOTS = (
        (0.54, -0.55),
        (0.68, 0.18),
    )

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.88
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    TRACK_ARM = "left"
    PICK_PRE_GRASP_DIS = 0.09
    PICK_GRASP_DIS = 0.01
    PICK_LIFT_Z = 0.05
    MOTION_STEPS = 60
    MOTION_SETTLE_STEPS = 4
    SEARCH_STEPS = 3
    PICK_SUCCESS_Z_DELTA = 0.035

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _point_from_cyl(self, cyl, z=None):
        r, theta = cyl
        if z is None:
            z = 0.74 + float(self.BLOCK_HALF_SIZE) + 0.002
        return np.array(
            place_point_cyl(
                [float(r), float(theta), float(z)],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="list",
            ),
            dtype=np.float64,
        ).reshape(3)

    def _pose_from_cyl(self, cyl):
        return sapien.Pose(self._point_from_cyl(cyl).tolist(), [1, 0, 0, 0])

    def _create_block(self, key, pose, color, label):
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
        registry = {"A": self.target}
        registry.update(self.distractors)
        self.configure_rotate_subtask_plan(
            object_registry=registry,
            subtask_defs=[
                {
                    "id": 1,
                    "name": "observe_target_before_motion",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": [],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "target_initially_seen",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "track_motion_direction",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": [],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "motion_direction_known",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "search_along_motion_direction",
                    "instruction_idx": 3,
                    "search_target_keys": ["A"],
                    "action_target_keys": [],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "moved_target_found",
                    "next_subtask_id": 4,
                },
                {
                    "id": 4,
                    "name": "pick_moved_target",
                    "instruction_idx": 4,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "moved_target_grasped",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction=(
                "Watch the green target block move out of view, follow its movement "
                "direction to find it, then pick it up."
            ),
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers = {}
        self.object_labels = {}
        self.initial_pose = self._pose_from_cyl(self.TARGET_INITIAL_CYL)
        self.final_pose = self._pose_from_cyl(self.TARGET_FINAL_CYL)
        self.target = self._create_block(
            key="A",
            pose=self.initial_pose,
            color=self.TARGET_COLOR,
            label="green moving target block",
        )
        self.target_initial_z = float(self.target.get_pose().p[2])
        self.distractors = {}
        for idx, slot in enumerate(self.DISTRACTOR_SLOTS, start=1):
            key = f"D{idx}"
            self.distractors[key] = self._create_block(
                key=key,
                pose=self._pose_from_cyl(slot),
                color=self.DISTRACTOR_COLORS[(idx - 1) % len(self.DISTRACTOR_COLORS)],
                label="stationary distractor block",
            )
        self.motion_start = np.array(self.initial_pose.p, dtype=np.float64).reshape(3)
        self.motion_end = np.array(self.final_pose.p, dtype=np.float64).reshape(3)
        self.motion_vector = self.motion_end - self.motion_start
        self.motion_direction_label = "leftward"
        self._configure_rotate_subtask_plan()

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    def _target_theta(self, point):
        local = world_to_robot(
            np.array(point, dtype=np.float64).reshape(3).tolist(),
            self.robot_root_xy,
            self.robot_yaw,
        )
        return float(local[1])

    def _focus_world_point(self, point, subtask_idx, stage, info_complete=0):
        point = np.array(point, dtype=np.float64).reshape(3)
        self._set_rotate_subtask_state(
            subtask_idx=subtask_idx,
            stage=stage,
            focus_object_key="A",
            search_target_keys=["A"],
            action_target_keys=[],
            info_complete=info_complete,
            camera_mode=1,
            camera_target_theta=self._target_theta(point),
        )
        scan_point = point.copy()
        scan_point[2] = float(self.SCAN_Z_BIAS) + float(self.table_z_bias)
        self.face_world_point_with_torso(
            scan_point,
            max_iter=35,
            tol_yaw_rad=2e-3,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        self._refresh_rotate_discovery_from_current_view()

    def _observe_initial_target(self):
        self.begin_rotate_subtask(1)
        self._reset_head_to_home_pose(save_freq=None)
        self._move_head_to_rotate_search_layer("lower")
        self._focus_world_point(self.motion_start, subtask_idx=1, stage=1, info_complete=0)
        self.delay(5)
        self.complete_rotate_subtask(1, carried_after=[])

    def _animate_target_motion(self):
        self.begin_rotate_subtask(2)
        self._focus_world_point(self.motion_start, subtask_idx=2, stage=1, info_complete=0)
        quat = list(self.initial_pose.q)
        steps = max(int(self.MOTION_STEPS), 1)
        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            pos = (1.0 - alpha) * self.motion_start + alpha * self.motion_end
            self.target.actor.set_pose(sapien.Pose(pos.tolist(), quat))
            self._set_rotate_subtask_state(
                subtask_idx=2,
                stage=1,
                focus_object_key="A",
                search_target_keys=["A"],
                action_target_keys=[],
                info_complete=0,
                camera_mode=1,
                camera_target_theta=self._target_theta(pos),
            )
            # Save every scripted motion step during replay so the demo shows a
            # continuous track rather than a before/after teleport.
            self.delay(1, save_freq=1)
        self.delay(int(self.MOTION_SETTLE_STEPS), save_freq=1)
        self._set_rotate_subtask_state(
            subtask_idx=2,
            stage=2,
            focus_object_key="A",
            search_target_keys=["A"],
            action_target_keys=[],
            info_complete=1,
            camera_mode=2,
            camera_target_theta=self._target_theta(self.motion_end),
        )
        self.complete_rotate_subtask(2, carried_after=[])

    def _search_along_motion_direction(self):
        self.begin_rotate_subtask(3)
        for alpha in np.linspace(0.45, 1.0, int(self.SEARCH_STEPS)):
            point = self.motion_start + float(alpha) * self.motion_vector
            self._focus_world_point(point, subtask_idx=3, stage=1, info_complete=0)
            self.delay(3)
        self._set_rotate_subtask_state(
            subtask_idx=3,
            stage=2,
            focus_object_key="A",
            search_target_keys=["A"],
            action_target_keys=[],
            info_complete=1,
            camera_mode=2,
            camera_target_theta=self._target_theta(self.motion_end),
        )
        self.complete_rotate_subtask(3, carried_after=[])

    def _pick_moved_target(self):
        self.enter_rotate_action_stage(4, focus_object_key="A")
        self.face_object_with_torso(self.target, joint_name_prefer=self.SCAN_JOINT_NAME)
        if not self.move(
            self.grasp_actor(
                self.target,
                arm_tag=self.TRACK_ARM,
                pre_grasp_dis=float(self.PICK_PRE_GRASP_DIS),
                grasp_dis=float(self.PICK_GRASP_DIS),
            )
        ):
            self.plan_success = False
            return
        self._set_carried_object_keys(["A"])
        if not self.move(self.move_by_displacement(arm_tag=self.TRACK_ARM, z=float(self.PICK_LIFT_Z))):
            self.plan_success = False
            return
        self.complete_rotate_subtask(4, carried_after=["A"])

    def play_once(self):
        self._observe_initial_target()
        if self.plan_success:
            self._animate_target_motion()
        if self.plan_success:
            self._search_along_motion_direction()
        if self.plan_success:
            self._pick_moved_target()
        self.info["info"] = self._build_info()
        return self.info

    def _build_info(self):
        return {
            "{A}": "green moving target block",
            "{B}": "stationary distractor blocks",
            "{a}": str(self.TRACK_ARM),
            "{d}": self.motion_direction_label,
        }

    def check_success(self):
        target_z = float(self.target.get_pose().p[2])
        lifted = target_z - float(getattr(self, "target_initial_z", target_z)) > float(self.PICK_SUCCESS_Z_DELTA)
        return bool(lifted and not self.is_left_gripper_open())
