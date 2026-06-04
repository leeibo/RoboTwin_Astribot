import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *


class inspect_underside_sort_block(Base_Task):
    """Manipulate a block to inspect its hidden underside inset, then sort it."""

    ROTATE_TABLE_SHAPE = "fan"

    OUTER_COLOR = (0.72, 0.72, 0.72)
    UNDERSIDE_COLOR_OPTIONS = (
        ("red", (0.90, 0.20, 0.20)),
        ("blue", (0.20, 0.45, 0.92)),
        ("yellow", (0.92, 0.74, 0.18)),
    )
    BLOCK_HALF_SIZE = 0.024
    # Make the hidden color patch large enough to be readable in the demo once
    # the block is lifted and rolled toward the camera.  It is still fully on
    # the underside, so it cannot be observed before manipulation.
    INSET_HALF_SIZE = (0.024, 0.024, 0.003)
    BLOCK_CYL = (0.52, 0.02)

    PAD_SPECS = {
        "red": {"cyl": (0.50, 0.45), "color": (0.90, 0.20, 0.20)},
        "blue": {"cyl": (0.62, 0.25), "color": (0.20, 0.45, 0.92)},
        "yellow": {"cyl": (0.58, -0.45), "color": (0.92, 0.74, 0.18)},
    }
    PAD_HALF_SIZE = (0.075, 0.055, 0.004)
    PAD_ANCHOR_HALF_SIZE = (0.006, 0.006, 0.004)

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.88
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    ARM = "left"
    PICK_PRE_GRASP_DIS = 0.08
    PICK_GRASP_DIS = 0.01
    LIFT_Z = 0.06
    # Keep the presentation pose in the verified reachable workspace; the
    # enlarged underside patch below is what makes the flip readable in the
    # demo without making the scripted lift/sort fail.
    INSPECT_CYL = (0.42, 0.0)
    INSPECT_Z = 0.88
    INSPECT_PRESENTATION_WORLD_OFFSET = (0.0, 0.075, 0.0)
    INSPECT_FLIP_STEPS = 14
    INSPECT_HOLD_STEPS = 8
    PLACE_RELEASE_Z = 0.80
    PLACE_RETREAT_Z = 0.08
    SUCCESS_XY_TOL = 0.10

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        override = kwargs.get("inspect_underside_color_override", None)
        self.inspect_underside_color_override = None if override is None else str(override).lower()
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

    def _pose_from_cyl(self, cyl, z=None):
        return sapien.Pose(self._point_from_cyl(cyl, z=z).tolist(), [1, 0, 0, 0])

    def _create_underside_inset_actor(self, pose, inset_color):
        scene, pose = preprocess(self, pose)
        entity = sapien.Entity()
        entity.set_name("underside_inset_block")
        entity.set_pose(pose)

        rigid_component = sapien.physx.PhysxRigidDynamicComponent()
        rigid_component.attach(
            sapien.physx.PhysxCollisionShapeBox(
                half_size=(self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE),
                material=scene.default_physical_material,
            )
        )

        render_component = sapien.render.RenderBodyComponent()
        outer_material = sapien.render.RenderMaterial(base_color=[*self.OUTER_COLOR[:3], 1])
        render_component.attach(
            sapien.render.RenderShapeBox(
                np.array([self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE], dtype=np.float32),
                outer_material,
            )
        )
        inset_material = sapien.render.RenderMaterial(base_color=[*tuple(float(v) for v in inset_color[:3]), 1])
        inset_shape = sapien.render.RenderShapeBox(np.array(self.INSET_HALF_SIZE, dtype=np.float32), inset_material)
        # Slightly protrude below the collision box to avoid z-fighting while
        # still being hidden by the table before the block is lifted/flipped.
        local_z = -float(self.BLOCK_HALF_SIZE) - 0.001
        inset_shape.set_local_pose(sapien.Pose([0.0, 0.0, local_z], [1, 0, 0, 0]))
        render_component.attach(inset_shape)

        entity.add_component(rigid_component)
        entity.add_component(render_component)
        entity.set_pose(pose)
        scene.add_entity(entity)

        half_size = [float(self.BLOCK_HALF_SIZE)] * 3
        data = {
            "center": [0, 0, 0],
            "extents": half_size,
            "scale": half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
        return Actor(entity, data)

    def _create_underside_block(self):
        choices = list(self.UNDERSIDE_COLOR_OPTIONS)
        if self.inspect_underside_color_override is None:
            label, color = choices[int(np.random.randint(len(choices)))]
        else:
            by_label = {label: color for label, color in choices}
            if self.inspect_underside_color_override not in by_label:
                raise ValueError(f"Unknown underside color override: {self.inspect_underside_color_override}")
            label = self.inspect_underside_color_override
            color = by_label[label]
        block = self._create_underside_inset_actor(
            pose=self._pose_from_cyl(self.BLOCK_CYL),
            inset_color=color,
        )
        block.set_mass(0.035)
        self.underside_color_label = str(label)
        self.underside_color = tuple(float(v) for v in color)
        self.object_layers["A"] = "lower"
        self.object_labels["A"] = "gray block with hidden underside inset"
        self.add_prohibit_area(block, padding=0.05)
        return block

    def _create_sort_regions(self):
        self.sort_pads = {}
        self.pad_centers = {}
        for label, spec in self.PAD_SPECS.items():
            center = self._point_from_cyl(spec["cyl"], z=0.742)
            self.pad_centers[label] = center
            create_visual_box(
                scene=self,
                pose=sapien.Pose(center.tolist(), [1, 0, 0, 0]),
                half_size=self.PAD_HALF_SIZE,
                color=tuple(float(v) for v in spec["color"]),
                name=f"{label}_sort_region_visual",
            )
            anchor = create_box(
                scene=self,
                pose=sapien.Pose([float(center[0]), float(center[1]), 0.747], [1, 0, 0, 0]),
                half_size=self.PAD_ANCHOR_HALF_SIZE,
                color=tuple(float(v) for v in spec["color"]),
                is_static=True,
                name=f"{label}_sort_region_anchor",
            )
            key = label.upper()
            self.sort_pads[label] = anchor
            self.object_layers[key] = "lower"
            self.object_labels[key] = f"{label} sorting region"

    def _configure_rotate_subtask_plan(self):
        registry = {"A": self.block}
        for label, pad in self.sort_pads.items():
            registry[label.upper()] = pad
        self.configure_rotate_subtask_plan(
            object_registry=registry,
            subtask_defs=[
                {
                    "id": 1,
                    "name": "observe_outer_block_color",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": [],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "outer_color_seen_but_underside_hidden",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "lift_and_flip_for_underside_inspection",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "underside_exposed",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "inspect_underside_color",
                    "instruction_idx": 3,
                    "search_target_keys": ["A"],
                    "action_target_keys": [],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "underside_color_identified",
                    "next_subtask_id": 4,
                },
                {
                    "id": 4,
                    "name": "sort_by_underside_color",
                    "instruction_idx": 4,
                    "search_target_keys": ["RED", "BLUE", "YELLOW"],
                    "action_target_keys": ["A", "RED", "BLUE", "YELLOW"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "block_sorted_by_underside_color",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction=(
                "Pick up the gray block, inspect the hidden colored inset on its underside, "
                "then place it on the sorting region with the same color."
            ),
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers = {}
        self.object_labels = {}
        self.inspected_underside_color_label = None
        self.block = self._create_underside_block()
        self.initial_block_z = float(self.block.get_pose().p[2])
        self._create_sort_regions()
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

    def _focus_world_point(self, point, subtask_idx, stage, focus_object_key="A", info_complete=0):
        point = np.array(point, dtype=np.float64).reshape(3)
        self._set_rotate_subtask_state(
            subtask_idx=subtask_idx,
            stage=stage,
            focus_object_key=focus_object_key,
            search_target_keys=[focus_object_key] if focus_object_key else [],
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

    def _observe_outer_block(self):
        self.begin_rotate_subtask(1)
        self._reset_head_to_home_pose(save_freq=None)
        self._move_head_to_rotate_search_layer("lower")
        self._focus_world_point(self.block.get_pose().p, subtask_idx=1, stage=1, info_complete=0)
        self.delay(4)
        self.complete_rotate_subtask(1, carried_after=[])

    def _pick_and_lift_block(self):
        self.enter_rotate_action_stage(2, focus_object_key="A")
        self.face_object_with_torso(self.block, joint_name_prefer=self.SCAN_JOINT_NAME)
        if not self.move(
            self.grasp_actor(
                self.block,
                arm_tag=self.ARM,
                pre_grasp_dis=float(self.PICK_PRE_GRASP_DIS),
                grasp_dis=float(self.PICK_GRASP_DIS),
            )
        ):
            self.plan_success = False
            return False
        self._set_carried_object_keys(["A"])
        if not self.move(self.move_by_displacement(arm_tag=self.ARM, z=float(self.LIFT_Z))):
            self.plan_success = False
            return False
        self.complete_rotate_subtask(2, carried_after=["A"])
        return True

    def _set_block_orientation(self, roll_rad, save_freq=None, position=None):
        if position is None:
            pos = np.array(self.block.get_pose().p, dtype=np.float64).reshape(3)
        else:
            pos = np.array(position, dtype=np.float64).reshape(3)
        quat = t3d.euler.euler2quat(float(roll_rad), 0.0, 0.0)
        self.block.actor.set_pose(sapien.Pose(pos.tolist(), quat))
        self.delay(1, save_freq=save_freq)

    def _inspect_underside(self):
        self.begin_rotate_subtask(3)
        inspect_point = self._point_from_cyl(self.INSPECT_CYL, z=float(self.INSPECT_Z))
        self._focus_world_point(inspect_point, subtask_idx=3, stage=1, info_complete=0)
        # Move the lifted block to a centered, elevated "show to camera" pose.
        # Then roll the held block so the bottom inset faces the external
        # observer camera (+Y direction).  This scripted manipulation is the
        # action-acquired information stage.
        if not self._move_carried_block_center_to(inspect_point):
            self.plan_success = False
            return
        presentation_point = inspect_point + np.array(
            self.INSPECT_PRESENTATION_WORLD_OFFSET,
            dtype=np.float64,
        ).reshape(3)
        flip_rolls = list(np.linspace(0.0, np.pi / 2.0, int(self.INSPECT_FLIP_STEPS)))
        for step_idx, roll in enumerate(flip_rolls):
            alpha = 0.0 if len(flip_rolls) <= 1 else float(step_idx) / float(len(flip_rolls) - 1)
            show_point = (1.0 - alpha) * inspect_point + alpha * presentation_point
            self._set_rotate_subtask_state(
                subtask_idx=3,
                stage=1,
                focus_object_key="A",
                search_target_keys=["A"],
                action_target_keys=[],
                info_complete=0,
                camera_mode=1,
                camera_target_theta=self._target_theta(self.block.get_pose().p),
            )
            self._set_block_orientation(float(roll), save_freq=1, position=show_point)
        self.inspected_underside_color_label = str(self.underside_color_label)
        self._set_rotate_subtask_state(
            subtask_idx=3,
            stage=2,
            focus_object_key="A",
            search_target_keys=["A"],
            action_target_keys=[],
            info_complete=1,
            camera_mode=2,
            camera_target_theta=self._target_theta(self.block.get_pose().p),
        )
        self.delay(int(self.INSPECT_HOLD_STEPS), save_freq=1)
        # Return the block upright before transport to keep placement robust.
        return_rolls = list(np.linspace(np.pi / 2.0, 0.0, int(self.INSPECT_FLIP_STEPS)))
        for step_idx, roll in enumerate(return_rolls):
            alpha = 0.0 if len(return_rolls) <= 1 else float(step_idx) / float(len(return_rolls) - 1)
            show_point = (1.0 - alpha) * presentation_point + alpha * inspect_point
            self._set_block_orientation(float(roll), save_freq=1, position=show_point)
        self._set_carried_object_keys(["A"])
        self.complete_rotate_subtask(3, carried_after=["A"])

    def _move_carried_block_center_to(self, target_center):
        target_center = np.array(target_center, dtype=np.float64).reshape(3)
        ee_pose = np.array(self.robot.get_left_ee_pose(), dtype=np.float64).reshape(7)
        block_center = np.array(self.block.get_pose().p, dtype=np.float64).reshape(3)
        target_ee_pose = ee_pose.copy()
        target_ee_pose[:3] += target_center - block_center
        return bool(self.move(self.move_to_pose(arm_tag=self.ARM, target_pose=target_ee_pose.tolist())))

    def _sort_by_underside_color(self):
        target_label = str(self.inspected_underside_color_label)
        target_key = target_label.upper()
        self.enter_rotate_action_stage(4, focus_object_key=target_key)

        if self.is_left_gripper_open():
            self.face_object_with_torso(self.block, joint_name_prefer=self.SCAN_JOINT_NAME)
            if not self.move(
                self.grasp_actor(
                    self.block,
                    arm_tag=self.ARM,
                    pre_grasp_dis=float(self.PICK_PRE_GRASP_DIS),
                    grasp_dis=float(self.PICK_GRASP_DIS),
                )
            ):
                self.plan_success = False
                return False
            self._set_carried_object_keys(["A"])
            if not self.move(self.move_by_displacement(arm_tag=self.ARM, z=0.04)):
                self.plan_success = False
                return False

        target_center = np.array(self.pad_centers[target_label], dtype=np.float64).reshape(3)
        target_center[2] = float(self.PLACE_RELEASE_Z)
        self.face_world_point_with_torso(
            target_center,
            max_iter=35,
            tol_yaw_rad=2e-3,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if not self._move_carried_block_center_to(target_center):
            self.plan_success = False
            return False
        if not self.move(self.open_gripper(self.ARM)):
            self.plan_success = False
            return False
        self._set_carried_object_keys([])
        self.delay(8)
        self.complete_rotate_subtask(4, carried_after=[])
        return True

    def play_once(self):
        self._observe_outer_block()
        if self.plan_success and self._pick_and_lift_block():
            self._inspect_underside()
        if self.plan_success:
            self._sort_by_underside_color()
        self.info["info"] = self._build_info()
        return self.info

    def _build_info(self):
        return {
            "{A}": "gray block with hidden underside inset",
            "{B}": f"{self.underside_color_label} sorting region",
            "{a}": str(self.ARM),
            "{c}": str(self.underside_color_label),
        }

    def check_success(self):
        label = getattr(self, "inspected_underside_color_label", None)
        if label is None or label not in self.pad_centers:
            return False
        block_xy = np.array(self.block.get_pose().p[:2], dtype=np.float64)
        target_xy = np.array(self.pad_centers[label][:2], dtype=np.float64)
        sorted_ok = float(np.linalg.norm(block_xy - target_xy)) < float(self.SUCCESS_XY_TOL)
        color_ok = str(label) == str(getattr(self, "underside_color_label", ""))
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return bool(sorted_ok and color_ok and gripper_open)
