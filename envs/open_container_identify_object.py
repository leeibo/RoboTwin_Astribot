import numpy as np

from .search_object import search_object
from .utils import *


class open_container_identify_object(search_object):
    """Open a closed container and identify which object is inside."""

    OBJECT_VARIANTS = (
        {
            "kind": "block",
            "label": "red block",
            "color": (0.90, 0.20, 0.20),
            "outward_offset": 0.015,
            "surface_z_offset": search_object.OBJECT_Z_BIAS,
            "mass": search_object.OBJECT_MASS,
        },
        {
            "kind": "block",
            "label": "blue block",
            "color": (0.20, 0.45, 0.92),
            "outward_offset": 0.015,
            "surface_z_offset": search_object.OBJECT_Z_BIAS,
            "mass": search_object.OBJECT_MASS,
        },
        {
            "kind": "block",
            "label": "yellow block",
            "color": (0.92, 0.74, 0.18),
            "outward_offset": 0.015,
            "surface_z_offset": search_object.OBJECT_Z_BIAS,
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
                    "action_target_keys": [],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "inside_object_identified",
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
            "{b}": str(self.cabinet_arm_tag),
        }

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
        self.complete_rotate_subtask(3, carried_after=[])

        self.info["info"] = self._build_info()
        return self.info

    def check_success(self):
        return bool(
            getattr(self, "cabinet_opened", False)
            and getattr(self, "identified_object_label", None) == str(getattr(self, "object_label", ""))
        )
