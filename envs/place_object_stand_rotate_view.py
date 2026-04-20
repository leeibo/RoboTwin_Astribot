from .place_object_stand import place_object_stand
from .utils import *
import glob
import os
import numpy as np
import transforms3d as t3d


class place_object_stand_rotate_view(place_object_stand):

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object,
                "B": self.displaystand,
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
                    "name": "place_object_on_stand",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "object_on_stand",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, is_test=False, **kwags):
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        kwags = init_rotate_theta_bounds(self, kwags)
        super().setup_demo(is_test=is_test, **kwags)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    def _scan_scene_two_views(self, object_list=None):
        scan_r = 0.62
        scan_z = 0.88 + self.table_z_bias
        for theta in self._get_scan_thetas_from_object_list(object_list, fallback_thetas=[0.95, -0.95]):
            scan_point = place_point_cyl(
                [scan_r, theta, scan_z],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="list",
            )
            self.face_world_point_with_torso(
                scan_point,
                max_iter=35,
                tol_yaw_rad=2e-3,
                joint_name_prefer="astribot_torso_joint_2",
            )

    def _get_available_model_ids(self, modelname):
        asset_path = os.path.join("assets/objects", modelname)
        json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
        available_ids = []
        for file in json_files:
            base = os.path.basename(file)
            try:
                available_ids.append(int(base.replace("model_data", "").replace(".json", "")))
            except ValueError:
                continue
        return available_ids

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        object_list = [
            # "047_mouse",
            # "048_stapler",
            "050_bell",
            "073_rubikscube",
            "057_toycar",
            # "079_remotecontrol",
        ]
        self.selected_modelname = str(np.random.choice(object_list))
        available_model_ids = self._get_available_model_ids(self.selected_modelname)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname}")
        self.selected_model_id = int(np.random.choice(available_model_ids))

        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_obj_lim = rotate_theta_side(self, side=side)
        theta_stand_lim = rotate_theta_side(self, side=-side)

        while True:
            object_pose = rand_pose_cyl(
                rlim=[0.4, 0.45],
                thetalim=theta_obj_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 3, 0],
            )
            object_cyl = world_to_robot(object_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(object_cyl[1]) < 0.35:
                continue
            break

        while True:
            displaystand_pose = rand_pose_cyl(
                rlim=[0.4, 0.45],
                thetalim=theta_stand_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 6, 0],
            )
            if np.linalg.norm(displaystand_pose.p[:2] - object_pose.p[:2]) < 0.12:
                continue
            stand_cyl = world_to_robot(displaystand_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(stand_cyl[1]) < 0.25:
                continue
            break

        self.object = create_actor(
            scene=self,
            pose=object_pose,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object.set_mass(0.1)

        self.displaystand_id = int(np.random.choice([0, 1, 2, 3, 4]))
        self.displaystand = create_actor(
            scene=self,
            pose=displaystand_pose,
            modelname="074_displaystand",
            convex=True,
            model_id=self.displaystand_id,
            is_static=True,
        )
        self.displaystand.set_mass(0.01)

        self.add_prohibit_area(self.displaystand, padding=0.05)
        self.add_prohibit_area(self.object, padding=0.1)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        object_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

        self.enter_rotate_action_stage(1, focus_object_key=(object_key or "A"))
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1,gripper_pos=0.2))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        self.complete_rotate_subtask(1, carried_after=["A"])

        stand_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        displaystand_pose = self.displaystand.get_functional_point(0)
        self.enter_rotate_action_stage(2, focus_object_key=(stand_key or "B"))
        self.move(
            self.place_actor(
                self.object,
                arm_tag=arm_tag,
                target_pose=displaystand_pose,
                constrain="free",
                pre_dis=0.07,
            )
        )
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": f"074_displaystand/base{self.displaystand_id}",
            "{a}": str(arm_tag),
        }
        return self.info
