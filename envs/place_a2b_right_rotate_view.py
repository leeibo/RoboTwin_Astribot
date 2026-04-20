import glob
import os
from .place_a2b_right import place_a2b_right
from .utils import *
import numpy as np
import transforms3d as t3d


class place_a2b_right_rotate_view(place_a2b_right):

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object,
                "B": self.target_object,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_object_A",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "object_A_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_A_right_of_B",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "object_A_right_of_B",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        kwags = init_rotate_theta_bounds(self, kwags)
        super().setup_demo(**kwags)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    def _pose_to_cyl(self, pose):
        world_p = pose.p.tolist() if hasattr(pose, "p") else np.array(pose, dtype=np.float64).reshape(-1)[:3].tolist()
        return world_to_robot(world_p, self.robot_root_xy, self.robot_yaw)

    def _side_place_pose(self, target_pose, arc_dis=0.13, to_left=True):
        target_cyl = self._pose_to_cyl(target_pose)
        r = max(float(target_cyl[0]), 1e-6)
        sign = 1.0 if to_left else -1.0
        theta = float(target_cyl[1]) + sign * float(arc_dis) / r
        return place_point_cyl(
            [r, theta, float(target_cyl[2])],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

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

    @staticmethod
    def _get_available_model_ids(modelname):
        asset_path = os.path.join("assets/objects", modelname)
        json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
        available_ids = []
        for file in json_files:
            base = os.path.basename(file)
            try:
                idx = int(base.replace("model_data", "").replace(".json", ""))
                available_ids.append(idx)
            except ValueError:
                continue
        return available_ids

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        object_list = [
            "047_mouse",
            "050_bell",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            # "086_woodenblock",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]
        object_list_np = np.array(object_list)

        try_num, try_lim = 0, 120
        while try_num <= try_lim:
            rand_pos = rand_pose_cyl(
                rlim=[0.40, 0.47],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            target_rand_pose = rand_pose_cyl(
                rlim=[0.40, 0.47],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            try_num += 1

            distance = float(np.linalg.norm(rand_pos.p[:2] - target_rand_pose.p[:2]))
            obj_theta = float(self._pose_to_cyl(rand_pos)[1])
            tgt_theta = float(self._pose_to_cyl(target_rand_pose)[1])
            theta_gap = float(self._wrap_to_pi(tgt_theta - obj_theta))

            # For right task, source starts at the left side (larger theta), then moves to target-right.
            if distance > 0.17 :
                break

        if try_num > try_lim:
            raise RuntimeError("Actor create limit!")

        self.selected_modelname_A = str(np.random.choice(object_list_np))
        available_model_ids = self._get_available_model_ids(self.selected_modelname_A)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_A}")
        self.selected_model_id_A = int(np.random.choice(available_model_ids))
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname_A,
            convex=True,
            model_id=self.selected_model_id_A,
        )

        self.selected_modelname_B = str(np.random.choice(object_list_np))
        while self.selected_modelname_B == self.selected_modelname_A:
            self.selected_modelname_B = str(np.random.choice(object_list_np))
        available_model_ids = self._get_available_model_ids(self.selected_modelname_B)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_B}")
        self.selected_model_id_B = int(np.random.choice(available_model_ids))
        self.target_object = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname=self.selected_modelname_B,
            convex=True,
            model_id=self.selected_model_id_B,
        )

        self.object.set_mass(0.05)
        self.target_object.set_mass(0.05)
        self.add_prohibit_area(self.object, padding=0.08)
        self.add_prohibit_area(self.target_object, padding=0.12)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        source_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        object_theta = float(self._pose_to_cyl(self.object.get_pose())[1])
        arm_tag = ArmTag("left" if object_theta >= 0.0 else "right")
        self.enter_rotate_action_stage(1, focus_object_key=(source_key or "A"))
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.08, gripper_pos=0.2))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self.complete_rotate_subtask(1, carried_after=["A"])

        target_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self._side_place_pose(self.target_object.get_pose(), arc_dis=0.1, to_left=False)
        self.enter_rotate_action_stage(2, focus_object_key=(target_key or "B"))
        self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose, constrain="free"))
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": f"{self.selected_modelname_A}/base{self.selected_model_id_A}",
            "{B}": f"{self.selected_modelname_B}/base{self.selected_model_id_B}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        object_pose = np.array(self.object.get_pose().p)
        target_pos = np.array(self.target_object.get_pose().p)
        distance = float(np.linalg.norm(object_pose[:2] - target_pos[:2]))

        obj_cyl = world_to_robot(object_pose.tolist(), self.robot_root_xy, self.robot_yaw)
        tgt_cyl = world_to_robot(target_pos.tolist(), self.robot_root_xy, self.robot_yaw)
        theta_diff = float(self._wrap_to_pi(obj_cyl[1] - tgt_cyl[1]))
        radial_diff = abs(float(obj_cyl[0] - tgt_cyl[0]))

        return np.all(
            distance < 0.2
            and distance > 0.08
            and object_pose[2] > 0.7
            and theta_diff < -0.02
            and radial_diff < 0.08
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
