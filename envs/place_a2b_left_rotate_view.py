import glob
import os
from .place_a2b_left import place_a2b_left
from .utils import *
import numpy as np
import transforms3d as t3d


class place_a2b_left_rotate_view(place_a2b_left):

    def setup_demo(self, **kwags):
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        super().setup_demo(**kwags)

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
            "048_stapler",
            "050_bell",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            "077_phone",
            "081_playingcards",
            "086_woodenblock",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]

        try_num, try_lim = 0, 100
        while try_num <= try_lim:
            rand_pos = rand_pose_cyl(
                rlim=[0.44, 0.5],
                thetalim=[-1.0, 1.0],
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            if rand_pos.p[0] > 0:
                tx_range = [0.18, 0.23]
            else:
                tx_range = [-0.1, 0.1]
            target_rand_pose = rand_pose_cyl(
                rlim=[0.44, 0.5],
                thetalim=[-1.0, 1.0],
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            refine = 0
            while (
                np.linalg.norm(target_rand_pose.p[:2] - rand_pos.p[:2]) < 0.1
                or abs(target_rand_pose.p[1] - rand_pos.p[1]) < 0.1
                or target_rand_pose.p[0] < tx_range[0]
                or target_rand_pose.p[0] > tx_range[1]
            ):
                refine += 1
                target_rand_pose = rand_pose_cyl(
                    rlim=[0.44, 0.5],
                    thetalim=[-1.0, 1.0],
                    zlim=[0.741, 0.741],
                    robot_root_xy=self.robot_root_xy,
                    robot_yaw_rad=self.robot_yaw,
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=True,
                    rotate_lim=[0, 3.14, 0],
                )
                if refine > 80:
                    break
            try_num += 1

            distance = np.linalg.norm(rand_pos.p[:2] - target_rand_pose.p[:2])
            if distance > 0.19 or rand_pos.p[0] > target_rand_pose.p[0]:
                break
        if try_num > try_lim:
            raise RuntimeError("Actor create limit!")

        self.selected_modelname_A = str(np.random.choice(object_list))
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

        self.selected_modelname_B = str(np.random.choice(object_list))
        while self.selected_modelname_B == self.selected_modelname_A:
            self.selected_modelname_B = str(np.random.choice(object_list))
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
        self.object.set_mass(0.2)
        self.target_object.set_mass(0.2)
        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.target_object, padding=0.1)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")
        self.face_object_with_torso(self.object, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        target_pose = self.target_object.get_pose().p.tolist()
        target_pose[0] -= 0.13
        self.face_world_point_with_torso(target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose, constrain="free"))

        self.info["info"] = {
            "{A}": f"{self.selected_modelname_A}/base{self.selected_model_id_A}",
            "{B}": f"{self.selected_modelname_B}/base{self.selected_model_id_B}",
            "{a}": str(arm_tag),
        }
        return self.info
