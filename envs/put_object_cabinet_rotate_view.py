from .put_object_cabinet import put_object_cabinet
from .utils import *
import glob
import os
import numpy as np
import transforms3d as t3d


class put_object_cabinet_rotate_view(put_object_cabinet):

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
        scan_z = 0.9 + self.table_z_bias
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

        self.model_name = "036_cabinet"
        self.model_id = 46653
        cabinet_pose = rand_pose_cyl(
            rlim=[0.46, 0.5],
            thetalim=[-0.12, 0.12],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.7071068, 0, 0, 0.7071068],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 16],
        )
        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            pose=cabinet_pose,
            modelname=self.model_name,
            modelid=self.model_id,
            fix_root_link=True,
        )

        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_obj_lim = [0.72, 1.1] if side > 0 else [-1.1, -0.72]
        while True:
            rand_pos = rand_pose_cyl(
                rlim=[0.5, 0.5],
                thetalim=theta_obj_lim,
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 3, 0],
            )
            obj_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(obj_cyl[1]) < 0.35:
                continue
            if np.linalg.norm(rand_pos.p[:2] - self.cabinet.get_pose().p[:2]) < 0.2:
                continue
            break

        object_list = [
            "047_mouse",
            "048_stapler",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            "077_phone",
            "081_playingcards",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]
        self.selected_modelname = str(np.random.choice(object_list))
        available_model_ids = self._get_available_model_ids(self.selected_modelname)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname}")
        self.selected_model_id = int(np.random.choice(available_model_ids))
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object.set_mass(0.01)

        self.add_prohibit_area(self.object, padding=0.01)
        self.add_prohibit_area(self.cabinet, padding=0.01)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")
        self.arm_tag = arm_tag
        self.origin_z = self.object.get_pose().p[2]

        self.face_object_with_torso(self.object, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))

        self.face_object_with_torso(self.cabinet, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.cabinet, arm_tag=arm_tag.opposite, pre_grasp_dis=0.05))

        for _ in range(4):
            self.move(self.move_by_displacement(arm_tag=arm_tag.opposite, y=-0.04))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))

        target_pose = self.cabinet.get_functional_point(0)
        self.face_world_point_with_torso(target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.object,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.13,
                dis=0.1,
                constrain="free",
            )
        )

        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": "036_cabinet/base0",
            "{a}": str(arm_tag),
            "{b}": str(arm_tag.opposite),
        }
        return self.info
