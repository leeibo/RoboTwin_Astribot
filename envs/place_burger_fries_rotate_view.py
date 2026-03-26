from .place_burger_fries import place_burger_fries
from .utils import *
import numpy as np
import transforms3d as t3d


class place_burger_fries_rotate_view(place_burger_fries):

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

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        tray_pose = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=[-0.05, 0.05],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.706527, 0.706483, -0.0291356, -0.0291767],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
            quat_frame="cyl",
        )
        self.tray_id = int(np.random.choice([0, 1, 2, 3], 1)[0])
        self.tray = create_actor(
            scene=self,
            pose=tray_pose,
            modelname="008_tray",
            convex=True,
            model_id=self.tray_id,
            scale=(2.0, 2.0, 2.0),
            is_static=True,
        )
        self.tray.set_mass(0.05)

        hamburg_pose = rand_pose_cyl(
            rlim=[0.35, 0.45],
            thetalim=[0.75, 1.1],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.object1_id = int(np.random.choice([0, 1, 2, 3, 4, 5], 1)[0])
        self.hamburg = create_actor(
            scene=self,
            pose=hamburg_pose,
            modelname="006_hamburg",
            convex=True,
            model_id=self.object1_id,
        )
        self.hamburg.set_mass(0.05)

        fries_pose = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=[-1.1, -0.75],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1.0, 0.0, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.object2_id = int(np.random.choice([0, 1], 1)[0])
        self.frenchfries = create_actor(
            scene=self,
            pose=fries_pose,
            modelname="005_french-fries",
            convex=True,
            model_id=self.object2_id,
        )
        self.frenchfries.set_mass(0.05)

        self.add_prohibit_area(self.tray, padding=0.1)
        self.add_prohibit_area(self.hamburg, padding=0.05)
        self.add_prohibit_area(self.frenchfries, padding=0.05)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())
        tray_place_pose_left = self.tray.get_functional_point(0)
        tray_place_pose_right = self.tray.get_functional_point(1)

        arm_tag_left = ArmTag("left")
        arm_tag_right = ArmTag("right")

        self.face_object_with_torso(self.hamburg, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.hamburg, arm_tag=arm_tag_left, pre_grasp_dis=0.1,gripper_pos=0.01))
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.1))
        self.face_world_point_with_torso(tray_place_pose_left[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.hamburg,
                arm_tag=arm_tag_left,
                target_pose=tray_place_pose_left,
                functional_point_id=0,
                constrain="free",
                pre_dis=0.1,
                pre_dis_axis="fp",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.08))
        self.move(self.back_to_origin(arm_tag=arm_tag_left))

        self.face_object_with_torso(self.frenchfries, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.frenchfries, arm_tag=arm_tag_right, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.1))        
        self.face_world_point_with_torso(tray_place_pose_right[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.frenchfries,
                arm_tag=arm_tag_right,
                target_pose=tray_place_pose_right,
                functional_point_id=0,
                constrain="free",
                pre_dis=0.1,
                pre_dis_axis="fp",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.08))

        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.object1_id}",
            "{B}": f"008_tray/base{self.tray_id}",
            "{C}": f"005_french-fries/base{self.object2_id}",
        }
        return self.info
