from .place_bread_skillet import place_bread_skillet
from .utils import *
import numpy as np
import transforms3d as t3d


class place_bread_skillet_rotate_view(place_bread_skillet):

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

        id_list = [0, 1, 3, 5, 6]
        self.bread_id = int(np.random.choice(id_list))
        bread_side = 1.0 if np.random.rand() < 0.5 else -1.0
        bread_pose = rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=rotate_theta_side(self, side=bread_side),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )
        self.bread = create_actor(
            self,
            pose=bread_pose,
            modelname="075_bread",
            model_id=self.bread_id,
            convex=True,
        )

        self.model_id_list = [0, 1, 2, 3]
        self.skillet_id = int(np.random.choice(self.model_id_list))
        skillet_pose = rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=rotate_theta_side(self, side=-bread_side),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0, 0, 0.707, 0.707],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 6, 0],
        )
        self.skillet = create_actor(
            self,
            pose=skillet_pose,
            modelname="106_skillet",
            model_id=self.skillet_id,
            convex=True,
        )

        self.bread.set_mass(0.001)
        self.skillet.set_mass(0.01)
        self.add_prohibit_area(self.bread, padding=0.03)
        self.add_prohibit_area(self.skillet, padding=0.05)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.skillet.get_pose().p[0] > 0 else "left")
        self.face_object_with_torso(self.skillet, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.skillet, arm_tag=arm_tag, pre_grasp_dis=0.07, gripper_pos=0))
        self.face_object_with_torso(self.bread, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.bread, arm_tag=arm_tag.opposite, pre_grasp_dis=0.07, gripper_pos=0))

        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"),
            self.move_by_displacement(arm_tag=arm_tag.opposite, z=0.1),
        )

        target_pose = self.get_arm_pose(arm_tag=arm_tag)
        if arm_tag == "left":
            target_pose[:2] = [-0.1, -0.05]
            target_pose[2] -= 0.05
            target_pose[3:] = [-0.707, 0, -0.707, 0]
        else:
            target_pose[:2] = [0.1, -0.05]
            target_pose[2] -= 0.05
            target_pose[3:] = [0, 0.707, 0, -0.707]

        self.face_world_point_with_torso(target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose))

        bread_target_pose = self.skillet.get_functional_point(0)
        self.face_world_point_with_torso(bread_target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.bread,
                target_pose=bread_target_pose,
                arm_tag=arm_tag.opposite,
                constrain="free",
                pre_dis=0.05,
                dis=0.05,
            )
        )

        self.info["info"] = {
            "{A}": f"106_skillet/base{self.skillet_id}",
            "{B}": f"075_bread/base{self.bread_id}",
            "{a}": str(arm_tag),
        }
        return self.info
