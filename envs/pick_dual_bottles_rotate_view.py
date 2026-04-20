from .pick_dual_bottles import pick_dual_bottles
from .utils import *
import numpy as np
import transforms3d as t3d


class pick_dual_bottles_rotate_view(pick_dual_bottles):

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

        bottle1_pose = rand_pose_cyl(
            rlim=[0.48, 0.5],
            thetalim=rotate_theta_side(self, side=1),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.66, 0.66, -0.25, -0.25],
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
        )
        self.bottle1 = create_actor(
            self,
            pose=bottle1_pose,
            modelname="001_bottle",
            convex=True,
            model_id=13,
        )

        while True:
            bottle2_pose = rand_pose_cyl(
                rlim=[0.48, 0.5],
                thetalim=rotate_theta_side(self, side=-1),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.65, 0.65, 0.27, 0.27],
                rotate_rand=True,
                rotate_lim=[0, 1, 0],
            )
            if np.linalg.norm(bottle2_pose.p[:2] - bottle1_pose.p[:2]) < 0.15:
                continue
            break
        self.bottle2 = create_actor(
            self,
            pose=bottle2_pose,
            modelname="001_bottle",
            convex=True,
            model_id=16,
        )

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.add_prohibit_area(self.bottle1, padding=0.1)
        self.add_prohibit_area(self.bottle2, padding=0.1)
        self.prohibited_area.append([-0.2, -0.2, 0.2, -0.02])
        self.left_target_pose = place_pose_cyl(
            [0.54, 0.20, 1.0, 0, 1, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self.right_target_pose = place_pose_cyl(
            [0.54, -0.20, 1.0, 0, 1, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        bottle1_arm_tag = ArmTag("left")
        bottle2_arm_tag = ArmTag("right")

        # self.face_object_with_torso(self.bottle1, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.bottle1, arm_tag=bottle1_arm_tag, pre_grasp_dis=0.08))
        # self.face_object_with_torso(self.bottle2, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.bottle2, arm_tag=bottle2_arm_tag, pre_grasp_dis=0.08))

        self.move(
            self.move_by_displacement(arm_tag=bottle1_arm_tag, z=0.1),
            self.move_by_displacement(arm_tag=bottle2_arm_tag, z=0.1),
        )

        self.face_world_point_with_torso(self.left_target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.bottle1,
                target_pose=self.left_target_pose,
                arm_tag=bottle1_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            )
        )
        self.face_world_point_with_torso(self.right_target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.bottle2,
                target_pose=self.right_target_pose,
                arm_tag=bottle2_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            )
        )

        self.info["info"] = {"{A}": "001_bottle/base13", "{B}": "001_bottle/base16"}
        return self.info
