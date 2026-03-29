from .press_stapler import press_stapler
from .utils import *
import numpy as np
import transforms3d as t3d


class press_stapler_rotate_view(press_stapler):

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

        rand_pos = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=rotate_theta_center(self),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
        )
        self.stapler_id = int(np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0])
        self.stapler = create_actor(
            self,
            pose=rand_pos,
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=True,
        )
        self.add_prohibit_area(self.stapler, padding=0.05)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("left" if self.stapler.get_pose().p[0] < 0 else "right")
        self.face_object_with_torso(self.stapler, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.1, contact_point_id=2)
        )
        self.move(self.close_gripper(arm_tag=arm_tag))

        self.face_object_with_torso(self.stapler, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.02, grasp_dis=0.02, contact_point_id=2)
        )

        self.info["info"] = {"{A}": f"048_stapler/base{self.stapler_id}", "{a}": str(arm_tag)}
        return self.info
