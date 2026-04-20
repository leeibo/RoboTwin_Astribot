from .lift_pot import lift_pot
from .utils import *
import numpy as np
import transforms3d as t3d


class lift_pot_rotate_view(lift_pot):

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

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.model_name = "060_kitchenpot"
        self.model_id = int(np.random.randint(0, 2))

        pot_pose = rand_pose_cyl(
            rlim=[0.46, 0.5],
            thetalim=rotate_theta_center(self),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.704141, 0, 0, 0.71006],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
            quat_frame="cyl",
        )
        self.pot = create_sapien_urdf_obj(
            scene=self,
            pose=pot_pose,
            modelname=self.model_name,
            modelid=self.model_id,
            fix_root_link=False,
        )
        x, y = self.pot.get_pose().p[0], self.pot.get_pose().p[1]
        self.prohibited_area.append([x - 0.3, y - 0.1, x + 0.3, y + 0.1])

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")
        # self.face_object_with_torso(self.pot, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.close_gripper(left_arm_tag, pos=0.5),
            self.close_gripper(right_arm_tag, pos=0.5),
        )
        self.move(
            self.grasp_actor(self.pot, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0),
            self.grasp_actor(self.pot, right_arm_tag, pre_grasp_dis=0.035, contact_point_id=1),
        )
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.88 - self.pot.get_pose().p[2]),
            self.move_by_displacement(right_arm_tag, z=0.88 - self.pot.get_pose().p[2]),
        )

        self.info["info"] = {"{A}": f"{self.model_name}/base{self.model_id}"}
        return self.info
