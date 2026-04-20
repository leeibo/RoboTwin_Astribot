from .grab_roller import grab_roller
from .utils import *
import numpy as np
import transforms3d as t3d


class grab_roller_rotate_view(grab_roller):

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

        ori_qpos = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0, 0, 0.707, 0.707]]
        self.model_id = int(np.random.choice([0, 2], 1)[0])

        rand_pos = rand_pose_cyl(
            rlim=[0.45, 0.5],
            thetalim=rotate_theta_center(self),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=ori_qpos[self.model_id],
            rotate_rand=True,
            rotate_lim=[0, 0.8, 0],
        )
        self.roller = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="102_roller",
            convex=True,
            model_id=self.model_id,
        )

        self.add_prohibit_area(self.roller, padding=0.1)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        # self.face_object_with_torso(self.roller, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.grasp_actor(self.roller, left_arm_tag, pre_grasp_dis=0.08, contact_point_id=0),
            self.grasp_actor(self.roller, right_arm_tag, pre_grasp_dis=0.08, contact_point_id=1),
        )
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.85 - self.roller.get_pose().p[2]),
            self.move_by_displacement(right_arm_tag, z=0.85 - self.roller.get_pose().p[2]),
        )

        self.info["info"] = {"{A}": f"102_roller/base{self.model_id}"}
        return self.info
