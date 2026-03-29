from .adjust_bottle import adjust_bottle
from .utils import *
import numpy as np
import transforms3d as t3d


class adjust_bottle_rotate_view(adjust_bottle):

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
        scan_z = 0.90 + self.table_z_bias
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

    def _sample_bottle_pose(self, qpos):
        side_thetalim = rotate_theta_side(self, side=1 if self.qpose_tag == 0 else -1)
        for _ in range(120):
            pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=side_thetalim,

                zlim=[0.752, 0.752],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.35],
                qpos=qpos,
            )
            cyl = world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if not (0.38 <= cyl[0] <= 0.55):
                continue
            if abs(cyl[1]) < 0.45:
                continue
            if self.qpose_tag == 0 and pose.p[0] >= -0.05:
                continue
            if self.qpose_tag == 1 and pose.p[0] <= 0.05:
                continue
            return pose

        fallback_band = rotate_theta_side(self, side=1 if self.qpose_tag == 0 else -1)
        fallback_theta = fallback_band[1] if self.qpose_tag == 0 else fallback_band[0]
        return place_pose_cyl(
            [0.46, fallback_theta, 0.752, qpos[0], qpos[1], qpos[2], qpos[3]],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="pose",
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.qpose_tag = int(np.random.randint(0, 2))
        qposes = [[0.707, 0.0, 0.0, -0.707], [0.707, 0.0, 0.0, 0.707]]
        self.model_id = int(np.random.choice([13, 16]))

        bottle_pose = self._sample_bottle_pose(qposes[self.qpose_tag])
        self.bottle = create_actor(
            scene=self,
            pose=bottle_pose,
            modelname="001_bottle",
            convex=True,
            model_id=self.model_id,
        )

        self.delay(4)
        self.add_prohibit_area(self.bottle, padding=0.15)

        self.left_target_pose = place_pose_cyl(
            [0.58, 0.72, 0.95, 0, 1, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self.right_target_pose = place_pose_cyl(
            [0.58, -0.72, 0.95, 0, 1, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.qpose_tag == 1 else "left")
        target_pose = self.right_target_pose if self.qpose_tag == 1 else self.left_target_pose

        self.face_object_with_torso(self.bottle, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        self.face_world_point_with_torso(
            target_pose[:3],
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.move(
            self.place_actor(
                self.bottle,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                is_open=False,
                constrain="free",
            )
        )

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info
