from .shake_bottle_horizontally import shake_bottle_horizontally
from .utils import *
import numpy as np
import transforms3d as t3d


class shake_bottle_horizontally_rotate_view(shake_bottle_horizontally):

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
        scan_r = 0.6
        scan_z = 0.9 + self.table_z_bias
        for theta in self._get_scan_thetas_from_object_list(object_list, fallback_thetas=[0.9, -0.9]):
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

        self.id_list = [i for i in range(20)]
        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_lim = rotate_theta_side(self, side=side)
        while True:
            rand_pos = rand_pose_cyl(
                rlim=[0.3, 0.45],
                thetalim=theta_lim,

                zlim=[0.785, 0.785],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0, 0, 1, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 4],
            )
            bottle_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(bottle_cyl[1]) < 0.35:
                continue
            break

        self.bottle_id = int(np.random.choice(self.id_list))
        self.bottle = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="001_bottle",
            convex=True,
            model_id=self.bottle_id,
        )
        self.bottle.set_mass(0.05)
        self.add_prohibit_area(self.bottle, padding=0.05)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] > 0 else "left")
        self.face_object_with_torso(self.bottle, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.bottle, arm_tag=arm_tag, pre_grasp_dis=0.1,gripper_pos=0.2))

        target_quat = [0.707, 0, 0, 0.707]
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, quat=target_quat))

        y_rotation = t3d.euler.euler2quat(0, (np.pi / 2), 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, target_quat)
        target_quat = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]
        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=target_quat))

        quat1 = deepcopy(target_quat)
        quat2 = deepcopy(target_quat)
        y_rotation = t3d.euler.euler2quat(0, (np.pi / 8) * 7, 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat1)
        quat1 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        y_rotation = t3d.euler.euler2quat(0, -7 * (np.pi / 8), 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat2)
        quat2 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.0, quat=quat1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.03, quat=quat2))
        for _ in range(2):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, quat=quat1))
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.05, quat=quat2))

        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=target_quat))

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle_id}",
            "{a}": str(arm_tag),
        }
        return self.info
