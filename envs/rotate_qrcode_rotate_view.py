from .rotate_qrcode import rotate_qrcode
from .utils import *
import numpy as np
import transforms3d as t3d


class rotate_qrcode_rotate_view(rotate_qrcode):

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

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_lim = [0.58, 1.05] if side > 0 else [-1.05, -0.58]
        while True:
            qrcode_pose = rand_pose_cyl(
                rlim=[-1.35, 1.35],
                thetalim=theta_lim,
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0, 0, 0.707, 0.707],
                rotate_rand=True,
                rotate_lim=[0, 0.7, 0],
            )
            qrcode_cyl = world_to_robot(qrcode_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(qrcode_cyl[1]) < 0.3:
                continue
            break

        self.model_id = int(np.random.choice([0, 1, 2, 3], 1)[0])
        self.qrcode = create_actor(
            self,
            pose=qrcode_pose,
            modelname="070_paymentsign",
            convex=True,
            model_id=self.model_id,
        )

        self.add_prohibit_area(self.qrcode, padding=0.12)

        target_pose = place_pose_cyl(
            [0.5, side * 0.35, 0.74 + self.table_z_bias, 1, 0, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
            quat_frame="world",
        )
        self.target_pose = target_pose

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("left" if self.qrcode.get_pose().p[0] < 0 else "right")

        self.face_object_with_torso(self.qrcode, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.qrcode, arm_tag=arm_tag, pre_grasp_dis=0.05, grasp_dis=-0.01,gripper_pos=-0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.face_world_point_with_torso(self.target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.qrcode,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                pre_dis=0.07,
                dis=0.01,
                constrain="align",  # QR sign rotation target requires orientation alignment.
            )
        )

        self.info["info"] = {
            "{A}": f"070_paymentsign/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info
