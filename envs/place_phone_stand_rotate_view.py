from .place_phone_stand import place_phone_stand
from .utils import *
import numpy as np
import transforms3d as t3d


class place_phone_stand_rotate_view(place_phone_stand):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs.setdefault("table_shape", "fan")
        kwargs.setdefault("fan_center_on_robot", True)
        kwargs.setdefault("fan_outer_radius", 0.9)
        kwargs.setdefault("fan_inner_radius", 0.3)
        kwargs.setdefault("fan_angle_deg", 220)
        kwargs.setdefault("fan_center_deg", 90)
        kwargs = init_rotate_theta_bounds(self, kwargs)
        super().setup_demo(is_test=is_test, **kwargs)

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

        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]

        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_phone_lim = rotate_theta_side(self, side=side)
        theta_stand_lim = rotate_theta_side(self, side=-side)

        self.phone_id = int(np.random.choice([0, 1, 2, 4], 1)[0])
        while True:
            phone_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=theta_phone_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=ori_quat[self.phone_id],
                rotate_rand=True,
                rotate_lim=[0, 0.7, 0],
            )
            phone_cyl = world_to_robot(phone_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(phone_cyl[1]) < 0.3:
                continue
            break

        while True:
            stand_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=theta_stand_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=False,
            )
            if np.linalg.norm(phone_pose.p[:2] - stand_pose.p[:2]) < 0.15:
                continue
            stand_cyl = world_to_robot(stand_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(stand_cyl[1]) < 0.2:
                continue
            break

        self.phone = create_actor(
            scene=self,
            pose=phone_pose,
            modelname="077_phone",
            convex=True,
            model_id=self.phone_id,
        )
        self.phone.set_mass(0.05)

        self.stand_id = int(np.random.choice([1, 2], 1)[0])
        self.stand = create_actor(
            scene=self,
            pose=stand_pose,
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=True,
        )
        self.add_prohibit_area(self.phone, padding=0.15)
        self.add_prohibit_area(self.stand, padding=0.15)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("left" if self.phone.get_pose().p[0] < 0 else "right")

        # self.face_object_with_torso(self.phone, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.phone, arm_tag=arm_tag, pre_grasp_dis=0.08,grasp_dis=0.01))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))

        stand_func_pose = self.stand.get_functional_point(0)
        self.face_world_point_with_torso(stand_func_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.phone,
                arm_tag=arm_tag,
                target_pose=stand_func_pose,
                functional_point_id=0,
                dis=0,
                constrain="align",  # Phone must dock into stand slot with orientation alignment.
            )
        )

        self.info["info"] = {
            "{A}": f"077_phone/base{self.phone_id}",
            "{B}": f"078_phonestand/base{self.stand_id}",
            "{a}": str(arm_tag),
        }
        return self.info
