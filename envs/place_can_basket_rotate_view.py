from .place_can_basket import place_can_basket
from .utils import *
import numpy as np
import transforms3d as t3d


class place_can_basket_rotate_view(place_can_basket):

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

        self.arm_tag = ArmTag({0: "left", 1: "right"}[int(np.random.randint(0, 2))])
        self.basket_name = "110_basket"
        self.basket_id = [0, 1][int(np.random.randint(0, 2))]

        can_dict = {"071_can": [0, 1, 2, 3, 5, 6]}
        self.can_name = "071_can"
        self.can_id = int(np.random.choice(can_dict[self.can_name]))

        if self.arm_tag == "left":
            basket_pose = place_pose_cyl(
                [0.5, -0.05, 0.741, 0.5, 0.5, 0.5, 0.5],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="pose",
            )
            can_pose = rand_pose_cyl(
                rlim=[0.35, 0.5],
                thetalim=rotate_theta_side(self, side=1),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.707225, 0.706849, -0.0100455, -0.00982061],
                rotate_rand=False,
            )
        else:
            basket_pose = place_pose_cyl(
                [0.5, 0.05, 0.741, 0.5, 0.5, 0.5, 0.5],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="pose",
            )
            can_pose = rand_pose_cyl(
                rlim=[0.35, 0.5],
                thetalim=rotate_theta_side(self, side=-1),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.707225, 0.706849, -0.0100455, -0.00982061],
                rotate_rand=False,
            )

        self.basket = create_actor(
            scene=self,
            pose=basket_pose,
            modelname=self.basket_name,
            model_id=self.basket_id,
            convex=True,
        )
        self.can = create_actor(
            scene=self,
            pose=can_pose,
            modelname=self.can_name,
            model_id=self.can_id,
            convex=True,
        )

        self.start_height = self.basket.get_pose().p[2]
        self.basket.set_mass(0.5)
        self.can.set_mass(0.01)
        self.add_prohibit_area(self.can, padding=0.1)
        self.add_prohibit_area(self.basket, padding=0.05)
        self.object_start_height = self.can.get_pose().p[2]

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        self.face_object_with_torso(self.can, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.can, arm_tag=self.arm_tag, pre_grasp_dis=0.12))

        place_pose = self.get_arm_pose(arm_tag=self.arm_tag)
        f0 = np.array(self.basket.get_functional_point(0))
        f1 = np.array(self.basket.get_functional_point(1))
        if np.linalg.norm(f0[:2] - place_pose[:2]) < np.linalg.norm(f1[:2] - place_pose[:2]):
            place_pose = f0
            place_pose[:2] = f0[:2]
            place_pose[3:] = (-1, 0, 0, 0) if self.arm_tag == "left" else (0.05, 0, 0, 0.99)
        else:
            place_pose = f1
            place_pose[:2] = f1[:2]
            place_pose[3:] = (-1, 0, 0, 0) if self.arm_tag == "left" else (0.05, 0, 0, 0.99)

        self.face_world_point_with_torso(place_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.can,
                arm_tag=self.arm_tag,
                target_pose=place_pose,
                dis=0.02,
                is_open=False,
                constrain="free",
            )
        )

        if self.plan_success is False:
            self.plan_success = True
            place_pose[0] += -0.15 if self.arm_tag == "left" else 0.15
            place_pose[2] += 0.15
            self.face_world_point_with_torso(place_pose[:3], joint_name_prefer="astribot_torso_joint_2")
            self.move(self.move_to_pose(arm_tag=self.arm_tag, target_pose=place_pose))
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=-0.1))
            self.move(self.open_gripper(arm_tag=self.arm_tag))
            self.face_object_with_torso(self.basket, joint_name_prefer="astribot_torso_joint_2")
            self.move(
                self.back_to_origin(arm_tag=self.arm_tag),
                self.grasp_actor(self.basket, arm_tag=self.arm_tag.opposite, pre_grasp_dis=0.02),
            )
        else:
            self.move(self.open_gripper(arm_tag=self.arm_tag))
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
            self.face_object_with_torso(self.basket, joint_name_prefer="astribot_torso_joint_2")
            self.move(
                self.back_to_origin(arm_tag=self.arm_tag),
                self.grasp_actor(self.basket, arm_tag=self.arm_tag.opposite, pre_grasp_dis=0.08),
            )

        self.move(self.close_gripper(arm_tag=self.arm_tag.opposite))
        self.move(
            self.move_by_displacement(
                arm_tag=self.arm_tag.opposite,
                x=-0.02 if self.arm_tag.opposite == "left" else 0.02,
                z=0.05,
            )
        )

        self.info["info"] = {
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{B}": f"{self.basket_name}/base{self.basket_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info
