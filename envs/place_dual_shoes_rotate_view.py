from .place_dual_shoes import place_dual_shoes
from .utils import *
import numpy as np
import transforms3d as t3d
from ._GLOBAL_CONFIGS import *

class place_dual_shoes_rotate_view(place_dual_shoes):

    def setup_demo(self, is_test=False, **kwags):
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        super().setup_demo(is_test=is_test, **kwags)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    def _scan_scene_two_views(self, object_list=None):
        scan_r = 0.62
        scan_z = 0.78 + self.table_z_bias
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

        shoe_box_pose = place_pose_cyl(
            [0.5, 0.0, 0.74, 0.5, 0.5, -0.5, -0.5],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="pose",
        )
        self.shoe_box = create_actor(
            self,
            pose=shoe_box_pose,
            modelname="007_shoe-box",
            convex=True,
            is_static=True,
        )

        self.shoe_id = int(np.random.choice([i for i in range(10)]))
        left_shoe_pose = rand_pose_cyl(
            rlim=[0.4, 0.45],
            thetalim=[0.8, 1.1],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        self.left_shoe = create_actor(
            self,
            pose=left_shoe_pose,
            modelname="041_shoe",
            convex=True,
            model_id=self.shoe_id,
        )

        right_shoe_pose = rand_pose_cyl(
            rlim=[0.4, 0.45],
            thetalim=[-1.1, -0.8],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        self.right_shoe = create_actor(
            self,
            pose=right_shoe_pose,
            modelname="041_shoe",
            convex=True,
            model_id=self.shoe_id,
        )

        self.add_prohibit_area(self.left_shoe, padding=0.02)
        self.add_prohibit_area(self.right_shoe, padding=0.02)
        self.prohibited_area.append([-0.15, -0.25, 0.15, 0.01])
        self.right_shoe_middle_pose = [0.35, -0.05, 0.79, 0, 1, 0, 0]

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")
        left_target = self.shoe_box.get_functional_point(0)
        right_target = self.shoe_box.get_functional_point(1)

        self.face_object_with_torso(self.left_shoe, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.left_shoe, arm_tag=left_arm_tag, pre_grasp_dis=0.1,gripper_pos=0.3,grasp_dis=-0.02))
        self.move(self.move_by_displacement(left_arm_tag, z=0.15))
        self.face_world_point_with_torso(left_target[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.left_shoe,
                target_pose=left_target,
                arm_tag=left_arm_tag,
                functional_point_id=0,
                pre_dis=0.07,
                dis=0.02,
                constrain="align",  # Shoe-box task requires pose orientation to match the box.
            )
        )
        self.move(self.back_to_origin(left_arm_tag))

        self.face_object_with_torso(self.right_shoe, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.right_shoe, arm_tag=right_arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(right_arm_tag, z=0.15))
        self.face_world_point_with_torso(right_target[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.right_shoe,
                target_pose=right_target,
                arm_tag=right_arm_tag,
                functional_point_id=0,
                pre_dis=0.07,
                dis=0.02,
                constrain="align",  # Shoe-box task requires pose orientation to match the box.
            )
        )

        self.delay(3)

        self.info["info"] = {
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": "007_shoe-box/base0",
        }
        return self.info
