from .scan_object import scan_object
from .utils import *
import numpy as np
import transforms3d as t3d


class scan_object_rotate_view(scan_object):

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
        scan_r = 0.64
        scan_z = 0.9 + self.table_z_bias
        for theta in self._get_scan_thetas_from_object_list(object_list, fallback_thetas=[1.0, -1.0]):
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

        scanner_side = 1.0 if np.random.rand() < 0.5 else -1.0
        scanner_theta_lim = rotate_theta_side(self, side=scanner_side)
        object_theta_lim = rotate_theta_side(self, side=-scanner_side)

        while True:
            scanner_pose = rand_pose_cyl(
                rlim=[0.5, 0.5],
                thetalim=scanner_theta_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0, 0, 0.707, 0.707],
                rotate_rand=True,
                rotate_lim=[0, 1.2, 0],
            )
            scanner_cyl = world_to_robot(scanner_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(scanner_cyl[1]) < 0.3:
                continue
            break

        while True:
            object_pose = rand_pose_cyl(
                rlim=[0.5, 0.5],
                thetalim=object_theta_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 1.2, 0],
            )
            object_cyl = world_to_robot(object_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(object_cyl[1]) < 0.3:
                continue
            if np.linalg.norm(scanner_pose.p[:2] - object_pose.p[:2]) < 0.18:
                continue
            break

        self.scanner_id = int(np.random.choice([0, 1, 2, 3, 4], 1)[0])
        self.scanner = create_actor(
            scene=self.scene,
            pose=scanner_pose,
            modelname="024_scanner",
            convex=True,
            model_id=self.scanner_id,
        )

        self.object_id = int(np.random.choice([0, 1, 2, 3, 4, 5], 1)[0])
        self.object = create_actor(
            scene=self.scene,
            pose=object_pose,
            modelname="112_tea-box",
            convex=True,
            model_id=self.object_id,
        )
        self.add_prohibit_area(self.scanner, padding=0.1)
        self.add_prohibit_area(self.object, padding=0.1)

        self.left_object_target_pose = place_pose_cyl(
            [0.42, 0.1, 0.95, 0.707, 0, -0.707, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self.right_object_target_pose = place_pose_cyl(
            [0.42, -0.1, 0.95, 0.707, 0, 0.707, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        scanner_arm_tag = ArmTag("left" if self.scanner.get_pose().p[0] < 0 else "right")
        object_arm_tag = scanner_arm_tag.opposite

        # self.face_object_with_torso(self.scanner, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.scanner, arm_tag=scanner_arm_tag, pre_grasp_dis=0.08))

        # self.face_object_with_torso(self.object, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.object, arm_tag=object_arm_tag, pre_grasp_dis=0.08))

        self.move(
            self.move_by_displacement(
                arm_tag=scanner_arm_tag,
                x=0.05 if scanner_arm_tag == "right" else -0.05,
                z=0.13,
            ),
            self.move_by_displacement(
                arm_tag=object_arm_tag,
                x=0.05 if object_arm_tag == "right" else -0.05,
                z=0.13,
            ),
        )

        object_target_pose = self.right_object_target_pose if object_arm_tag == "right" else self.left_object_target_pose
        self.face_world_point_with_torso(object_target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.object,
                arm_tag=object_arm_tag,
                target_pose=object_target_pose,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            )
        )

        scanner_target_pose = self.object.get_functional_point(1)
        self.face_world_point_with_torso(scanner_target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.scanner,
                arm_tag=scanner_arm_tag,
                target_pose=scanner_target_pose,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.05,
                is_open=False,
                constrain="free",
            )
        )

        self.info["info"] = {
            "{A}": f"112_tea-box/base{self.object_id}",
            "{B}": f"024_scanner/base{self.scanner_id}",
            "{a}": str(object_arm_tag),
            "{b}": str(scanner_arm_tag),
        }
        return self.info
