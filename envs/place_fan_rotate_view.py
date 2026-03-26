from .place_fan import place_fan
from .utils import *
import numpy as np
import transforms3d as t3d


class place_fan_rotate_view(place_fan):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs.setdefault("table_shape", "fan")
        kwargs.setdefault("fan_center_on_robot", True)
        kwargs.setdefault("fan_outer_radius", 0.9)
        kwargs.setdefault("fan_inner_radius", 0.3)
        kwargs.setdefault("fan_angle_deg", 220)
        kwargs.setdefault("fan_center_deg", 90)
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

        fan_pose = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=[-0.35, 0.35],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.0, 0.0, 0.707, 0.707],
            rotate_rand=True,
            rotate_lim=[0, 2 * np.pi, 0],
        )
        self.fan_id = int(np.random.choice([4, 5]))
        self.fan = create_actor(
            scene=self,
            pose=fan_pose,
            modelname="099_fan",
            convex=True,
            model_id=self.fan_id,
        )
        self.fan.set_mass(0.01)

        pad_theta = -0.85 if self.fan.get_pose().p[0] > 0 else 0.85
        pad_pose = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=[pad_theta - 0.15, pad_theta + 0.15],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
            "Orange": (1, 0.5, 0),
            "Purple": (0.5, 0, 0.5),
            "Brown": (0.65, 0.4, 0.16),
            "Pink": (1, 0.75, 0.8),
            "Lime": (0.5, 1, 0),
            "Olive": (0.5, 0.5, 0),
            "Teal": (0, 0.5, 0.5),
            "Maroon": (0.5, 0, 0),
            "Navy": (0, 0, 0.5),
            "Coral": (1, 0.5, 0.31),
            "Turquoise": (0.25, 0.88, 0.82),
            "Indigo": (0.29, 0, 0.51),
            "Beige": (0.96, 0.91, 0.81),
            "Tan": (0.82, 0.71, 0.55),
            "Silver": (0.75, 0.75, 0.75),
        }
        color_items = list(colors.items())
        idx = int(np.random.choice(len(color_items)))
        self.color_name, self.color_value = color_items[idx]

        self.pad = create_box(
            scene=self,
            pose=pad_pose,
            half_size=(0.05, 0.05, 0.001),
            color=self.color_value,
            name="box",
        )
        self.pad.set_mass(1)
        self.add_prohibit_area(self.fan, padding=0.07)
        self.prohibited_area.append([
            pad_pose.p[0] - 0.15,
            pad_pose.p[1] - 0.15,
            pad_pose.p[0] + 0.15,
            pad_pose.p[1] + 0.15,
        ])
        target_pose = self.pad.get_pose().p
        self.target_pose = target_pose.tolist() + [1, 0, 0, 0]

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.fan.get_pose().p[0] > 0 else "left")
        self.face_object_with_torso(self.fan, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.fan, arm_tag=arm_tag, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))

        self.face_world_point_with_torso(self.target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.fan,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="free",  # Orientation is explicitly checked in this task.
                pre_dis=0.04,
                dis=0.005,
            )
        )

        self.info["info"] = {
            "{A}": f"099_fan/base{self.fan_id}",
            "{B}": self.color_name,
            "{a}": str(arm_tag),
        }
        return self.info
    def check_success(self):
        fan_qpose = self.fan.get_pose().q
        fan_pose = self.fan.get_pose().p

        target_pose = self.target_pose[:3]
        target_qpose = np.array([0.707, 0.707, 0.0, 0.0])

        if fan_qpose[0] < 0:
            fan_qpose *= -1

        eps = np.array([0.05, 0.05, 0.05, 0.05])

        return np.all(abs(fan_pose - target_pose) < np.array([0.04, 0.04, 0.04]))
