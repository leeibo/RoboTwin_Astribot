from .move_stapler_pad import move_stapler_pad
from .utils import *
import numpy as np
import transforms3d as t3d


class move_stapler_pad_rotate_view(move_stapler_pad):

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

        while True:
            rand_pos = rand_pose_cyl(
                rlim=[0.35, 0.45],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < 0.2:
                continue
            break

        self.stapler_id = int(np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0])
        self.stapler = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
        )

        stapler_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
        same_side = 1.0 if stapler_cyl[1] >= 0 else -1.0
        while True:
            target_rand_pose = rand_pose_cyl(
                rlim=[0.48, 0.5],
                thetalim=rotate_theta_side(self, side=-same_side),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            if np.linalg.norm(target_rand_pose.p[:2] - rand_pos.p[:2]) < 0.1:
                continue
            break

        half_size = [0.03, 0.03, 0.0005]
        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
        }
        color_items = list(colors.items())
        color_index = int(np.random.choice(len(color_items)))
        self.color_name, self.color_value = color_items[color_index]

        self.pad = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
        )
        self.add_prohibit_area(self.stapler, padding=0.1)
        self.add_prohibit_area(self.pad, padding=0.15)
        self.pad_pose = self.pad.get_pose().p.tolist() + [0.707, 0, 0, 0.707]

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.stapler.get_pose().p[0] > 0 else "left")
        self.face_object_with_torso(self.stapler, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))

        self.face_world_point_with_torso(self.pad_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.stapler,
                target_pose=self.pad_pose,
                arm_tag=arm_tag,
                pre_dis=0.1,
                dis=0.0,
                constrain="free",  # Success check requires orientation consistency on the pad.
            )
        )

        self.info["info"] = {
            "{A}": f"048_stapler/base{self.stapler_id}",
            "{B}": self.color_name,
            "{a}": str(arm_tag),
        }
        return self.info
    def check_success(self):
        stapler_pose = self.stapler.get_pose().p
        target_pos = self.pad.get_pose().p
        eps = [0.02, 0.02, 0.01]
        return (np.all(abs(stapler_pose - target_pos) < np.array(eps))) and (self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
  