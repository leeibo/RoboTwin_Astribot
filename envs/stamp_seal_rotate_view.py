from .stamp_seal import stamp_seal
from .utils import *
import numpy as np
import transforms3d as t3d


class stamp_seal_rotate_view(stamp_seal):

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
        theta_lim = rotate_theta_side(self, side=side)
        target_lim = rotate_theta_side(self, side=-side)
        while True:
            rand_pos = rand_pose_cyl(
                rlim=[0.35, 0.45],
                thetalim=theta_lim,

                zlim=[0.741, 0.741],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            seal_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(seal_cyl[1]) < 0.25:
                continue
            break

        self.seal_id = int(np.random.choice([0, 2, 3, 4, 6], 1)[0])
        self.seal = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="100_seal",
            convex=True,
            model_id=self.seal_id,
        )
        self.seal.set_mass(0.05)

        while True:
            target_rand_pose = rand_pose_cyl(
                rlim=[0.45, 0.5],
                thetalim=target_lim,

                zlim=[0.741, 0.741],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            if np.linalg.norm(target_rand_pose.p[:2] - rand_pos.p[:2]) < 0.1:
                continue
            break

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

        half_size = [0.035, 0.035, 0.0005]
        self.target = create_visual_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
        )
        self.add_prohibit_area(self.seal, padding=0.1)
        self.add_prohibit_area(self.target, padding=0.1)
        self.target_pose = self.target.get_pose()

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.seal.get_pose().p[0] > 0 else "left")

        self.face_object_with_torso(self.seal, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.seal, arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=[4, 5, 6, 7]))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))

        target_pose = self.target.get_pose()
        self.face_world_point_with_torso(target_pose.p.tolist(), joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.seal,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.1,
                constrain="free",
            )
        )

        self.info["info"] = {
            "{A}": f"100_seal/base{self.seal_id}",
            "{B}": f"{self.color_name}",
            "{a}": str(arm_tag),
        }
        return self.info
