from .move_can_pot import move_can_pot
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d


class move_can_pot_rotate_view(move_can_pot):

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

        self.pot_id = int(np.random.randint(0, 7))
        pot_pose = rand_pose_cyl(
            rlim=[0.48, 0.5],
            thetalim=[-0.12, 0.12],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0, 0, 0, 1],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
        )
        self.pot = create_sapien_urdf_obj(
            scene=self,
            pose=pot_pose,
            modelname="060_kitchenpot",
            modelid=self.pot_id,
            fix_root_link=False,
        )
        pot_pose = self.pot.get_pose()

        while True:
            rand_pos = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=[-1.1, 1.1],
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
            can_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(can_cyl[1]) < 0.45:
                continue
            if (pot_pose.p[0] - rand_pos.p[0])**2 + (pot_pose.p[1] - rand_pos.p[1])**2 < 0.09:
                continue
            break

        id_list = [0, 2, 4, 5, 6]
        self.can_id = int(np.random.choice(id_list))
        self.can = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="105_sauce-can",
            convex=True,
            model_id=self.can_id,
        )
        self.arm_tag = ArmTag("right" if self.can.get_pose().p[0] > 0 else "left")
        self.add_prohibit_area(self.pot, padding=0.03)
        self.add_prohibit_area(self.can, padding=0.1)
        pot_x, pot_y = self.pot.get_pose().p[0], self.pot.get_pose().p[1]
        if self.arm_tag == "left":
            self.prohibited_area.append([pot_x - 0.15, pot_y - 0.1, pot_x, pot_y + 0.1])
        else:
            self.prohibited_area.append([pot_x, pot_y - 0.1, pot_x + 0.15, pot_y + 0.1])
        self.orig_z = self.pot.get_pose().p[2]

        pot_pose = self.pot.get_pose()
        self.pot.set_mass(0.1)
        self.can.set_mass(0.1)
        self.target_pose = sapien.Pose(
            [
                pot_pose.p[0] - 0.2 if self.arm_tag == "left" else pot_pose.p[0] + 0.2,
                pot_pose.p[1],
                0.741 + self.table_z_bias,
            ],
            pot_pose.q,
        )

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = self.arm_tag
        self.face_object_with_torso(self.can, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.can, arm_tag=arm_tag, pre_grasp_dis=0.12,gripper_pos=0.3))
        self.move(self.move_by_displacement(arm_tag, y=-0.1, z=0.1))

        self.face_world_point_with_torso(self.target_pose.p.tolist(), joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.can,
                target_pose=self.target_pose,
                arm_tag=arm_tag,
                pre_dis=0.05,
                dis=0.0,
                constrain="free",
            )
        )

        self.info["info"] = {
            "{A}": f"060_kitchenpot/base{self.pot_id}",
            "{B}": f"105_sauce-can/base{self.can_id}",
            "{a}": str(arm_tag),
        }
        return self.info
