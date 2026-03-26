from .move_pillbottle_pad import move_pillbottle_pad
from .utils import *
import numpy as np
import transforms3d as t3d


class move_pillbottle_pad_rotate_view(move_pillbottle_pad):

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
                thetalim=[-1.02, 1.02],
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < 0.2:
                continue
            break

        self.pillbottle_id = int(np.random.choice([1, 2, 3, 4, 5], 1)[0])
        self.pillbottle = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="080_pillbottle",
            convex=True,
            model_id=self.pillbottle_id,
        )
        self.pillbottle.set_mass(0.05)

        pill_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
        same_side = 1.0 if pill_cyl[1] >= 0 else -1.0
        while True:
            target_rand_pose = rand_pose_cyl(
                rlim=[0.35, 0.45],
                thetalim=[same_side * -1.1, same_side * 0.45],
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            if np.linalg.norm(target_rand_pose.p[:2] - rand_pos.p[:2]) < 0.1:
                continue
            break

        self.pad = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=[0.04, 0.04, 0.0005],
            color=(0, 0, 1),
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.pillbottle, padding=0.05)
        self.add_prohibit_area(self.pad, padding=0.1)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.pillbottle.get_pose().p[0] > 0 else "left")
        self.face_object_with_torso(self.pillbottle, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.pillbottle, arm_tag=arm_tag, pre_grasp_dis=0.12, gripper_pos=0.3,grasp_dis=-0.02))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))

        target_pose = self.pad.get_functional_point(1)
        self.face_world_point_with_torso(target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.pillbottle,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.05,
                dis=0,
                functional_point_id=0,
                pre_dis_axis="fp",
                constrain="free",
            )
        )

        self.info["info"] = {
            "{A}": f"080_pillbottle/base{self.pillbottle_id}",
            "{a}": str(arm_tag),
        }
        return self.info
