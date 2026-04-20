from .move_playingcard_away import move_playingcard_away
from .utils import *
import numpy as np
import transforms3d as t3d


class move_playingcard_away_rotate_view(move_playingcard_away):

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

        self.playingcards_id = int(np.random.choice([0, 1, 2], 1)[0])
        self.playingcards = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="081_playingcards",
            convex=True,
            model_id=self.playingcards_id,
        )

        self.prohibited_area.append([-100, -0.3, 100, 0.1])
        self.add_prohibit_area(self.playingcards, padding=0.1)
        self.target_pose = self.playingcards.get_pose()

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("right" if self.playingcards.get_pose().p[0] > 0 else "left")
        # self.face_object_with_torso(self.playingcards, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.playingcards, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=-0.02,gripper_pos=0.3))
        self.move(self.move_by_displacement(arm_tag, x=0.3 if arm_tag == "right" else -0.3))
        self.move(self.open_gripper(arm_tag))

        self.info["info"] = {
            "{A}": f"081_playingcards/base{self.playingcards_id}",
            "{a}": str(arm_tag),
        }
        return self.info
