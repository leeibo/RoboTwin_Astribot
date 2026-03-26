from .handover_mic import handover_mic
from .utils import *
import numpy as np
import transforms3d as t3d
from ._GLOBAL_CONFIGS import *

class handover_mic_rotate_view(handover_mic):

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
                rlim=[0.45, 0.5],
                thetalim=[-0.95, 0.95],
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=False,
            )
            cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < 0.45:
                continue
            break

        self.microphone_id = int(np.random.choice([0, 4, 5], 1)[0])
        self.microphone = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="018_microphone",
            convex=True,
            model_id=self.microphone_id,
        )

        self.add_prohibit_area(self.microphone, padding=0.07)
        self.handover_middle_pose = place_pose_cyl(
            [0.5, 0.0, 0.98, 0, 1, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self.grasp_arm_tag = ArmTag("right" if self.microphone.get_pose().p[0] > 0 else "left")
        self.handover_arm_tag = self.grasp_arm_tag.opposite

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        grasp_arm_tag = ArmTag("right" if self.microphone.get_pose().p[0] > 0 else "left")
        handover_arm_tag = grasp_arm_tag.opposite

        self.face_object_with_torso(self.microphone, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.grasp_actor(
                self.microphone,
                arm_tag=grasp_arm_tag,
                contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15],
                pre_grasp_dis=0.1,
            )
        )
        self.move(
            self.move_by_displacement(
                grasp_arm_tag,
                z=0.12,
                quat=(
                    GRASP_DIRECTION_DIC["front_right"]
                    if grasp_arm_tag == "left"
                    else GRASP_DIRECTION_DIC["front_left"]
                ),
                move_axis="arm",
            )
        )

        self.face_world_point_with_torso(self.handover_middle_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.microphone,
                arm_tag=grasp_arm_tag,
                target_pose=self.handover_middle_pose,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            )
        )

        self.face_object_with_torso(self.microphone, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.grasp_actor(
                self.microphone,
                arm_tag=handover_arm_tag,
                contact_point_id=[0, 2, 3, 4, 5, 6, 7, 8],
                pre_grasp_dis=0.1,
            )
        )
        self.move(self.open_gripper(grasp_arm_tag))
        self.move(
            self.move_by_displacement(grasp_arm_tag, z=0.07, move_axis="arm"),
            self.move_by_displacement(handover_arm_tag, x=0.05 if handover_arm_tag == "right" else -0.05),
        )

        self.info["info"] = {
            "{A}": f"018_microphone/base{self.microphone_id}",
            "{a}": str(grasp_arm_tag),
            "{b}": str(handover_arm_tag),
        }
        return self.info
