from .handover_block import handover_block
from .utils import *
import numpy as np
import transforms3d as t3d


class handover_block_rotate_view(handover_block):

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

        box_pose = rand_pose_cyl(
            rlim=[0.45, 0.5],
            thetalim=rotate_theta_side(self, side=1),

            zlim=[0.842, 0.842],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.981, 0, 0, 0.195],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.2],
        )
        self.box = create_box(
            scene=self,
            pose=box_pose,
            half_size=(0.03, 0.03, 0.1),
            color=(1, 0, 0),
            name="box",
            boxtype="long",
        )

        target_pose = rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=rotate_theta_side(self, side=-1),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        self.target_box = create_box(
            scene=self,
            pose=target_pose,
            half_size=(0.05, 0.05, 0.005),
            color=(0, 0, 1),
            name="target_box",
            is_static=True,
        )

        self.add_prohibit_area(self.box, padding=0.1)
        self.add_prohibit_area(self.target_box, padding=0.1)
        self.block_middle_pose = place_pose_cyl(
            [0.5, 0.0, 0.9 + self.table_z_bias, 0, 1, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        box_cyl = world_to_robot(self.box.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        grasp_arm_tag = ArmTag("left" if box_cyl[1] >= 0 else "right")
        place_arm_tag = grasp_arm_tag.opposite

        # self.face_object_with_torso(self.box, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.grasp_actor(
                self.box,
                arm_tag=grasp_arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=[0, 1, 2, 3],
            )
        )
        self.move(self.move_by_displacement(grasp_arm_tag, z=0.1))

        self.face_world_point_with_torso(self.block_middle_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.box,
                target_pose=self.block_middle_pose,
                arm_tag=grasp_arm_tag,
                functional_point_id=0,
                pre_dis=0,
                dis=0,
                is_open=False,
                constrain="free",
            )
        )

        # self.face_object_with_torso(self.box, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.grasp_actor(
                self.box,
                arm_tag=place_arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=[ 5, 6, 7],
            )
        )
        self.move(self.open_gripper(grasp_arm_tag))
        self.move(self.move_by_displacement(grasp_arm_tag, z=0.1, move_axis="arm"))

        target_pose = self.target_box.get_functional_point(1, "pose")
        self.face_world_point_with_torso(target_pose.p.tolist(), joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.back_to_origin(grasp_arm_tag),
            self.place_actor(
                self.box,
                target_pose=target_pose,
                arm_tag=place_arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                constrain="align",  # Keep long block orientation matched to target support pad.
                pre_dis_axis="fp",
            ),
        )

        return self.info
