from ._base_task import Base_Task
from .utils import *
import numpy as np
import transforms3d as t3d


class blocks_ranking_rgb_rotate_view(Base_Task):

    def setup_demo(self, **kwags):
        # Use fan table by default for rotating-view data.
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        super()._init_task_env_(**kwags)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    def _sample_block_pose_cyl(self, size, existing_pose_lst):
        z = 0.741 + self.table_z_bias + size
        while True:
            pose = rand_pose_cyl(
                rlim=[0.40, 0.5],
                thetalim=[-1.05, 1.05],
                zlim=[z, z],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.75],
                qpos=[1, 0, 0, 0],
            )
            if self._check_block_pose_valid(pose, existing_pose_lst):
                return pose

    @staticmethod
    def _check_block_pose_valid(block_pose, block_pose_lst):
        for pose in block_pose_lst:
            if np.sum((block_pose.p[:2] - pose.p[:2])**2) < 0.01:
                return False
        return True

    def _is_already_ranked(self, block_pose_lst):
        cyl = [
            world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            for pose in block_pose_lst
        ]
        theta = [c[1] for c in cyl]
        # red-left > green-mid > blue-right in theta (robot-centric).
        return theta[0] > theta[1] > theta[2]

    def _scan_head_views(self):
        scan_r = 0.66
        scan_z = 0.86 + self.table_z_bias
        # One global sweep from left to right.
        for theta in [1,-1]:
            scan_point = place_point_cyl(
                [scan_r, theta, scan_z],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="list",
            )
            self.look_at_world_point_with_head(scan_point, max_iter=35, tol_angle_rad=2e-3)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        while True:
            block_pose_lst = []
            size = np.random.uniform(0.015, 0.025)
            for _ in range(3):
                block_pose_lst.append(self._sample_block_pose_cyl(size=size, existing_pose_lst=block_pose_lst))
            if self._is_already_ranked(block_pose_lst):
                continue
            break

        half_size = (size, size, size)
        self.block1 = create_box(
            scene=self,
            pose=block_pose_lst[0],
            half_size=half_size,
            color=(1, 0, 0),
            name="box",
        )
        self.block2 = create_box(
            scene=self,
            pose=block_pose_lst[1],
            half_size=half_size,
            color=(0, 1, 0),
            name="box",
        )
        self.block3 = create_box(
            scene=self,
            pose=block_pose_lst[2],
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
        )

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)

        target_r = np.random.uniform(0.6, 0.7)
        target_theta_mid = np.random.uniform(-0.04, 0.04)
        target_theta_gap = np.random.uniform(0.17, 0.22)
        target_z = 0.74 + self.table_z_bias
        target_quat = [0, 1, 0, 0]

        y_pose = np.random.uniform(-0.1, 0)

        # Define target poses for each block with random x positions
        self.block1_target_pose = [
            np.random.uniform(-0.09, -0.08),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block2_target_pose = [
            np.random.uniform(-0.01, 0.01),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block3_target_pose = [
            np.random.uniform(0.08, 0.09),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

    def play_once(self):
        self.last_gripper = None
        self._scan_head_views()

        arm_tag1 = self.pick_and_place_block(self.block1, self.block1_target_pose)
        arm_tag2 = self.pick_and_place_block(self.block2, self.block2_target_pose)
        arm_tag3 = self.pick_and_place_block(self.block3, self.block3_target_pose)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }
        return self.info

    def pick_and_place_block(self, block, target_pose=None):
        self.look_at_object(block)

        block_pose = block.get_pose().p.tolist()
        block_cyl = world_to_robot(block_pose, self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if block_cyl[1] >= 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.01),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.01))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        self.look_at_world_point_with_head(target_pose[:3], max_iter=35, tol_angle_rad=2e-3)

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="free",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))

        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        block1_pose = self.block1.get_pose().p.tolist()
        block2_pose = self.block2.get_pose().p.tolist()
        block3_pose = self.block3.get_pose().p.tolist()

        c1 = world_to_robot(block1_pose, self.robot_root_xy, self.robot_yaw)
        c2 = world_to_robot(block2_pose, self.robot_root_xy, self.robot_yaw)
        c3 = world_to_robot(block3_pose, self.robot_root_xy, self.robot_yaw)

        theta_eps = 0.35
        radius_eps = 0.14
        same_arc = (abs(c1[0] - c2[0]) < radius_eps and abs(c2[0] - c3[0]) < radius_eps)
        ordered = (c1[1] > c2[1] > c3[1])
        compact = (abs(c1[1] - c2[1]) < theta_eps and abs(c2[1] - c3[1]) < theta_eps)
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return same_arc and ordered and compact and gripper_open
