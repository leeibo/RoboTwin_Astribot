from .blocks_ranking_rgb import blocks_ranking_rgb
from .utils import *
import numpy as np
import transforms3d as t3d


class blocks_ranking_rgb_rotate_view(blocks_ranking_rgb):

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

    @staticmethod
    def _valid_spacing(new_pose, existing_pose_lst, min_dist_sq=0.012):
        for pose in existing_pose_lst:
            if np.sum(np.square(new_pose.p[:2] - pose.p[:2])) < min_dist_sq:
                return False
        return True

    @staticmethod
    def _far_from_target_band(new_pose, target_xy_lst, min_dist_sq=0.012):
        for xy in target_xy_lst:
            if np.sum(np.square(new_pose.p[:2] - xy)) < min_dist_sq:
                return False
        return True

    def _sample_target_gap(self):
        return float(np.random.uniform(0.16, 0.21))

    def _sample_anchor_pose(self, size, target_gap):
        theta_half = rotate_theta_half(self)
        theta_lo = max(0.15, -theta_half + 2 * target_gap + 0.15)
        theta_hi = theta_half - 0.12
        if theta_lo > theta_hi:
            theta_lo = max(-theta_half + 2 * target_gap + 0.05, 0.0)
            theta_hi = max(theta_lo, theta_half - 0.05)

        for _ in range(120):
            pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=[theta_lo, theta_hi],
                zlim=[0.741 + size, 0.741 + size],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.75],
            )
            if world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)[0] < 0.4:
                continue
            return deepcopy(pose)

        fallback_theta = max(theta_lo, min(theta_hi, max(0.15, target_gap + 0.15)))
        return rand_pose_cyl(
            rlim=[0.45, 0.45],
            thetalim=[fallback_theta, fallback_theta],
            zlim=[0.741 + size, 0.741 + size],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

    def _sample_block_pose(self, size, existing_pose_lst, avoid_xy_lst):
        for _ in range(120):
            pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.741 + size, 0.741 + size],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.75],
            )
            cyl = world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if cyl[0] < 0.4:
                continue
            if not self._valid_spacing(pose, existing_pose_lst):
                continue
            if not self._far_from_target_band(pose, avoid_xy_lst):
                continue
            return pose
        fallback_theta = float(np.clip(-0.35 * rotate_theta_half(self), -rotate_theta_half(self), rotate_theta_half(self)))
        return rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=[fallback_theta, fallback_theta],
            zlim=[0.741 + size, 0.741 + size],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        size = float(np.random.uniform(0.015, 0.025))
        target_gap = self._sample_target_gap()
        anchor_pose = self._sample_anchor_pose(size=size, target_gap=target_gap)
        anchor_cyl = world_to_robot(anchor_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
        target_theta_lst = [anchor_cyl[1] - target_gap, anchor_cyl[1] - 2 * target_gap]
        target_xy_lst = [
            place_point_cyl(
                [anchor_cyl[0], theta, anchor_pose.p[2]],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="array",
            )[:2]
            for theta in target_theta_lst
        ]
        block_pose_lst = [anchor_pose]
        while len(block_pose_lst) < 3:
            block_pose_lst.append(self._sample_block_pose(size=size, existing_pose_lst=block_pose_lst, avoid_xy_lst=target_xy_lst))

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
        target_z = 0.74 + self.table_z_bias
        target_quat = [0, 1, 0, 0]

        self.block2_target_pose = place_pose_cyl(
            [anchor_cyl[0], target_theta_lst[0], target_z] + target_quat,
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self.block3_target_pose = place_pose_cyl(
            [anchor_cyl[0], target_theta_lst[1], target_z] + target_quat,
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def play_once(self):
        self.last_gripper = None
        self._scan_scene_two_views(self._get_default_scan_object_list())

        block1_cyl = world_to_robot(self.block1.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        anchor_arm = ArmTag("left" if block1_cyl[1] >= 0 else "right")
        arm_tag2 = self.pick_and_place_block(self.block2, self.block2_target_pose)
        arm_tag3 = self.pick_and_place_block(self.block3, self.block3_target_pose)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": str(anchor_arm),
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }
        return self.info

    def pick_and_place_block(self, block, target_pose):
     
        block_cyl = world_to_robot(block.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if block_cyl[1] >= 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(self.back_to_origin(arm_tag=arm_tag.opposite))
        self.face_object_with_torso(block, joint_name_prefer="astribot_torso_joint_2")

        self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.01))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.face_world_point_with_torso(
            target_pose[:3],
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="free",
            )
        )
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
        same_arc = abs(c1[0] - c2[0]) < radius_eps and abs(c2[0] - c3[0]) < radius_eps
        ordered = c1[1] > c2[1] > c3[1]
        compact = abs(c1[1] - c2[1]) < theta_eps and abs(c2[1] - c3[1]) < theta_eps
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return same_arc and ordered and compact and gripper_open
