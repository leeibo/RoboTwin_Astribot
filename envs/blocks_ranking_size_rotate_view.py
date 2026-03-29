from .blocks_ranking_size import blocks_ranking_size
from .utils import *
import numpy as np
import transforms3d as t3d


class blocks_ranking_size_rotate_view(blocks_ranking_size):

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
    def _valid_spacing(new_pose, existing_pose_lst, min_dist_sq=0.014):
        for pose in existing_pose_lst:
            if np.sum(np.square(new_pose.p[:2] - pose.p[:2])) < min_dist_sq:
                return False
        return True

    def _sample_block_pose(self, size, existing_pose_lst):
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
            if cyl[0] < 0.4 or abs(cyl[1]) < 0.12:
                continue
            if not self._valid_spacing(pose, existing_pose_lst):
                continue
            return pose
        return rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=[0.6 * rotate_theta_half(self) * (1.0 - len(existing_pose_lst))] * 2,

            zlim=[0.741 + size, 0.741 + size],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

    def _is_already_ranked(self, block_pose_lst):
        cyl = [world_to_robot(p.p.tolist(), self.robot_root_xy, self.robot_yaw) for p in block_pose_lst]
        theta = [c[1] for c in cyl]
        return theta[0] > theta[1] > theta[2]

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        color_lst = [(np.random.random(), np.random.random(), np.random.random()) for _ in range(3)]
        halfsize_lst = [
            float(np.random.uniform(0.03, 0.033)),
            float(np.random.uniform(0.024, 0.027)),
            float(np.random.uniform(0.018, 0.021)),
        ]

        while True:
            block_pose_lst = []
            for i in range(3):
                block_pose_lst.append(self._sample_block_pose(size=halfsize_lst[i], existing_pose_lst=block_pose_lst))
            if self._is_already_ranked(block_pose_lst):
                continue
            break

        def create_block(block_pose, size, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(size, size, size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], halfsize_lst[0], color_lst[0])
        self.block2 = create_block(block_pose_lst[1], halfsize_lst[1], color_lst[1])
        self.block3 = create_block(block_pose_lst[2], halfsize_lst[2], color_lst[2])

        self.add_prohibit_area(self.block1, padding=0.1)
        self.add_prohibit_area(self.block2, padding=0.1)
        self.add_prohibit_area(self.block3, padding=0.1)
        self.prohibited_area.append([-0.2, -0.2, 0.2, -0.08])

        target_r = float(np.random.uniform(0.4, 0.5))
        target_theta_mid = float(np.random.uniform(-0.03, 0.03))
        target_gap = float(np.random.uniform(0.16, 0.21))
        target_z = 0.74 + self.table_z_bias
        target_quat = [0, 1, 0, 0]

        self.block1_target_pose = place_pose_cyl(
            [target_r, target_theta_mid + target_gap, target_z] + target_quat,
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self.block2_target_pose = place_pose_cyl(
            [target_r, target_theta_mid, target_z] + target_quat,
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self.block3_target_pose = place_pose_cyl(
            [target_r, target_theta_mid - target_gap, target_z] + target_quat,
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def play_once(self):
        self.last_gripper = None
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag3 = self.pick_and_place_block(self.block3, self.block3_target_pose)
        arm_tag2 = self.pick_and_place_block(self.block2, self.block2_target_pose)
        arm_tag1 = self.pick_and_place_block(self.block1, self.block1_target_pose)

        self.info["info"] = {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }
        return self.info

    def pick_and_place_block(self, block, target_pose):
        self.face_object_with_torso(block, joint_name_prefer="astribot_torso_joint_2")

        block_cyl = world_to_robot(block.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if block_cyl[1] >= 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(self.back_to_origin(arm_tag=arm_tag.opposite))
        self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12))

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
