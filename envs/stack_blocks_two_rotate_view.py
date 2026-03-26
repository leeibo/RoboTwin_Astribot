from .stack_blocks_two import stack_blocks_two
from .utils import *
import numpy as np
import transforms3d as t3d


class stack_blocks_two_rotate_view(stack_blocks_two):

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
        scan_r = 0.64
        scan_z = 0.9 + self.table_z_bias
        for theta in self._get_scan_thetas_from_object_list(object_list, fallback_thetas=[1.0, -1.0]):
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

        block_half_size = 0.025
        block_pose_lst = []
        target_center = place_point_cyl(
            [0.48, 0.0, 0.75 + self.table_z_bias],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="array",
        )

        while len(block_pose_lst) < 2:
            block_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=[-1.08, 1.08],
                zlim=[0.741 + block_half_size, 0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            block_cyl = world_to_robot(block_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(block_cyl[1]) < 0.25:
                continue
            if np.linalg.norm(block_pose.p[:2] - target_center[:2]) < 0.15:
                continue
            valid = True
            for existing_pose in block_pose_lst:
                if np.sum((block_pose.p[:2] - existing_pose.p[:2])**2) < 0.01:
                    valid = False
                    break
            if not valid:
                continue
            block_pose_lst.append(deepcopy(block_pose))

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
        self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        self.block1_target_pose = place_pose_cyl(
            [0.48, 0.0, 0.75 + self.table_z_bias, 0, 1, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def pick_and_place_block(self, block: Actor):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        self.face_object_with_torso(block, joint_name_prefer="astribot_torso_joint_2")
        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(self.back_to_origin(arm_tag=arm_tag.opposite))
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        target_pose = self.block1_target_pose if self.last_actor is None else self.last_actor.get_functional_point(1)
        self.face_world_point_with_torso(target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="free",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        self.last_gripper = None
        self.last_actor = None

        arm_tag1 = self.pick_and_place_block(self.block1)
        arm_tag2 = self.pick_and_place_block(self.block2)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
        }
        return self.info
