from .stack_blocks_two import stack_blocks_two
from .utils import *
import numpy as np
import transforms3d as t3d


class stack_blocks_two_rotate_view(stack_blocks_two):

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.block1,
                "B": self.block2,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_green_block",
                    "instruction_idx": 1,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["B"],
                    "allow_stage2_from_memory": True,
                    "done_when": "green_block_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "stack_green_on_red",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["B"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "green_block_stacked",
                    "next_subtask_id": -1,
                },
            ]
        )

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

    @staticmethod
    def _valid_spacing(new_pose, existing_pose_lst, min_dist_sq=0.01):
        for pose in existing_pose_lst:
            if np.sum(np.square(new_pose.p[:2] - pose.p[:2])) < min_dist_sq:
                return False
        return True

    def _sample_block_pose(self, block_half_size, existing_pose_lst):
        for _ in range(120):
            pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_center(self),
                zlim=[0.741 + block_half_size, 0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            if world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)[0] < 0.4:
                continue
            if not self._valid_spacing(pose, existing_pose_lst):
                continue
            return deepcopy(pose)

        fallback_theta = float(
            np.clip(
                0.35 * rotate_theta_half(self) * (1.0 - len(existing_pose_lst)),
                -rotate_theta_half(self),
                rotate_theta_half(self),
            )
        )
        return rand_pose_cyl(
            rlim=[0.45, 0.45],
            thetalim=[fallback_theta, fallback_theta],
            zlim=[0.741 + block_half_size, 0.741 + block_half_size],
            qpos=[1, 0, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            rotate_rand=False,
        )

    def _get_block_arm_tag(self, block):
        block_cyl = world_to_robot(block.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if block_cyl[1] >= 0 else "right")

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        block_half_size = 0.025
        block_pose_lst = []
        while len(block_pose_lst) < 2:
            block_pose_lst.append(self._sample_block_pose(block_half_size, block_pose_lst))

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
        self._configure_rotate_subtask_plan()

    def pick_and_place_block(self, block: Actor):
        arm_tag = self._get_block_arm_tag(block)

        self.face_object_with_torso(block, joint_name_prefer="astribot_torso_joint_2")
        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(self.back_to_origin(arm_tag=arm_tag.opposite))
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        target_pose = self.last_actor.get_functional_point(1)
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
        block2_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.64,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        anchor_arm = self._get_block_arm_tag(self.block1)
        arm_tag2 = self._get_block_arm_tag(self.block2)

        self.enter_rotate_action_stage(1, focus_object_key=(block2_key or "B"))
        self.move(self.grasp_actor(self.block2, arm_tag=arm_tag2, pre_grasp_dis=0.09))
        self._set_carried_object_keys(["B"])
        self.move(self.move_by_displacement(arm_tag=arm_tag2, z=0.1))
        self.complete_rotate_subtask(1, carried_after=["B"])

        block1_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.64,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.block1.get_functional_point(1)
        self.enter_rotate_action_stage(2, focus_object_key=(block1_key or "A"))
        self.move(
            self.place_actor(
                self.block2,
                target_pose=target_pose,
                arm_tag=arm_tag2,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag=arm_tag2, z=0.07))
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{a}": str(anchor_arm),
            "{b}": str(arm_tag2),
        }
        return self.info
