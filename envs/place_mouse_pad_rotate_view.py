from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np


class _place_mouse_pad(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        while abs(rand_pos.p[0]) < 0.05:
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )

        self.mouse_id = np.random.choice([0, 1, 2], 1)[0]
        self.mouse = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
        )
        self.mouse.set_mass(0.05)

        if rand_pos.p[0] > 0:
            xlim = [0.05, 0.25]
        else:
            xlim = [-0.25, -0.05]
        target_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[-0.2, 0.0],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        while (np.sqrt((target_rand_pose.p[0] - rand_pos.p[0])**2 + (target_rand_pose.p[1] - rand_pos.p[1])**2) < 0.1):
            target_rand_pose = rand_pose(
                xlim=xlim,
                ylim=[-0.2, 0.0],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
        }

        color_items = list(colors.items())
        color_index = np.random.choice(len(color_items))
        self.color_name, self.color_value = color_items[color_index]

        half_size = [0.035, 0.065, 0.0005]
        self.target = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.target, padding=0.12)
        self.add_prohibit_area(self.mouse, padding=0.03)
        # Construct target pose with position from target object and identity orientation
        self.target_pose = self.target.get_pose().p.tolist() + [0, 0, 0, 1]

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("right" if self.mouse.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.mouse, arm_tag=arm_tag, pre_grasp_dis=0.1))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        # Place the mouse at the target location with alignment constraint
        self.move(
            self.place_actor(
                self.mouse,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"047_mouse/base{self.mouse_id}",
            "{B}": f"{self.color_name}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        mouse_pose = self.mouse.get_pose().p
        mouse_qpose = np.abs(self.mouse.get_pose().q)
        target_pos = self.target.get_pose().p
        eps1 = 0.015
        eps2 = 0.012

        return (np.all(abs(mouse_pose[:2] - target_pos[:2]) < np.array([eps1, eps2]))
                and (np.abs(mouse_qpose[2] * mouse_qpose[3] - 0.49) < eps1
                     or np.abs(mouse_qpose[0] * mouse_qpose[1] - 0.49) < eps1) and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())


from .utils import *
import numpy as np
import transforms3d as t3d


class place_mouse_pad_rotate_view(_place_mouse_pad):

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.mouse,
                "B": self.target,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_mouse",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "mouse_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_mouse_on_pad",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "mouse_on_pad",
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
            mouse_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
            c = world_to_robot(mouse_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(c[1]) < 0.2:
                continue
            break

        self.mouse_id = int(np.random.choice([0, 1, 2], 1)[0])
        self.mouse = create_actor(
            scene=self,
            pose=mouse_pose,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
        )
        self.mouse.set_mass(0.05)

        side = 1.0 if mouse_pose.p[0] > 0 else -1.0
        while True:
            target_rand_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_side(self, side=side),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            if np.linalg.norm(target_rand_pose.p[:2] - mouse_pose.p[:2]) < 0.1:
                continue
            break

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
        }
        color_items = list(colors.items())
        color_index = int(np.random.choice(len(color_items)))
        self.color_name, self.color_value = color_items[color_index]

        self.target = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=[0.035, 0.035, 0.0005],
            color=self.color_value,
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.target, padding=0.12)
        self.add_prohibit_area(self.mouse, padding=0.03)
        self.target_pose = self.target.get_pose().p.tolist() + [0, 0, 0, 1]
        self._configure_rotate_subtask_plan()

    def play_once(self):
        mouse_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag = ArmTag("right" if self.mouse.get_pose().p[0] > 0 else "left")
        self.enter_rotate_action_stage(1, focus_object_key=(mouse_key or "A"))
        self.move(self.grasp_actor(self.mouse, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        self.complete_rotate_subtask(1, carried_after=["A"])

        target_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(2, focus_object_key=(target_key or "B"))
        self.move(
            self.place_actor(
                self.mouse,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="free",  # Mouse orientation is part of success condition.
                pre_dis=0.07,
                dis=0.005,
            )
        )
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": f"047_mouse/base{self.mouse_id}",
            "{B}": f"{self.color_name}",
            "{a}": str(arm_tag),
        }
        return self.info
    def check_success(self):
        mouse_pose = self.mouse.get_pose().p
        target_pose = self.target.get_pose().p
        eps = np.array([0.015, 0.012])
        return np.all(abs(mouse_pose[:2] - target_pose[:2]) < eps)
