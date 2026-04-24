from .click_bell import click_bell
from .utils import *
import numpy as np
import transforms3d as t3d


class click_bell_rotate_view(click_bell):

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.bell,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "click_bell",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "bell_pressed",
                    "next_subtask_id": -1,
                }
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
            rand_pos = rand_pose_cyl(
                rlim=[0.44, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
            cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < 0.2:
                continue
            break

        self.bell_id = int(np.random.choice([0, 1], 1)[0])
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )

        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = self.is_left_gripper_close if self.bell.get_pose().p[0] < 0 else self.is_right_gripper_close
        self._configure_rotate_subtask_plan()

    def play_once(self):
        bell_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        bell_cyl = world_to_robot(self.bell.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if bell_cyl[1] >= 0 else "right")

        self.enter_rotate_action_stage(1, focus_object_key=(bell_key or "A"))
        self.move(
            self.grasp_actor(
                self.bell,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                gripper_pos=-0.1,
                contact_point_id=0,
            )
        )

        self.move(self.move_by_displacement(arm_tag, z=-0.045))
        self.check_success()
        self.move(self.move_by_displacement(arm_tag, z=0.045))
        self.check_success()
        self.complete_rotate_subtask(1, carried_after=[])

        self.info["info"] = {"{A}": f"050_bell/base{self.bell_id}", "{a}": str(arm_tag)}
        return self.info
