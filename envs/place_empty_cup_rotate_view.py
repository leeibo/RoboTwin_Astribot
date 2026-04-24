from .place_empty_cup import place_empty_cup
from .utils import *
import numpy as np
import transforms3d as t3d


class place_empty_cup_rotate_view(place_empty_cup):

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.cup,
                "B": self.coaster,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_cup",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "cup_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_cup_on_coaster",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "cup_on_coaster",
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

        tag = int(np.random.randint(0, 2))
        cup_thetas = [
            rotate_theta_side(self, side=-1),
            rotate_theta_side(self, side=1),
        ]
        coaster_thetas = [
            rotate_theta_center(self),
            rotate_theta_center(self),
        ]
        self.cup = create_actor(
            self,
            pose=rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=cup_thetas[tag],

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            ),
            modelname="021_cup",
            convex=True,
            model_id=0,
        )
        cup_pose = self.cup.get_pose().p

        while True:
            coaster_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=coaster_thetas[tag],

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            if np.sum(np.square(cup_pose[:2] - coaster_pose.p[:2])) < 0.01:
                continue
            break
        self.coaster = create_actor(
            self,
            pose=coaster_pose,
            modelname="019_coaster",
            convex=True,
            model_id=0,
            is_static=True,
        )
        self.coaster.set_mass(0.1)
        self.add_prohibit_area(self.cup, padding=0.05)
        self.add_prohibit_area(self.coaster, padding=0.05)
        self.delay(2)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        cup_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        cup_pose = self.cup.get_pose().p
        arm_tag = ArmTag("right" if cup_pose[0] > 0 else "left")

        self.enter_rotate_action_stage(1, focus_object_key=(cup_key or "A"))
        # self.move(self.close_gripper(arm_tag, pos=0.6))
        self.move(
            self.grasp_actor(
                self.cup,
                arm_tag,
                pre_grasp_dis=0.1,
                contact_point_id=[0, 2][int(arm_tag == "left")],
                gripper_pos=-0.1,
                grasp_dis=-0.02,
            )
        )
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag, z=0.08, move_axis="arm"))
        self.complete_rotate_subtask(1, carried_after=["A"])

        coaster_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.coaster.get_functional_point(0)
        self.enter_rotate_action_stage(2, focus_object_key=(coaster_key or "B"))
        self.move(
            self.place_actor(
                self.cup,
                arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.05,
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag, z=0.05, move_axis="arm"))
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {"{A}": "021_cup/base0", "{B}": "019_coaster/base0"}
        return self.info
