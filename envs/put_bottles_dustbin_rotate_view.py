from .put_bottles_dustbin import put_bottles_dustbin
from .utils import *
import numpy as np
import transforms3d as t3d


class put_bottles_dustbin_rotate_view(put_bottles_dustbin):

    def setup_demo(self, **kwags):
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        kwags.setdefault("table_xy_bias", [0.3, 0])
        super().setup_demo(**kwags)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    def _scan_scene_two_views(self, object_list=None):
        scan_r = 0.64
        scan_z = 0.96 + self.table_z_bias
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

        pose_lst = []
        self.bottles = []
        self.bottle_id = [1, 2, 3]
        self.bottle_num = 3

        dustbin_pose = place_pose_cyl(
            [0.54, 1.2, 0.0, 0.5, 0.5, 0.5, 0.5],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="pose",
        )
        self.dustbin = create_actor(
            self.scene,
            pose=dustbin_pose,
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )

        for model_id in self.bottle_id:
            gen_lim = 100
            created = False
            for _ in range(gen_lim):
                bottle_pose = rand_pose_cyl(
                    rlim=[0.4, 0.5],
                    thetalim=[-1.05, 1.05],
                    zlim=[0.741, 0.741],
                    robot_root_xy=self.robot_root_xy,
                    robot_yaw_rad=self.robot_yaw,
                    rotate_rand=False,
                    rotate_lim=[0, 1, 0],
                    qpos=[0.707, 0.707, 0, 0],
                )
                bottle_cyl = world_to_robot(bottle_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
                if abs(bottle_cyl[1]) < 0.2:
                    continue

                valid = True
                for pose in pose_lst:
                    if np.sum(np.power(np.array(pose[:2]) - np.array(bottle_pose.p[:2]), 2)) < 0.0169:
                        valid = False
                        break
                if np.linalg.norm(bottle_pose.p[:2] - self.dustbin.get_pose().p[:2]) < 0.22:
                    valid = False
                if not valid:
                    continue
                pose_lst.append(bottle_pose.p[:2])
                bottle = create_actor(
                    self,
                    bottle_pose,
                    modelname="114_bottle",
                    convex=True,
                    model_id=model_id,
                )
                self.bottles.append(bottle)
                self.add_prohibit_area(bottle, padding=0.1)
                created = True
                break
            if not created:
                raise RuntimeError("Failed to sample non-overlapping bottle poses on fan table.")

        self.delay(2)
        self.right_middle_pose = place_pose_cyl(
            [0.48, -0.35, 0.88, 0, 1, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        bottle_lst = sorted(self.bottles, key=lambda x: [x.get_pose().p[0] > 0, x.get_pose().p[1]])
        dustbin_center = self.dustbin.get_pose().p.tolist()
        delta_dis = 0.06

        for bottle in bottle_lst:
            arm_tag = ArmTag("left" if bottle.get_pose().p[0] < 0 else "right")

            if arm_tag == "left":
                self.face_object_with_torso(bottle, joint_name_prefer="astribot_torso_joint_2")
                self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
                self.move(self.move_by_displacement(arm_tag, z=0.1))
            else:
                self.face_object_with_torso(bottle, joint_name_prefer="astribot_torso_joint_2")
                right_action = self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1)
                right_action[1][0].target_pose[2] += delta_dis
                right_action[1][1].target_pose[2] += delta_dis
                self.move(right_action, self.back_to_origin("left"))
                self.move(self.move_by_displacement(arm_tag, z=0.1))

                self.face_world_point_with_torso(
                    self.right_middle_pose[:3],
                    joint_name_prefer="astribot_torso_joint_2",
                )
                self.move(
                    self.place_actor(
                        bottle,
                        target_pose=self.right_middle_pose,
                        arm_tag=arm_tag,
                        functional_point_id=0,
                        pre_dis=0.0,
                        dis=0.0,
                        is_open=False,
                        constrain="free",
                    )
                )

                self.face_object_with_torso(bottle, joint_name_prefer="astribot_torso_joint_2")
                left_action = self.grasp_actor(bottle, arm_tag="left", pre_grasp_dis=0.1)
                left_action[1][0].target_pose[2] -= delta_dis
                left_action[1][1].target_pose[2] -= delta_dis
                self.move(left_action)
                self.move(self.open_gripper(ArmTag("right")))
                self.move(self.back_to_origin("right"))

            drop_pose = [dustbin_center[0], dustbin_center[1], 0.93, 0.65, -0.25, 0.25, 0.65]
            self.face_world_point_with_torso(drop_pose[:3], joint_name_prefer="astribot_torso_joint_2")
            self.move(self.move_to_pose(arm_tag="left", target_pose=drop_pose))
            drop_pose[2] = 0.86
            self.face_world_point_with_torso(drop_pose[:3], joint_name_prefer="astribot_torso_joint_2")
            self.move(self.move_to_pose(arm_tag="left", target_pose=drop_pose))
            self.move(self.open_gripper("left"))
            self.move(self.move_by_displacement("left", z=0.1))
            self.move(self.back_to_origin("left"))

        self.info["info"] = {
            "{A}": f"114_bottle/base{self.bottle_id[0]}",
            "{B}": f"114_bottle/base{self.bottle_id[1]}",
            "{C}": f"114_bottle/base{self.bottle_id[2]}",
            "{D}": "011_dustbin/base0",
        }
        return self.info

    def stage_reward(self):
        target_pose = self.dustbin.get_pose().p[:2]
        eps = np.array([0.221, 0.325])
        reward = 0
        reward_step = 1 / 3
        for i in range(self.bottle_num):
            bottle_pose = self.bottles[i].get_pose().p
            if np.all(np.abs(bottle_pose[:2] - target_pose) < eps) and 0.2 < bottle_pose[2] < 0.7:
                reward += reward_step
        return reward

    def check_success(self):
        target_pose = self.dustbin.get_pose().p[:2]
        eps = np.array([0.221, 0.325])
        for i in range(self.bottle_num):
            bottle_pose = self.bottles[i].get_pose().p
            if np.all(np.abs(bottle_pose[:2] - target_pose) < eps) and 0.2 < bottle_pose[2] < 0.7:
                continue
            return False
        return True
