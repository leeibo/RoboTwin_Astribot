from .place_bread_basket import place_bread_basket
from .utils import *
import numpy as np
import transforms3d as t3d


class place_bread_basket_rotate_view(place_bread_basket):

    def setup_demo(self, **kwargs):
        kwargs.setdefault("table_shape", "fan")
        kwargs.setdefault("fan_center_on_robot", True)
        kwargs.setdefault("fan_outer_radius", 0.9)
        kwargs.setdefault("fan_inner_radius", 0.3)
        kwargs.setdefault("fan_angle_deg", 220)
        kwargs.setdefault("fan_center_deg", 90)
        kwargs = init_rotate_theta_bounds(self, kwargs)
        super().setup_demo(**kwargs)

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

        basket_pose = place_pose_cyl(
            [0.45,0, 0.741, 0.5, 0.5, 0.5, 0.5],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="pose",
        )
        self.basket_id = int(np.random.choice([0, 1, 2, 3, 4]))
        self.breadbasket = create_actor(
            scene=self,
            pose=basket_pose,
            modelname="076_breadbasket",
            convex=True,
            model_id=self.basket_id,
        )

        self.bread = []
        self.bread_id = []
        bread_sides = [1.0, -1.0]
        for side in bread_sides:
            bread_pose = rand_pose_cyl(
                rlim=[0.5, 0.5],
                thetalim=rotate_theta_side(self, side=side),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
            bid = int(np.random.choice([0, 1, 3, 5, 6]))
            self.bread_id.append(bid)
            self.bread.append(
                create_actor(
                    scene=self,
                    pose=bread_pose,
                    modelname="075_bread",
                    convex=True,
                    model_id=bid,
                )
            )

        for i in range(len(self.bread)):
            self.add_prohibit_area(self.bread[i], padding=0.03)
        self.add_prohibit_area(self.breadbasket, padding=0.05)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())
        last_arm_tag = None

        basket_pose = self.breadbasket.get_functional_point(0)
        arm_info = []
        for bread in sorted(self.bread, key=lambda a: a.get_pose().p[0]):
            arm_tag = ArmTag("right" if bread.get_pose().p[0] > 0 else "left")
            arm_info.append(str(arm_tag))
            if last_arm_tag is not None and last_arm_tag != arm_tag:
                self.move(self.open_gripper(arm_tag=last_arm_tag))
            # self.face_object_with_torso(bread, joint_name_prefer="astribot_torso_joint_2")
            self.move(self.grasp_actor(bread, arm_tag=arm_tag, pre_grasp_dis=0.12))
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

            self.face_world_point_with_torso(basket_pose[:3], joint_name_prefer="astribot_torso_joint_2")
            self.move(
                self.place_actor(
                    bread,
                    arm_tag=arm_tag,
                    target_pose=basket_pose,
                    constrain="free",
                    pre_dis=0.12,
                )
            )
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.04, move_axis="arm"))
            self.move(self.back_to_origin(arm_tag=arm_tag))
            last_arm_tag = arm_tag

        self.info["info"] = {
            "{A}": f"076_breadbasket/base{self.basket_id}",
            "{B}": f"075_bread/base{self.bread_id[0]}",
            "{a}": "dual" if len(set(arm_info)) > 1 else arm_info[0],
        }
        if len(self.bread) >= 2:
            self.info["info"]["{C}"] = f"075_bread/base{self.bread_id[1]}"
        return self.info
