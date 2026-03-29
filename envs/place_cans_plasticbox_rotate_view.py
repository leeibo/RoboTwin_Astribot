from .place_cans_plasticbox import place_cans_plasticbox
from .utils import *
import numpy as np
import transforms3d as t3d


class place_cans_plasticbox_rotate_view(place_cans_plasticbox):

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

        box_pose = rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=rotate_theta_center(self),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.plasticbox_id = int(np.random.choice([3, 5], 1)[0])
        self.plasticbox = create_actor(
            scene=self,
            pose=box_pose,
            modelname="062_plasticbox",
            convex=True,
            model_id=self.plasticbox_id,
        )
        self.plasticbox.set_mass(0.05)

        obj1_pose = rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=rotate_theta_side(self, side=1),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.object1_id = int(np.random.choice([0, 1, 2, 3, 5, 6], 1)[0])
        self.object1 = create_actor(
            scene=self,
            pose=obj1_pose,
            modelname="071_can",
            convex=True,
            model_id=self.object1_id,
        )
        self.object1.set_mass(0.05)

        obj2_pose = rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=rotate_theta_side(self, side=-1),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.object2_id = int(np.random.choice([0, 1, 2, 3, 5, 6], 1)[0])
        self.object2 = create_actor(
            scene=self,
            pose=obj2_pose,
            modelname="071_can",
            convex=True,
            model_id=self.object2_id,
        )
        self.object2.set_mass(0.05)

        self.add_prohibit_area(self.plasticbox, padding=0.1)
        self.add_prohibit_area(self.object1, padding=0.05)
        self.add_prohibit_area(self.object2, padding=0.05)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag_left = ArmTag("left")
        arm_tag_right = ArmTag("right")
        t1 = self.plasticbox.get_functional_point(1)
        t0 = self.plasticbox.get_functional_point(0)
        self.face_object_with_torso(self.object1, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.object1, arm_tag=arm_tag_left, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.2))
        self.face_world_point_with_torso(t1[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.object1,
                arm_tag=arm_tag_left,
                target_pose=t1,
                constrain="free",
                pre_dis=0.1,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.08))
        self.move(self.back_to_origin(arm_tag=arm_tag_left))

        self.face_object_with_torso(self.object2, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.object2, arm_tag=arm_tag_right, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.2))

        self.face_world_point_with_torso(t0[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.object2,
                arm_tag=arm_tag_right,
                target_pose=t0,
                constrain="free",
                pre_dis=0.1,
            ),
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.08))

        self.info["info"] = {
            "{A}": f"071_can/base{self.object1_id}",
            "{B}": f"062_plasticbox/base{self.plasticbox_id}",
            "{C}": f"071_can/base{self.object2_id}",
        }
        return self.info
