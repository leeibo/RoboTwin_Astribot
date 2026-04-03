from .place_container_plate import place_container_plate
from .utils import *
import numpy as np
import transforms3d as t3d


class place_container_plate_rotate_view(place_container_plate):

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
            container_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            if abs(container_pose.p[0]) < 0.2:
                continue
            break

        id_list = {"002_bowl": [1, 2, 3, 5], "021_cup": [1, 2, 3, 4, 5, 6, 7]}
        self.actor_name = str(np.random.choice(["002_bowl", "021_cup"]))
        self.container_id = int(np.random.choice(id_list[self.actor_name]))
        self.container = create_actor(
            self,
            pose=container_pose,
            modelname=self.actor_name,
            model_id=self.container_id,
            convex=True,
        )

        x = 0.05 if self.container.get_pose().p[0] > 0 else -0.05
        self.plate_id = 0
        plate_pose = rand_pose_cyl(
            rlim=[0.43, 0.5],
            thetalim=rotate_theta_center(self),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        plate_pose.set_p([x, float(plate_pose.p[1]), float(plate_pose.p[2])])
        self.plate = create_actor(
            self,
            pose=plate_pose,
            modelname="003_plate",
            scale=[0.025, 0.025, 0.025],
            is_static=True,
            convex=True,
        )
        self.container.set_mass(0.1)
        self.add_prohibit_area(self.container, padding=0.1)
        self.add_prohibit_area(self.plate, padding=0.1)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        container_pose = self.container.get_pose().p
        arm_tag = ArmTag("right" if container_pose[0] > 0 else "left")

        self.face_object_with_torso(self.container, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.grasp_actor(
                self.container,
                arm_tag=arm_tag,
                contact_point_id=[0, 2][int(arm_tag == "left")],
                pre_grasp_dis=0.1,
                grasp_dis=-0.02,
                gripper_pos=-0.1,
            )
        )
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))

        target_pose = self.plate.get_functional_point(0)
        self.face_world_point_with_torso(target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.container,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.12,
                dis=0.03,
                constrain="free",
            )
        )
        self.move(self.move_by_displacement(arm_tag, z=0.08, move_axis="arm"))

        self.info["info"] = {
            "{A}": f"003_plate/base{self.plate_id}",
            "{B}": f"{self.actor_name}/base{self.container_id}",
            "{a}": str(arm_tag),
        }
        return self.info
