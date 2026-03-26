from .place_shoe import place_shoe
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d


class place_shoe_rotate_view(place_shoe):

    def setup_demo(self, is_test=False, **kwags):
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        super().setup_demo(is_test=is_test, **kwags)

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

        target_pose = place_pose_cyl(
            [0.47, 0.0, 0.74, 1, 0, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="pose",
        )
        self.target_block = create_box(
            scene=self,
            pose=target_pose,
            half_size=(0.08, 0.08, 0.0005),
            color=(0, 0, 1),
            is_static=True,
            name="box",
        )
        self.target_block.config["functional_matrix"] = [[
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ], [
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]]

        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_lim = [0.8, 1.38] if side > 0 else [-1.38, -0.8]
        while True:
            shoe_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=theta_lim,
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                rotate_rand=True,
                rotate_lim=[0, np.pi, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
            shoe_cyl = world_to_robot(shoe_pose.get_p().tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(shoe_cyl[1]) < 0.35:
                continue
            if np.sum((shoe_pose.get_p()[:2] - self.target_block.get_pose().p[:2])**2) < 0.03:
                continue
            break

        self.shoe_id = int(np.random.choice([i for i in range(10)]))
        self.shoe = create_actor(
            scene=self,
            pose=shoe_pose,
            modelname="041_shoe",
            convex=True,
            model_id=self.shoe_id,
        )

        self.add_prohibit_area(self.target_block, padding=0.08)
        self.add_prohibit_area(self.shoe, padding=0.1)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        shoe_pose = self.shoe.get_pose().p
        arm_tag = ArmTag("left" if shoe_pose[0] < 0 else "right")

        self.face_object_with_torso(self.shoe, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.shoe, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=-0.01))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        target_pose = self.target_block.get_functional_point(0)
        self.face_world_point_with_torso(target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.shoe,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.12,
                constrain="free",  # Shoe placement needs orientation alignment on the target pad.
            )
        )
        self.move(self.open_gripper(arm_tag=arm_tag))

        self.info["info"] = {"{A}": f"041_shoe/base{self.shoe_id}", "{a}": str(arm_tag)}
        return self.info
    def check_success(self):
        shoe_pose = self.shoe.get_pose().p
        target_pose = self.target_block.get_pose().p
        eps = np.array([0.05, 0.05, 0.05])
        return np.all(abs(shoe_pose - target_pose) < eps)