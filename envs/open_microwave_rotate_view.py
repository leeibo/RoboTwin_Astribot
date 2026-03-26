from .open_microwave import open_microwave
from .utils import *
import numpy as np
import transforms3d as t3d


class open_microwave_rotate_view(open_microwave):

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
        self.model_name = "044_microwave"
        self.model_id = int(np.random.randint(0, 2))
        microwave_pose = rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=[0.5, 1.0],
            zlim=[0.8, 0.8],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.707, 0, 0, 0.707],
            rotate_rand=False,
        )
        self.microwave = create_sapien_urdf_obj(
            scene=self,
            pose=microwave_pose,
            modelname=self.model_name,
            modelid=self.model_id,
            fix_root_link=True,
        )
        self.microwave.set_mass(0.01)
        self.microwave.set_properties(0.0, 0.0)

        self.add_prohibit_area(self.microwave)
        self.prohibited_area.append([-0.25, -0.25, 0.25, 0.1])

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        arm_tag = ArmTag("left")
        self.face_object_with_torso(self.microwave, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.microwave, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=0))

        start_qpos = self.microwave.get_qpos()[0]
        for _ in range(50):
            self.face_object_with_torso(self.microwave, joint_name_prefer="astribot_torso_joint_2")
            self.move(
                self.grasp_actor(
                    self.microwave,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.0,
                    grasp_dis=0.0,
                    contact_point_id=4,
                )
            )
            new_qpos = self.microwave.get_qpos()[0]
            if new_qpos - start_qpos <= 0.001:
                break
            start_qpos = new_qpos
            if not self.plan_success:
                break
            if self.check_success(target=0.7):
                break

        if not self.check_success(target=0.7):
            self.plan_success = True
            self.move(self.open_gripper(arm_tag=arm_tag))
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.05, z=0.05))

            self.face_object_with_torso(self.microwave, joint_name_prefer="astribot_torso_joint_2")
            self.move(self.grasp_actor(self.microwave, arm_tag=arm_tag, contact_point_id=1))
            self.move(
                self.grasp_actor(
                    self.microwave,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.02,
                    contact_point_id=1,
                )
            )

            start_qpos = self.microwave.get_qpos()[0]
            for _ in range(30):
                self.face_object_with_torso(self.microwave, joint_name_prefer="astribot_torso_joint_2")
                self.move(
                    self.grasp_actor(
                        self.microwave,
                        arm_tag=arm_tag,
                        pre_grasp_dis=0.0,
                        grasp_dis=0.0,
                        contact_point_id=2,
                    )
                )
                new_qpos = self.microwave.get_qpos()[0]
                if new_qpos - start_qpos <= 0.001:
                    break
                start_qpos = new_qpos
                if not self.plan_success:
                    break
                if self.check_success(target=0.7):
                    break

        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info
