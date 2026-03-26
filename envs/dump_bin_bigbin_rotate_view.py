from .dump_bin_bigbin import dump_bin_bigbin
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d


class dump_bin_bigbin_rotate_view(dump_bin_bigbin):

    def setup_demo(self, **kwags):
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        super().setup_demo(**kwags)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    def _scan_scene_two_views(self, object_list=None):
        scan_r = 0.64
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

    def _sample_deskbin_pose(self):
        for _ in range(120):
            side_thetalim = [0.55, 1.05] if np.random.rand() < 0.5 else [-1.05, -0.55]
            pose = rand_pose_cyl(
                rlim=[0.45, 0.5],
                thetalim=side_thetalim,
                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.651892, 0.651428, 0.274378, 0.274584],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 8.5, 0],
            )
            cyl = world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < 0.3:
                continue
            return pose
        return rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=[0.82, 0.82],
            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.651892, 0.651428, 0.274378, 0.274584],
            rotate_rand=False,
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        dustbin_pose = place_pose_cyl(
            [0.68, 0.95, 0.0, 0.5, 0.5, 0.5, 0.5],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="pose",
        )
        self.dustbin = create_actor(
            self,
            pose=dustbin_pose,
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )

        deskbin_pose = self._sample_deskbin_pose()
        self.deskbin_id = int(np.random.choice([0, 3, 7, 8, 9, 10], 1)[0])
        self.deskbin = create_actor(
            self,
            pose=deskbin_pose,
            modelname="063_tabletrashbin",
            model_id=self.deskbin_id,
            convex=True,
        )

        self.garbage_num = 5
        self.sphere_lst = []
        for i in range(self.garbage_num):
            sphere_pose = sapien.Pose(
                [
                    deskbin_pose.p[0] + np.random.rand() * 0.02 - 0.01,
                    deskbin_pose.p[1] + np.random.rand() * 0.02 - 0.01,
                    deskbin_pose.p[2] + 0.04 + i * 0.005,
                ],
                [1, 0, 0, 0],
            )
            sphere = create_sphere(
                self.scene,
                pose=sphere_pose,
                radius=0.008,
                color=[1, 0, 0],
                name="garbage",
            )
            self.sphere_lst.append(sphere)
            self.sphere_lst[-1].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.0001

        self.add_prohibit_area(self.deskbin, padding=0.04)
        self.add_prohibit_area(self.dustbin, padding=0.08)
        self.prohibited_area.append([-0.2, -0.2, 0.2, 0.2])

        self.middle_pose = place_pose_cyl(
            [0.52, 0.0, 0.741 + self.table_z_bias, 1, 0, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

        dustbin_center = self.dustbin.get_pose().p.tolist()
        action_lst = [
            Action(
                ArmTag("left"),
                "move",
                [
                    dustbin_center[0],
                    dustbin_center[1] - 0.05,
                    1.05,
                    -0.694654,
                    -0.178228,
                    0.165979,
                    -0.676862,
                ],
            ),
            Action(
                ArmTag("left"),
                "move",
                [
                    dustbin_center[0],
                    dustbin_center[1] - 0.05 - np.random.rand() * 0.02,
                    1.05 - np.random.rand() * 0.02,
                    -0.694654,
                    -0.178228,
                    0.165979,
                    -0.676862,
                ],
            ),
        ]
        self.pour_actions = (ArmTag("left"), action_lst)

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        deskbin_cyl = world_to_robot(self.deskbin.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        grasp_deskbin_arm_tag = ArmTag("left" if deskbin_cyl[1] >= 0 else "right")
        place_deskbin_arm_tag = ArmTag("left")

        if grasp_deskbin_arm_tag == "right":
            self.face_object_with_torso(self.deskbin, joint_name_prefer="astribot_torso_joint_2")
            self.move(
                self.grasp_actor(
                    self.deskbin,
                    arm_tag=grasp_deskbin_arm_tag,
                    pre_grasp_dis=0.08,
                    contact_point_id=3,
                )
            )
            self.move(self.move_by_displacement(grasp_deskbin_arm_tag, z=0.08, move_axis="arm"))
            self.face_world_point_with_torso(self.middle_pose[:3], joint_name_prefer="astribot_torso_joint_2")
            self.move(
                self.place_actor(
                    self.deskbin,
                    target_pose=self.middle_pose,
                    arm_tag=grasp_deskbin_arm_tag,
                    pre_dis=0.08,
                    dis=0.01,
                    constrain="free",
                )
            )
            self.move(self.move_by_displacement(grasp_deskbin_arm_tag, z=0.1, move_axis="arm"))
            self.face_object_with_torso(self.deskbin, joint_name_prefer="astribot_torso_joint_2")
            self.move(
                self.back_to_origin(grasp_deskbin_arm_tag),
                self.grasp_actor(
                    self.deskbin,
                    arm_tag=place_deskbin_arm_tag,
                    pre_grasp_dis=0.08,
                    contact_point_id=1,
                ),
            )
        else:
            self.face_object_with_torso(self.deskbin, joint_name_prefer="astribot_torso_joint_2")
            self.move(
                self.grasp_actor(
                    self.deskbin,
                    arm_tag=place_deskbin_arm_tag,
                    pre_grasp_dis=0.08,
                    contact_point_id=1,
                )
            )

        self.move(self.move_by_displacement(arm_tag=place_deskbin_arm_tag, z=0.08, move_axis="arm"))
        for _ in range(3):
            self.move(self.pour_actions)
        self.delay(6)

        self.info["info"] = {"{A}": f"063_tabletrashbin/base{self.deskbin_id}"}
        return self.info
