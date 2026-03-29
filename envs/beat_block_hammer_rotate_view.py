from .beat_block_hammer import beat_block_hammer
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d


class beat_block_hammer_rotate_view(beat_block_hammer):

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

    def _sample_block_pose(self):
        for _ in range(100):
            pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.76, 0.76],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.5],
            )
            cyl = world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if cyl[0] < 0.45 or abs(cyl[1]) < 0.12:
                continue
            if np.sum(np.square(pose.p[:2])) < 0.04:
                continue
            return pose
        return rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=rotate_theta_fixed(self, side=-1),

            zlim=[0.76, 0.76],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        hammer_pos = place_point_cyl(
            [0.44, 0.0, 0.783],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        # Keep hammer orientation in world frame (same as original task).
        # `place_pose_cyl` interprets quaternion in cylindrical local frame.
        hammer_pose = sapien.Pose(hammer_pos, [0, 0, 0.995, 0.105])
        self.hammer = create_actor(
            scene=self,
            pose=hammer_pose,
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )

        block_pose = self._sample_block_pose()
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box",
            is_static=True,
        )
        self.hammer.set_mass(0.01)

        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            block_pose.p[0] - 0.05,
            block_pose.p[1] - 0.05,
            block_pose.p[0] + 0.05,
            block_pose.p[1] + 0.05,
        ])

    def play_once(self):
        # print(self._get_default_scan_object_list())
        self._scan_scene_two_views(self._get_default_scan_object_list())

        block_pose = self.block.get_functional_point(0, "pose").p.tolist()
        block_cyl = world_to_robot(block_pose, self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if block_cyl[1] >= 0 else "right")

        self.face_object_with_torso(self.hammer, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

        target_pose = self.block.get_functional_point(1, "pose")
        self.face_world_point_with_torso(
            target_pose.p.tolist(),
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.move(
            self.place_actor(
                self.hammer,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
                constrain="free",
            )
        )

        self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
        return self.info
