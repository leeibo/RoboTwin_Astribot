from .hanging_mug import hanging_mug
from .utils import *
import numpy as np
import transforms3d as t3d
from ._GLOBAL_CONFIGS import *

class hanging_mug_rotate_view(hanging_mug):

    def setup_demo(self, is_test=False, **kwags):
        kwags.setdefault("table_shape", "fan")
        kwags.setdefault("fan_center_on_robot", True)
        kwags.setdefault("fan_outer_radius", 0.9)
        kwags.setdefault("fan_inner_radius", 0.3)
        kwags.setdefault("fan_angle_deg", 220)
        kwags.setdefault("fan_center_deg", 90)
        kwags = init_rotate_theta_bounds(self, kwags)
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

        self.mug_id = int(np.random.choice([i for i in range(10)]))
        mug_pose = rand_pose_cyl(
            rlim=[0.35, 0.45],
            thetalim=rotate_theta_side(self, side=1),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 1.57, 0],
        )
        self.mug = create_actor(
            self,
            pose=mug_pose,
            modelname="039_mug",
            convex=True,
            model_id=self.mug_id,
        )

        rack_pose = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=rotate_theta_side(self, side=-1),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[-0.22, -0.22, 0.67, 0.67],
            rotate_rand=True,
            rotate_lim=[0, 0.2, 0],
            # quat_frame="cyl_legacy",
        )
        self.mug.set_mass(0.1)
        self.rack = create_actor(self, pose=rack_pose, modelname="040_rack", is_static=True, convex=True)

        self.add_prohibit_area(self.mug, padding=0.1)
        self.add_prohibit_area(self.rack, padding=0.1)
        self.middle_pos = place_pose_cyl(
            [0.5, 0.0, 0.75 + self.table_z_bias, 1, 0, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        grasp_arm_tag = ArmTag("left")
        hang_arm_tag = ArmTag("right")

        # self.face_object_with_torso(self.mug, joint_name_prefer="astribot_torso_joint_2")
        self.move(self.grasp_actor(self.mug, arm_tag=grasp_arm_tag, pre_grasp_dis=0.1,grasp_dis=-0.02,gripper_pos=-0.1))
        self.move(self.move_by_displacement(arm_tag=grasp_arm_tag, z=0.08))

        self.face_world_point_with_torso(self.middle_pos[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=grasp_arm_tag,
                target_pose=self.middle_pos,
                pre_dis=0.05,
                dis=0.0,
                constrain="free",
            )
        )
        self.move(self.move_by_displacement(arm_tag=grasp_arm_tag, z=0.1))

        # self.face_object_with_torso(self.mug, joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.back_to_origin(grasp_arm_tag),
            self.grasp_actor(self.mug, arm_tag=hang_arm_tag, pre_grasp_dis=0.05),
        )
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, quat=GRASP_DIRECTION_DIC["front"]))

        target_pose = self.rack.get_functional_point(0)
        self.face_world_point_with_torso(target_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=hang_arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                constrain="align",  # Hanging needs hook-ring orientation consistency.
                pre_dis=0.05,
                dis=-0.05,
                pre_dis_axis="fp",
            )
        )
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, move_axis="arm"))
        self.info["info"] = {"{A}": f"039_mug/base{self.mug_id}", "{B}": "040_rack/base0"}
        return self.info
