from .stack_bowls_three import stack_bowls_three
from .utils import *
import numpy as np
import transforms3d as t3d


class stack_bowls_three_rotate_view(stack_bowls_three):

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
        scan_z = 0.9 + self.table_z_bias
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

        bowl_pose_lst = []
        target_center = place_point_cyl(
            [0.48, 0.0, 0.76],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="array",
        )
        while len(bowl_pose_lst) < 3:
            bowl_pose = rand_pose_cyl(
                rlim=[0.45, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            bowl_cyl = world_to_robot(bowl_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(bowl_cyl[1]) < 0.3:
                continue
            if np.linalg.norm(bowl_pose.p[:2] - target_center[:2]) < 0.14:
                continue
            valid = True
            for existing_pose in bowl_pose_lst:
                if np.sum((bowl_pose.p[:2] - existing_pose.p[:2])**2) < 0.0169:
                    valid = False
                    break
            if not valid:
                continue
            bowl_pose_lst.append(deepcopy(bowl_pose))

        bowl_pose_lst = sorted(bowl_pose_lst, key=lambda x: x.p[1])

        def create_bowl(bowl_pose):
            return create_actor(self, pose=bowl_pose, modelname="002_bowl", model_id=3, convex=True)

        self.bowl1 = create_bowl(bowl_pose_lst[0])
        self.bowl2 = create_bowl(bowl_pose_lst[1])
        self.bowl3 = create_bowl(bowl_pose_lst[2])

        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.add_prohibit_area(self.bowl3, padding=0.07)
        self.bowl1_target_pose = place_point_cyl(
            [0.48, 0.0, 0.76],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="array",
        )
        self.quat_of_target_pose = [0, 0.707, 0.707, 0]

    def move_bowl(self, actor, target_pose):
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        # self.face_object_with_torso(actor, joint_name_prefer="astribot_torso_joint_2")
        if self.las_arm is None or arm_tag == self.las_arm:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    contact_point_id=[0, 2][int(arm_tag == "left")],
                    pre_grasp_dis=0.1,
                    grasp_dis=-0.01,
                    gripper_pos=-0.1,
                )
            )
        else:
            self.move(self.back_to_origin(arm_tag=arm_tag.opposite))
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    contact_point_id=[0, 2][int(arm_tag == "left")],
                    pre_grasp_dis=0.1,
                    grasp_dis=-0.01,
                    gripper_pos=-0.1,
                ),
                
            )
        self.move(self.move_by_displacement(arm_tag, z=0.1))

        place_pose = target_pose.tolist() + self.quat_of_target_pose
        self.face_world_point_with_torso(place_pose[:3], joint_name_prefer="astribot_torso_joint_2")
        self.move(
            self.place_actor(
                actor,
                target_pose=place_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0,
                constrain="align",  # Bowl stacking requires orientation alignment for stable nesting.
            )
        )
        self.move(self.move_by_displacement(arm_tag, z=0.09))
        self.las_arm = arm_tag
        return arm_tag

    def play_once(self):
        self._scan_scene_two_views(self._get_default_scan_object_list())

        self.las_arm = None
        self.move_bowl(self.bowl1, self.bowl1_target_pose)
        self.move_bowl(self.bowl2, self.bowl1.get_pose().p + [0, 0, 0.05])
        self.move_bowl(self.bowl3, self.bowl2.get_pose().p + [0, 0, 0.05])

        self.info["info"] = {"{A}": "002_bowl/base3"}
        return self.info
