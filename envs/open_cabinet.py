from ._base_task import Base_Task
from .utils import *
import sapien
import transforms3d as t3d


class open_cabinet(Base_Task):
    DRAWER_OPEN_SUCCESS_DIS = 0.08
    DRAWER_PULL_TOTAL_DIS = 0.20
    DRAWER_PULL_STEPS = 2
    CABINET_PRE_GRASP_DIS = 0.05
    CABINET_GRASP_DIS = 0.01
    CABINET_GRIPPER_POS = -0.02
    CABINET_ROTATE_LIM_ABS = (0.25, 1.0)

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags, table_static=False)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    def _get_cabinet_arm_tag(self):
        cabinet_cyl = world_to_robot(self.cabinet.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if float(cabinet_cyl[1]) >= 0.0 else "right")

    def _get_drawer_pull_step_xy(self):
        cabinet_xy = np.array(self.cabinet.get_pose().p[:2], dtype=np.float64)
        robot_xy = np.array(self.robot_root_xy, dtype=np.float64)
        direction = robot_xy - cabinet_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([0.0, -1.0], dtype=np.float64)
        else:
            direction = direction / norm
        step_dis = float(self.DRAWER_PULL_TOTAL_DIS) / float(max(int(self.DRAWER_PULL_STEPS), 1))
        return (direction * step_dis).tolist()

    def _get_drawer_world_point(self):
        return np.array(self.cabinet.get_functional_point(0)[:3], dtype=np.float64)

    @staticmethod
    def _build_mirrored_rotate_lim(arm_tag, positive_rotate_lim):
        min_abs, max_abs = sorted(abs(float(v)) for v in positive_rotate_lim)
        if ArmTag(arm_tag) == "left":
            return (min_abs, max_abs)
        return (-max_abs, -min_abs)

    def _get_cabinet_arm_rotate_lim(self, arm_tag):
        return self._build_mirrored_rotate_lim(arm_tag, self.CABINET_ROTATE_LIM_ABS)

    def _run_with_arm_rotate_lim(self, arm_tag, rotate_lim, action_fn):
        rotate_attr = "left_rotate_lim" if ArmTag(arm_tag) == "left" else "right_rotate_lim"
        original_rotate_lim = list(getattr(self.robot, rotate_attr))
        try:
            setattr(self.robot, rotate_attr, list(rotate_lim))
            return action_fn()
        finally:
            setattr(self.robot, rotate_attr, original_rotate_lim)

    def _grasp_cabinet_with_tilted_pose(self):
        cabinet_actions = self._run_with_arm_rotate_lim(
            self.arm_tag,
            self._get_cabinet_arm_rotate_lim(self.arm_tag),
            lambda: self.grasp_actor(
                self.cabinet,
                arm_tag=self.arm_tag,
                pre_grasp_dis=self.CABINET_PRE_GRASP_DIS,
                grasp_dis=self.CABINET_GRASP_DIS,
                gripper_pos=self.CABINET_GRIPPER_POS,
            ),
        )
        self.move(cabinet_actions)
        return bool(self.plan_success)

    def choose_best_pose(self, res_pose, center_pose, arm_tag=None):
        if not self.plan_success:
            return [-1, -1, -1, -1, -1, -1, -1]
        if arm_tag == "left":
            plan_multi_pose = self.robot.left_plan_multi_path
        elif arm_tag == "right":
            plan_multi_pose = self.robot.right_plan_multi_path
        else:
            return None

        target_lst = self.robot.create_target_pose_list(res_pose, center_pose, arm_tag)
        traj_lst = plan_multi_pose(target_lst)
        best_pose = None
        best_step = None
        for i, pose in enumerate(target_lst):
            if traj_lst["status"][i] != "Success":
                continue
            step_count = len(traj_lst["position"][i])
            if best_step is None or step_count < best_step:
                best_pose = pose
                best_step = step_count
        return best_pose

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.model_name = "036_cabinet"
        self.model_id = 46653
        self.cabinet = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[0.155, 0.155],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 1],
            fix_root_link=True,
        )
        self.initial_drawer_world_point = self._get_drawer_world_point()
        self.add_prohibit_area(self.cabinet, padding=0.01)
        self.prohibited_area.append([-0.15, -0.3, 0.15, 0.3])

    def play_once(self):
        self.arm_tag = self._get_cabinet_arm_tag()
        self.initial_drawer_world_point = self._get_drawer_world_point()

        if not self._grasp_cabinet_with_tilted_pose():
            return self.info

        step_xy = self._get_drawer_pull_step_xy()
        for _ in range(max(int(self.DRAWER_PULL_STEPS), 1)):
            self.move(
                self.move_by_displacement(
                    arm_tag=self.arm_tag,
                    x=float(step_xy[0]),
                    y=float(step_xy[1]),
                )
            )
            if not self.plan_success:
                break

        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if not hasattr(self, "initial_drawer_world_point"):
            return False
        current_drawer_world_point = self._get_drawer_world_point()
        open_dis = float(
            np.linalg.norm(
                current_drawer_world_point[:2] - np.array(self.initial_drawer_world_point[:2], dtype=np.float64)
            )
        )
        return open_dis > float(self.DRAWER_OPEN_SUCCESS_DIS)
