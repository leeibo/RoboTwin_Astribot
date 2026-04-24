from .adjust_bottle_test_no_rotate_and_head1 import adjust_bottle_test_no_rotate_and_head
from .utils import *
import numpy as np
import sapien


class adjust_bottle_head(adjust_bottle_test_no_rotate_and_head):

    def load_actors(self):
        self.model_id = 13
        self.initial_bottle_z = None
        self.arm_tag = None

        self.bottle = rand_create_actor(
            scene=self,
            modelname="001_bottle",
            # Keep the bottle on the cabinet top while shifting it slightly
            # away from the exact center line for an easier one-arm grasp.
            xlim=[0.05, 0.05],
            ylim=[0, 0],
            zlim=[1.212, 1.213],
            qpos=[0, 0, 1, 0],
            rotate_rand=False,
            convex=True,
            model_id=self.model_id,
        )
        self.bottle.set_mass(0.01)

        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            pose=sapien.Pose([0.07, 0.1, 0.741], [0.7071068, 0, 0, 0.7071068]),
            modelname="036_cabinet",
            modelid=46653,
            fix_root_link=True,
        )

    def _get_head_joint2_index(self, head_joint2_name="astribot_head_joint_2"):
        for i, joint in enumerate(getattr(self.robot, "head_joints", [])):
            if joint is not None and joint.get_name() == str(head_joint2_name):
                return i
        head_now = self._get_head_joint_state_now()
        if head_now is None or head_now.shape[0] == 0:
            return None
        return min(1, head_now.shape[0] - 1)

    def _get_bottle_head_projection(self):
        camera_pose = self._get_scan_camera_pose("camera_head")
        camera_spec = self._get_scan_camera_runtime_spec("camera_head")
        if camera_pose is None or camera_spec is None:
            return None

        try:
            (u_norm, v_norm), debug = project_object_to_image_uv(
                obj=self.bottle,
                camera_pose=camera_pose,
                image_w=int(camera_spec["w"]),
                image_h=int(camera_spec["h"]),
                fovy_rad=float(camera_spec["fovy_rad"]),
                mode="aabb",
                far=camera_spec.get("far", None),
                ret_debug=True,
            )
        except Exception:
            return None

        return {
            "inside": bool(debug.get("inside", False)),
            "u_norm": None if not np.isfinite(u_norm) else float(u_norm),
            "v_norm": None if not np.isfinite(v_norm) else float(v_norm),
            "world_point": np.array(debug.get("world_point", self._resolve_object_world_point(self.bottle)),
                                    dtype=np.float64).reshape(-1).tolist(),
        }

    def _coarse_search_bottle_with_head_joint2(
        self,
        head_joint2_name="astribot_head_joint_2",
        step_rad=0.15,
        settle_steps=12,
    ):
        proj = self._get_bottle_head_projection()
        if proj is not None and proj["inside"]:
            return proj

        head_now = self._get_head_joint_state_now()
        head_joint2_idx = self._get_head_joint2_index(head_joint2_name=head_joint2_name)
        if head_now is None or head_joint2_idx is None:
            return None

        lower, upper = -1.22, 1.22
        try:
            limits = self.robot.head_joints[head_joint2_idx].get_limits()
            if limits is not None and len(limits) > 0:
                lower = float(limits[0][0])
                upper = float(limits[0][1])
        except Exception:
            pass

        current_joint2 = float(np.clip(head_now[head_joint2_idx], lower, upper))
        target_values = list(np.arange(current_joint2 - abs(step_rad), lower - 1e-9, -abs(step_rad), dtype=np.float64))
        if len(target_values) == 0 or abs(target_values[-1] - lower) > 1e-6:
            target_values.append(lower)

        for target_joint2 in target_values:
            head_target = np.array(head_now, dtype=np.float64)
            head_target[head_joint2_idx] = float(np.clip(target_joint2, lower, upper))
            self.move_head_to(head_target, settle_steps=settle_steps)
            proj = self._get_bottle_head_projection()
            if proj is not None and proj["inside"]:
                return proj
        return None

    def _precisely_focus_bottle_with_head_joint2(
        self,
        head_joint2_name="astribot_head_joint_2",
        v_tol=0.08,
        settle_steps=12,
        max_refine_iter=2,
    ):
        head_joint2_idx = self._get_head_joint2_index(head_joint2_name=head_joint2_name)
        if head_joint2_idx is None:
            return False

        for _ in range(max(1, int(max_refine_iter))):
            proj = self._get_bottle_head_projection()
            if proj is None or (not proj["inside"]):
                return False
            if proj["v_norm"] is not None and abs(float(proj["v_norm"]) - 0.5) <= float(v_tol):
                return True

            solve_res = self.solve_head_lookat_joint_target(world_point=proj["world_point"])
            if solve_res is None:
                return False

            head_now = self._get_head_joint_state_now()
            if head_now is None:
                return False

            solved_head_target = np.array(solve_res["target"], dtype=np.float64).reshape(-1)
            if solved_head_target.shape[0] <= head_joint2_idx:
                return False

            head_target = np.array(head_now, dtype=np.float64)
            head_target[head_joint2_idx] = solved_head_target[head_joint2_idx]
            self.move_head_to(head_target, settle_steps=settle_steps)

        proj = self._get_bottle_head_projection()
        return bool(proj is not None and proj["inside"])

    def play_once(self):
        # Stage 1: keep the body fixed and only raise head joint2 until the bottle enters view.
        proj = self._coarse_search_bottle_with_head_joint2()
        if proj is None or (not proj["inside"]):
            self.plan_success = False
            self.arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] >= 0 else "left")
            self.info["info"] = {
                "{A}": f"001_bottle/base{self.model_id}",
                "{a}": str(self.arm_tag),
            }
            return self.info

        # Stage 2: once visible, refine the vertical alignment with head joint2 only.
        self._precisely_focus_bottle_with_head_joint2()

        # Stage 3: run the task-specific grasp/lift parameters here instead of
        # relying on the no-rotate baseline implementation.
        self.arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] >= 0 else "left")
        self.initial_bottle_z = float(self.bottle.get_pose().p[2])

        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.08,
                grasp_dis=-0.01,
                gripper_pos=0.2,
            )
        )
        self.delay(3)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.04))
        self.delay(2)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.04))

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if self.initial_bottle_z is None:
            return False
        bottle_z = float(self.bottle.get_pose().p[2])
        return bottle_z > (self.initial_bottle_z + 0.02)
