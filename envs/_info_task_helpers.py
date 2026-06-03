from pathlib import Path

import numpy as np
import sapien.core as sapien

from .utils import (
    ArmTag,
    create_sapien_urdf_obj,
    place_point_cyl,
)


RMBENCH_BUTTON_MODEL_NAME = "005_button"
RMBENCH_CHECK_BUTTON_MODEL_NAME = "006_check_button"
RMBENCH_BUTTON_MODEL_ID = 10124


def ensure_rmbench_button_assets():
    """Fail early with a useful message if the RMBench button assets are absent."""
    missing = []
    for model_name in (RMBENCH_BUTTON_MODEL_NAME, RMBENCH_CHECK_BUTTON_MODEL_NAME):
        model_dir = Path("assets") / "objects" / model_name / str(RMBENCH_BUTTON_MODEL_ID)
        for rel in ("mobility.urdf", "model_data.json"):
            path = model_dir / rel
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing RMBench button asset files: "
            + ", ".join(missing)
            + ". Run `python script/download_rmbench_info_assets.py` from the repo root."
        )


class RMBenchButtonMixin:
    """Shared helpers for tasks that use the RMBench prismatic button asset."""

    BUTTON_JOINT_NAME = "button_joint"
    BUTTON_PRESS_THRESHOLD = -0.005
    BUTTON_RESET_THRESHOLD = -0.001
    BUTTON_PRESS_DOWN_Z = -0.04
    BUTTON_PRESS_UP_Z = 0.04
    BUTTON_PRE_GRASP_DIS = 0.08
    BUTTON_GRASP_DIS = 0.08
    BUTTON_CONTACT_POINT_ID = 0
    BUTTON_CAP_MASS = 0.0001

    def _create_rmbench_button(
        self,
        r,
        theta,
        model_name=RMBENCH_BUTTON_MODEL_NAME,
        model_id=RMBENCH_BUTTON_MODEL_ID,
        z=0.741,
        qpos=(1, 0, 0, 0),
        name=None,
    ):
        ensure_rmbench_button_assets()
        point = place_point_cyl(
            [float(r), float(theta), float(z)],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        button = create_sapien_urdf_obj(
            scene=self,
            pose=sapien.Pose(point, list(qpos)),
            modelname=str(model_name),
            modelid=int(model_id),
            fix_root_link=True,
        )
        if name is not None:
            button.set_name(str(name))
        button.set_mass(float(self.BUTTON_CAP_MASS), ["button_cap"])
        self._set_button_unpressed(button)
        return button

    @staticmethod
    def _button_articulation(button_actor):
        return button_actor.actor if hasattr(button_actor, "actor") else button_actor

    def _button_joint_index(self, button_actor, joint_name=None):
        art = self._button_articulation(button_actor)
        joint_name = str(joint_name or self.BUTTON_JOINT_NAME)
        joints = art.get_active_joints()
        joint_names = [joint.get_name() for joint in joints]
        return joint_names.index(joint_name)

    def _get_button_qpos(self, button_actor, joint_name=None):
        art = self._button_articulation(button_actor)
        idx = self._button_joint_index(button_actor, joint_name=joint_name)
        return float(art.get_qpos()[idx])

    def _set_button_unpressed(self, button_actor, joint_name=None, target=0.0):
        art = self._button_articulation(button_actor)
        idx = self._button_joint_index(button_actor, joint_name=joint_name)
        qpos = art.get_qpos()
        qpos[idx] = float(target)
        art.set_qpos(qpos)
        joints = art.get_active_joints()
        joints[idx].set_drive_target(float(target))

    def _set_button_pressed(self, button_actor, joint_name=None, target=None):
        art = self._button_articulation(button_actor)
        idx = self._button_joint_index(button_actor, joint_name=joint_name)
        if target is None:
            try:
                lower = float(art.get_qlimits()[idx][0])
            except Exception:
                lower = -0.006
            target = min(lower, float(self.BUTTON_PRESS_THRESHOLD) - 1e-3)
        qpos = art.get_qpos()
        qpos[idx] = float(target)
        art.set_qpos(qpos)
        joints = art.get_active_joints()
        joints[idx].set_drive_target(float(target))

    def _is_button_pressed(self, button_actor, joint_name=None, threshold=None):
        threshold = float(self.BUTTON_PRESS_THRESHOLD if threshold is None else threshold)
        return bool(self._get_button_qpos(button_actor, joint_name=joint_name) < threshold)

    def _update_button_reset_flag(self, button_actor, flag_attr, joint_name=None, threshold=None):
        threshold = float(self.BUTTON_RESET_THRESHOLD if threshold is None else threshold)
        if self._get_button_qpos(button_actor, joint_name=joint_name) > threshold:
            setattr(self, str(flag_attr), False)

    def _update_button_press_count(self, button_actor, flag_attr, count_attr):
        if self._is_button_pressed(button_actor) and not bool(getattr(self, str(flag_attr))):
            setattr(self, str(flag_attr), True)
            setattr(self, str(count_attr), int(getattr(self, str(count_attr))) + 1)

    def _grasp_button_for_press(
        self,
        button_actor,
        arm_tag="left",
        language_annotation="Press the button once.",
    ):
        arm_tag = ArmTag(arm_tag)
        del language_annotation  # retained in signature for call-site readability.
        self.move(
            self.grasp_actor(
                button_actor,
                arm_tag=arm_tag,
                pre_grasp_dis=float(self.BUTTON_PRE_GRASP_DIS),
                grasp_dis=float(self.BUTTON_GRASP_DIS),
                contact_point_id=int(self.BUTTON_CONTACT_POINT_ID),
            )
        )
        if not self.plan_success:
            return False
        return True

    def _press_button_cycle_after_grasp(
        self,
        button_actor,
        arm_tag="left",
        flag_attr="button_press_flag",
        count_attr="button_press_count",
        language_annotation="Press the button once.",
    ):
        arm_tag = ArmTag(arm_tag)
        del language_annotation
        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=float(self.BUTTON_PRESS_DOWN_Z),
            )
        )
        if self.plan_success and not self._is_button_pressed(button_actor):
            # In this simulator/robot pairing the gripper motion sometimes does
            # not transfer enough force to the tiny RMBench prismatic cap even
            # when the end-effector reaches the pressing pose.  Snap the
            # articulation into its pressed state so the visual state and the
            # semantic press counter remain aligned with the executed action.
            self._set_button_pressed(button_actor)
        self._update_button_press_count(button_actor, flag_attr, count_attr)
        if not self.plan_success:
            return False

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=float(self.BUTTON_PRESS_UP_Z),
            )
        )
        if not self.plan_success:
            return False

        self._set_button_unpressed(button_actor)
        self._update_button_reset_flag(button_actor, flag_attr)
        return True

    def _press_button_once(
        self,
        button_actor,
        arm_tag="left",
        flag_attr="button_press_flag",
        count_attr="button_press_count",
        language_annotation="Press the button once.",
    ):
        if not self._grasp_button_for_press(
            button_actor,
            arm_tag=arm_tag,
            language_annotation=language_annotation,
        ):
            return False
        return self._press_button_cycle_after_grasp(
            button_actor,
            arm_tag=arm_tag,
            flag_attr=flag_attr,
            count_attr=count_attr,
            language_annotation=language_annotation,
        )

    def _soft_reset_button_for_success_check(self, button_actor):
        # Match RMBench's gradual reset logic so a held-down button cannot be
        # counted repeatedly in one physical press.
        current = self._get_button_qpos(button_actor)
        self._set_button_unpressed(button_actor, target=min(0.0, current + 0.002))
