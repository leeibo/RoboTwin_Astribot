from .adjust_bottle import adjust_bottle
from .utils import *
import sapien


class adjust_bottle_test_no_rotate_and_head(adjust_bottle):

    def load_actors(self):
        self.model_id = 13
        self.initial_bottle_z = None
        self.arm_tag = None

        self.bottle = rand_create_actor(
            scene=self,
            modelname="001_bottle",
            xlim=[0, 0],
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
            pose=sapien.Pose([0.0, 0.1, 0.741], [0.7071068, 0, 0, 0.7071068]),
            modelname="036_cabinet",
            modelid=46653,
            fix_root_link=True,
        )

    def play_once(self):
        self.arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] >= 0 else "left")
        self.initial_bottle_z = float(self.bottle.get_pose().p[2])

        self.move(self.grasp_actor(self.bottle, arm_tag=self.arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if self.initial_bottle_z is None or self.arm_tag is None:
            return False

        bottle_z = float(self.bottle.get_pose().p[2])
        gripper_closed = self.is_right_gripper_close() if self.arm_tag == "right" else self.is_left_gripper_close()
        return gripper_closed and bottle_z > (self.initial_bottle_z + 0.06)
