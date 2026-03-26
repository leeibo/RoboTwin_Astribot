import torch
import yaml
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.util import logger
from tqdm import tqdm
logger.setup_logger(level="error", logger_name="curobo")

yml_path = "/home/admin1/Desktop/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml"

# robot_pose from config.yml: [0.0, -0.78, 0.0, 0.707, 0, 0, 0.707]
robot_origin_p = [0.0, -0.78, 0.0]

# same table collision as planner.py
world_config = {
    "cuboid": {
        "table": {
            "dims": [0.7, 2, 0.04],
            "pose": [robot_origin_p[1], 0.0, 0.74 - robot_origin_p[2], 1, 0, 0, 0.0],
        },
    }
}

motion_gen_config = MotionGenConfig.load_from_robot_config(
    yml_path,
    world_config,
    interpolation_dt=1 / 250,
    num_trajopt_seeds=4,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

with open(yml_path, "r") as f:
    yml_data = yaml.safe_load(f)
retract_config = yml_data["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
joint_names = yml_data["robot_cfg"]["kinematics"]["cspace"]["joint_names"]

# MotionGen kinematics only uses arm joints (base_link -> ee_link chain),
# exclude the gripper joint which is after ee_link
ee_link = yml_data["robot_cfg"]["kinematics"]["ee_link"]
arm_joint_names = [n for n in joint_names]
arm_retract_config = retract_config[:len(arm_joint_names)]

start_joint_states = JointState.from_position(
    torch.tensor(arm_retract_config, dtype=torch.float32).cuda().reshape(1, -1),
    joint_names=arm_joint_names,
)
print(start_joint_states)
x_values = torch.linspace(-0.35, 0.0, 10).tolist() 
y_values = torch.linspace(-0.45, 0.0, 10).tolist() 
z_values = torch.linspace(0.8, 0.5, 10).tolist() 
quaternion = [1.0, 0.0, 0.0, 0.0]

plan_config = MotionGenPlanConfig(max_attempts=10)

def test_xyz(x, y, z):
    goal_pose = CuroboPose.from_list([float(x), float(y), float(z)] + quaternion)
    result = motion_gen.plan_single(start_joint_states, goal_pose, plan_config)
    if result.success.item():
        print(f"{x:.2f}, {y:.2f}, {z:.2f}, Success")
        print(result.get_interpolated_plan().position.shape)
# test_xyz(-0.27, 0.00, 0.67)
for x in tqdm(x_values,total=len(x_values)):
    for y in y_values:
        for z in z_values:
            test_xyz(x, y, z)
