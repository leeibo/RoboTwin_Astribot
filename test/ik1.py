"""
Calibration IK scan: directly scan in CuRobo base_link frame.
Find valid (x, y, z) positions to hardcode in planner.py for delta_matrix calibration.
"""
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from tqdm import tqdm

YML_PATH = "/home/admin1/Desktop/RoboTwin/assets/embodiments/astribot_descriptions/curobo_left.yml"

config_file = load_yaml(YML_PATH)
tensor_args = TensorDeviceType()
urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    None,
    num_seeds=20,
    self_collision_check=False,
    self_collision_opt=False,
    tensor_args=tensor_args,
    use_cuda_graph=True,
)
ik_solver = IKSolver(ik_config)

# Scan directly in CuRobo base_link frame
x_values = torch.linspace(0.35, 0.0, 25).tolist() + torch.linspace(0.35, 0.7, 25).tolist()
y_values = torch.linspace(0.25, 0.0, 25).tolist() + torch.linspace(0.25, 0.5, 25).tolist()
z_values = torch.linspace(0.25, 0.0, 25).tolist() + torch.linspace(0.25, 0.5, 25).tolist()
quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda:0')

print("Scanning IK in base_link frame (for delta_matrix calibration):")
print("x, y, z, success")
for x in tqdm(x_values, total=len(x_values)):
    for y in y_values:
        for z in z_values:
            goal = Pose(
                position=torch.tensor([[float(x), float(y), float(z)]], device='cuda:0'),
                quaternion=quaternion,
            )
            result = ik_solver.solve_single(goal)
            if result.success.item():
                print(f"{x:.2f}, {y:.2f}, {z:.2f}, SUCCESS")