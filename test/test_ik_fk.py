"""
Verification: FK -> IK round-trip test.
1. Use CuRobo FK to compute ee pose at homestate (all joints = 0)
2. Use CuRobo IK to solve for that known pose
3. Verify round-trip consistency
"""
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig

YML_PATH = "/home/admin1/Desktop/RoboTwin/assets/embodiments/astribot_descriptions/curobo_left.yml"

config_file = load_yaml(YML_PATH)
tensor_args = TensorDeviceType()
urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

print(f"URDF: {urdf_file}")
print(f"base_link: {base_link}")
print(f"ee_link: {ee_link}")

robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
n_dof = robot_cfg.kinematics.kinematics_config.n_dof
joint_names = robot_cfg.kinematics.kinematics_config.joint_names
print(f"\nCuRobo joint names: {joint_names}")
print(f"CuRobo n_dof: {n_dof}")

# ========== Step 1: FK at homestate ==========
cuda_model = CudaRobotModel(robot_cfg.kinematics)
homestate = torch.zeros(1, n_dof, device='cuda:0')
fk_result = cuda_model.forward(homestate)
ee_pos = fk_result[0].cpu().numpy()[0]
ee_quat = fk_result[1].cpu().numpy()[0]
print(f"\n=== FK at homestate (all joints=0) ===")
print(f"ee position (in base_link frame): [{ee_pos[0]:.6f}, {ee_pos[1]:.6f}, {ee_pos[2]:.6f}]")
print(f"ee quaternion [w,x,y,z]: [{ee_quat[0]:.6f}, {ee_quat[1]:.6f}, {ee_quat[2]:.6f}, {ee_quat[3]:.6f}]")

# ========== Step 2: IK for FK pose ==========
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

goal = Pose(
    position=torch.tensor([[ee_pos[0], ee_pos[1], ee_pos[2]]], device='cuda:0', dtype=torch.float32),
    quaternion=torch.tensor([[ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]]], device='cuda:0', dtype=torch.float32),
)
result = ik_solver.solve_single(goal)
print(f"\n=== IK solve for FK homestate pose ===")
print(f"Success: {result.success.item()}")
if result.success.item():
    solved_q = result.solution.cpu().numpy()[0][0]
    print(f"Solved joint angles: {solved_q}")
    # FK verification
    fk_check = cuda_model.forward(result.solution[0])
    check_pos = fk_check[0].cpu().numpy()[0]
    print(f"FK of solved joints: [{check_pos[0]:.6f}, {check_pos[1]:.6f}, {check_pos[2]:.6f}]")
    print(f"Position error: {((ee_pos - check_pos)**2).sum()**0.5:.8f}")

# ========== Step 3: IK for identity quaternion at same position ==========
goal2 = Pose(
    position=torch.tensor([[ee_pos[0], ee_pos[1], ee_pos[2]]], device='cuda:0', dtype=torch.float32),
    quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda:0', dtype=torch.float32),
)
result2 = ik_solver.solve_single(goal2)
print(f"\n=== IK solve: same position, quaternion=[1,0,0,0] ===")
print(f"Success: {result2.success.item()}")
if result2.success.item():
    solved_q2 = result2.solution.cpu().numpy()[0][0]
    print(f"Solved joint angles: {solved_q2}")

# ========== Step 4: test a few positions ==========
print(f"\n=== Quick IK scan in base_link frame ===")
test_positions = [
    [0.1, 0.0, 0.0],
    [0.2, 0.0, 0.0],
    [0.3, 0.0, 0.0],
    [0.0, 0.1, 0.0],
    [0.0, 0.2, 0.0],
    [0.0, 0.0, 0.1],
    [0.0, 0.0, 0.2],
    [0.1, 0.1, 0.1],
    [0.2, 0.2, 0.0],
    [0.3, 0.1, 0.1],
    [-0.1, 0.0, 0.0],
    [-0.2, 0.0, 0.0],
    [0.0, -0.1, 0.0],
    [0.0, 0.0, -0.1],
    [0.0, 0.0, -0.2],
    [ee_pos[0]*0.5, ee_pos[1]*0.5, ee_pos[2]*0.5],
]
for pos in test_positions:
    goal_t = Pose(
        position=torch.tensor([pos], device='cuda:0', dtype=torch.float32),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda:0', dtype=torch.float32),
    )
    res = ik_solver.solve_single(goal_t)
    status = "OK" if res.success.item() else "FAIL"
    print(f"  pos=[{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}] -> {status}")
