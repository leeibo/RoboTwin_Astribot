source /home/lee/anaconda3/etc/profile.d/conda.sh
conda activate robotwin2_copy
cd /home/lee/dyj_code/robotwin2/RoboTwin_Astribot

python - <<'PY'
import math
import os
import yaml
import sapien.core as sapien
import transforms3d as t3d
from envs.utils import create_actor, create_sapien_urdf_obj, rand_create_actor
from envs.utils.cylindrical_coords import world_to_robot
from envs._base_task import Base_Task
from envs.robot.robot import Robot
from envs._GLOBAL_CONFIGS import CONFIGS_PATH

# ============================================================================
# Edit only this section.
# Both cabinet and bottle are placed by cylindrical coordinates relative to the
# robot center on the fan-shaped table.
# ============================================================================

# 036_cabinet
CABINET_CYL_R = 0.6
CABINET_CYL_THETA_DEG = 12.0
CABINET_CYL_Z = 0.741
CABINET_CYL_SPIN_DEG = 0.0

# 001_bottle
BOTTLE_CYL_R = 0.56
BOTTLE_CYL_THETA_DEG = -18.0
BOTTLE_CYL_Z = 0.8
BOTTLE_CYL_SPIN_DEG = 0.0

# 003_plate fixed pose on the upper fan layer.
PLATE_CYL_R = 0.70
PLATE_CYL_THETA_DEG = 0.0
PLATE_Z = 1.04
PLATE_QPOS = [0.5, 0.5, 0.5, 0.5]

task = Base_Task()
#######################################改fan fan_double在这里
with open("task_config/demo_clean_fan_double.yml", "r", encoding="utf-8") as f:
    args = yaml.safe_load(f)
with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
    emb_cfg = yaml.safe_load(f)

robot_file = emb_cfg["astribot_texture"]["file_path"]
with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
    robot_cfg = yaml.safe_load(f)

task.render_freq = 1
task.random_light = False
task.random_background = False
task.clean_background_rate = 0
task.eval_mode = False
task.table_z_bias = 0.0
task.random_head_camera_dis = 0

args.update({
    "render_freq": 1,
    "task_name": "robot_preview_simple",
    "left_robot_file": robot_file,
    "right_robot_file": robot_file,
    "left_embodiment_config": robot_cfg,
    "right_embodiment_config": robot_cfg,
    "dual_arm_embodied": True,
})

task.setup_scene(**args)
task.create_table_and_wall(table_height=0.74, **args)
task.robot = Robot(task.scene, False, **args)
task.robot.init_joints()
task.load_camera(**args)
task.robot.move_to_homestate()

robot_root_xy = task.robot.left_entity_origion_pose.p[:2].tolist()
robot_yaw = float(t3d.euler.quat2euler(task.robot.left_entity_origion_pose.q)[2])


def quat_from_yaw(yaw_rad):
    return t3d.euler.euler2quat(0.0, 0.0, yaw_rad)


def yaw_deg_from_quat(quat):
    return math.degrees(float(t3d.euler.quat2euler(quat)[2]))


def world_xy_from_cyl(r, theta_deg):
    theta_rad = math.radians(theta_deg)
    phi_world = robot_yaw + theta_rad
    x = robot_root_xy[0] + r * math.cos(phi_world)
    y = robot_root_xy[1] + r * math.sin(phi_world)
    return x, y


def cabinet_pose_from_cyl(r, theta_deg, z=0.741, spin_deg=0.0):
    x, y = world_xy_from_cyl(r, theta_deg)
    # For 036_cabinet in this repo, radial-outward yaw makes the cabinet
    # opening face back toward the robot center.
    radial_out_yaw = math.atan2(y - robot_root_xy[1], x - robot_root_xy[0])
    return sapien.Pose([x, y, z], quat_from_yaw(radial_out_yaw + math.radians(spin_deg)))


def bottle_quat_from_spin(spin_deg=0.0):
    base_upright_quat = [0.0, 0.0, 1.0, 0.0]
    return t3d.quaternions.qmult(quat_from_yaw(math.radians(spin_deg)), base_upright_quat)


def plate_pose_on_upper_layer():
    plate_x, plate_y = world_xy_from_cyl(PLATE_CYL_R, PLATE_CYL_THETA_DEG)
    return sapien.Pose([plate_x, plate_y, PLATE_Z], PLATE_QPOS)


def print_pose_summary(name, pose):
    cyl = world_to_robot(
        pose.p.tolist(),
        robot_root_xy=robot_root_xy,
        robot_yaw_rad=robot_yaw,
    )
    print(f"[watch_robot] {name}")
    print(f"  world xyz: ({pose.p[0]:.4f}, {pose.p[1]:.4f}, {pose.p[2]:.4f})")
    print(f"  cyl   r/theta/z: ({cyl[0]:.4f}, {math.degrees(cyl[1]):.2f} deg, {cyl[2]:.4f})")
    print(f"  yaw deg: {yaw_deg_from_quat(pose.q):.2f}")
    print(f"  quat: ({pose.q[0]:.6f}, {pose.q[1]:.6f}, {pose.q[2]:.6f}, {pose.q[3]:.6f})")


##############################################################################
# Load plate / cabinet / bottle

plate_pose = plate_pose_on_upper_layer()
print_pose_summary("plate", plate_pose)

# plate = create_actor(
#     scene=task,
#     pose=plate_pose,
#     modelname="003_plate",
#     is_static=True,
#     convex=True,
# )

cabinet_pose = cabinet_pose_from_cyl(
    r=CABINET_CYL_R,
    theta_deg=CABINET_CYL_THETA_DEG,
    z=CABINET_CYL_Z,
    spin_deg=CABINET_CYL_SPIN_DEG,
)
print_pose_summary("cabinet", cabinet_pose)

cabinet = create_sapien_urdf_obj(
    scene=task,
    pose=cabinet_pose,
    modelname="036_cabinet",
    modelid=46653,
    fix_root_link=True,
)

# bottle_x, bottle_y = world_xy_from_cyl(BOTTLE_CYL_R, BOTTLE_CYL_THETA_DEG)
# bottle_quat = bottle_quat_from_spin(BOTTLE_CYL_SPIN_DEG)
# bottle_pose = sapien.Pose([bottle_x, bottle_y, BOTTLE_CYL_Z], bottle_quat)
# print_pose_summary("bottle", bottle_pose)

# bottle = rand_create_actor(
#     scene=task,
#     modelname="001_bottle",
#     xlim=[bottle_x, bottle_x],
#     ylim=[bottle_y, bottle_y],
#     zlim=[BOTTLE_CYL_Z, BOTTLE_CYL_Z],
#     qpos=bottle_quat,
#     rotate_rand=False,
#     convex=True,
#     model_id=13,
# )
# bottle.set_mass(0.01)

##############################################################################

while not task.viewer.closed:
    task.scene.step()
    task.scene.update_render()
    task.viewer.render()
PY
