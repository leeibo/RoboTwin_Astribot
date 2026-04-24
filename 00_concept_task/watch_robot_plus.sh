source /home/lee/anaconda3/etc/profile.d/conda.sh
conda activate robotwin2_copy
cd /home/lee/dyj_code/robotwin2/RoboTwin_Astribot

python - <<'PY'
import math
import os
import yaml
import sapien.core as sapien
import transforms3d as t3d
from envs.utils import create_sapien_urdf_obj, rand_create_actor
from envs.utils.cylindrical_coords import world_to_robot
from envs._base_task import Base_Task
from envs.robot.robot import Robot
from envs._GLOBAL_CONFIGS import CONFIGS_PATH

# ============================================================================
# Manual placement config: edit only this section.
#
# PLACEMENT_MODE:
#   - "cyl": use robot-centered fan-table coordinates and face robot center
#   - "face_world": use world xyz and face robot center
#   - "manual_world": use world xyz + explicit yaw
# ============================================================================
PLACEMENT_MODE = "cyl"

# Mode 1: cylindrical placement relative to robot center.
CYL_R = 0.56
CYL_THETA_DEG = 12.0
CYL_Z = 0.741
CYL_SPIN_DEG = 0.0

# Mode 2: world placement while still facing robot center.
FACE_WORLD_X = 0.10
FACE_WORLD_Y = 0.40
FACE_WORLD_Z = 0.741
FACE_WORLD_SPIN_DEG = 0.0

# Mode 3: fully manual world placement.
MANUAL_WORLD_X = 0.10
MANUAL_WORLD_Y = 0.40
MANUAL_WORLD_Z = 0.741
MANUAL_WORLD_YAW_DEG = 90.0

# Optional extra objects for reference.
ENABLE_BOTTLE = True
BOTTLE_X = 0.0
BOTTLE_Y = 0.0
BOTTLE_Z = 0.75

task = Base_Task()

with open("task_config/demo_clean.yml", "r", encoding="utf-8") as f:
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
    "task_name": "robot_preview_fixed",
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


def cabinet_pose_face_robot_center(x, y, z=0.741, spin_deg=0.0):
    # For 036_cabinet in this repo, radial-outward yaw makes the opening
    # face back toward the robot center.
    radial_out_yaw = math.atan2(y - robot_root_xy[1], x - robot_root_xy[0])
    return sapien.Pose([x, y, z], quat_from_yaw(radial_out_yaw + math.radians(spin_deg)))


def cabinet_pose_from_cyl(r, theta_deg, z=0.741, spin_deg=0.0):
    x, y = world_xy_from_cyl(r, theta_deg)
    return cabinet_pose_face_robot_center(x, y, z=z, spin_deg=spin_deg)


def build_cabinet_pose():
    if PLACEMENT_MODE == "cyl":
        return cabinet_pose_from_cyl(
            r=CYL_R,
            theta_deg=CYL_THETA_DEG,
            z=CYL_Z,
            spin_deg=CYL_SPIN_DEG,
        )
    if PLACEMENT_MODE == "face_world":
        return cabinet_pose_face_robot_center(
            x=FACE_WORLD_X,
            y=FACE_WORLD_Y,
            z=FACE_WORLD_Z,
            spin_deg=FACE_WORLD_SPIN_DEG,
        )
    if PLACEMENT_MODE == "manual_world":
        return sapien.Pose(
            [MANUAL_WORLD_X, MANUAL_WORLD_Y, MANUAL_WORLD_Z],
            quat_from_yaw(math.radians(MANUAL_WORLD_YAW_DEG)),
        )
    raise ValueError(f"Unsupported PLACEMENT_MODE: {PLACEMENT_MODE}")


def print_pose_summary(pose):
    cyl = world_to_robot(
        pose.p.tolist(),
        robot_root_xy=robot_root_xy,
        robot_yaw_rad=robot_yaw,
    )
    print("[watch_robot_plus] placement summary")
    print(f"  mode: {PLACEMENT_MODE}")
    print(f"  world xyz: ({pose.p[0]:.4f}, {pose.p[1]:.4f}, {pose.p[2]:.4f})")
    print(f"  cyl   r/theta/z: ({cyl[0]:.4f}, {math.degrees(cyl[1]):.2f} deg, {cyl[2]:.4f})")
    print(f"  yaw deg: {yaw_deg_from_quat(pose.q):.2f}")
    print(f"  quat: ({pose.q[0]:.6f}, {pose.q[1]:.6f}, {pose.q[2]:.6f}, {pose.q[3]:.6f})")


##############################################################################
# Load reference objects

if ENABLE_BOTTLE:
    bottle = rand_create_actor(
        scene=task,
        modelname="001_bottle",
        xlim=[BOTTLE_X, BOTTLE_X],
        ylim=[BOTTLE_Y, BOTTLE_Y],
        zlim=[BOTTLE_Z, BOTTLE_Z],
        qpos=[0, 0, 1, 0],
        rotate_rand=False,
        convex=True,
        model_id=13,
    )
    bottle.set_mass(0.01)

cabinet_pose = build_cabinet_pose()
print_pose_summary(cabinet_pose)

cabinet = create_sapien_urdf_obj(
    scene=task,
    pose=cabinet_pose,
    modelname="036_cabinet",
    modelid=46653,
    fix_root_link=True,
)

##############################################################################

while not task.viewer.closed:
    task.scene.step()
    task.scene.update_render()
    task.viewer.render()
PY
