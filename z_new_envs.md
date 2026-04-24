这是一个记录当前新增任务与结果的文档。

## 统计说明

- 成功率数据来自各任务目录下的 `data_show/.../seed.txt`。
- `script/collect_data.py` 中，`seed.txt` 只记录满足 `plan_success && check_success()` 的成功 seed。
- 当前 `seed.txt` 采用空格分隔，因此本文按“任意空白分隔”读取 seed。
- 成功数 = `seed.txt` 中的 seed 个数。
- 总尝试数 = `max(seed) + 1`，默认认为采集从 `seed=0` 递增尝试。
- 成功率 = 成功数 / 总尝试数。
- 如果同一个任务存在多个结果目录，则分别列出。

## 通用可调参数

### fan_double 任务

- 默认配置文件：`task_config/demo_clean_fan_double.yml`
- 常调场景参数：`fan_angle_deg`、`fan_double_layer_gap`、`fan_double_upper_theta_start_deg`、`fan_double_upper_theta_end_deg`、`fan_double_support_theta_deg`
- 常调采集参数：`episode_num`、`save_freq`、`language_num`、`save_path`
- 常调随机化参数：`domain_randomization.random_background`、`domain_randomization.cluttered_table`、`domain_randomization.random_light`

### 单层 fan 任务

- 默认配置文件：`task_config/demo_clean.yml`
- 常调场景参数：`fan_outer_radius`、`fan_inner_radius`、`fan_angle_deg`、`fan_center_deg`
- 常调采集参数：`episode_num`、`save_freq`、`language_num`、`save_path`

## put_block_on

对应文件：`envs/put_block_on.py`  
共享主逻辑：`envs/_put_block_target_fan_double_base.py`

### 任务描述

- 在双层扇形桌面上生成若干 block 和一个 plate。
- 机器人需要先搜索 block，再逐个抓取，并把所有 block 放到 plate 上。
- 当前任务同时支持上层/下层抓取和上层/下层放置，因此包含同层抓放、跨层抓放和 direct release。

### 主要可调参数

- 物体数量与分层：`BLOCK_COUNT`、`BLOCK_LAYER_SEQUENCE`、`PLATE_LAYER`
- 物体尺寸与生成范围：`BLOCK_SIZE_RANGE`、`BLOCK_LAYER_SPECS`、`PLATE_LAYER_SPECS`
- plate 槽位：`PLATE_PLACE_SLOT_OFFSETS`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抓取参数：`PICK_PRE_GRASP_DIS`、`PICK_GRASP_DIS`、`PICK_LIFT_Z`、`UPPER_PICK_*`
- 放置参数：`LOWER_PLACE_*`、`DIRECT_RELEASE_*`、`UPPER_TO_LOWER_*`
- 成功阈值：`SUCCESS_EPS`

### 成功判定

- 所有 block 都需要落在 plate 目标区域附近。
- 默认位置误差阈值为 `SUCCESS_EPS = [0.08, 0.08, 0.08]`。
- 左右夹爪都需要张开。

### 已有结果

- `data_show/put_block_on/demo_clean_fan_double__medium_fan180`：`8 / 46 = 17.39%`
- `data_show/put_block_on/demo_clean_fan_double__medium_fan180_platelower`：`10 / 33 = 30.30%`
- `data_show/put_block_on/demo_clean_fan_double__medium_fan180_plateupper`：`10 / 71 = 14.08%`

### Tips

稍微调整了一下参数，现在成功率很高

## put_block_plasticbox_fan_double

对应文件：`envs/put_block_plasticbox_fan_double.py`  
共享主逻辑：`envs/_put_block_target_fan_double_base.py`

### 任务描述

- 在双层扇形桌面上生成一个绿色 block 和一个 plastic box。
- 当前默认设置下，block 在下层，plastic box 在上层。
- 任务重点是先搜索 block，再搜索容器，并沿 `put_block_on` 的同一条放置链把 block 放进 plastic box。

### 主要可调参数

- 容器型号：`TARGET_MODEL_ID`
- 容器分层与位置：`TARGET_LAYER`、`TARGET_LAYER_SPECS`
- 容器避让范围：`TARGET_PADDING`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抓放参数：`BLOCK_LAYER_SEQUENCE`、`LOWER_PLACE_*`、`DIRECT_RELEASE_*`、`UPPER_TO_LOWER_*`
- 成功阈值：`SUCCESS_XY_TOL`、`SUCCESS_Z_TOL`

### 成功判定

- block 的 functional point 需要靠近 plastic box 的目标 functional point。
- `xy` 误差小于 `0.08`，`z` 误差小于 `0.06`。
- block 需要与 plastic box 接触，且左右夹爪都张开。

### 已有结果

- `data_show/put_block_plasticbox_fan_double/demo_clean_fan_double__medium_fan180`：`5 / 6 = 83.33%`

### Tips

- 共享于_put_block_target_fan_double_base.py

## put_block_breadbasket_fan_double

对应文件：`envs/put_block_breadbasket_fan_double.py`  
共享主逻辑：`envs/_put_block_target_fan_double_base.py`

### 任务描述

- 在双层扇形桌面上生成一个绿色 block 和一个 bread basket。
- 当前默认设置下，block 在下层，bread basket 在上层。
- 任务流程和 `put_block_plasticbox_fan_double` 一致，但目标容器换成了 basket 类容器。

### 主要可调参数

- 容器型号：`TARGET_MODEL_ID`
- 容器分层与位置：`TARGET_LAYER`、`TARGET_LAYER_SPECS`
- 容器避让范围：`TARGET_PADDING`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抓放参数：`BLOCK_LAYER_SEQUENCE`、`LOWER_PLACE_*`、`DIRECT_RELEASE_*`、`UPPER_TO_LOWER_*`
- 成功阈值：`SUCCESS_XY_TOL`、`SUCCESS_Z_TOL`

### 成功判定

- block 的 functional point 需要靠近 bread basket 的目标 functional point。
- `xy` 误差小于 `0.09`，`z` 误差小于 `0.08`。
- block 需要与 bread basket 接触，且左右夹爪都张开。

### 已有结果

- `data_show/put_block_breadbasket_fan_double/demo_clean_fan_double__medium_fan180`：`5 / 6 = 83.33%`

### Tips

- 共享于_put_block_target_fan_double_base.py

## put_block_skillet_fan_double

对应文件：`envs/put_block_skillet_fan_double.py`  
共享主逻辑：`envs/_put_block_target_fan_double_base.py`

### 任务描述

- 在双层扇形桌面上生成一个绿色 block 和一个 skillet。
- 当前默认设置下，block 在下层，skillet 在上层。
- 任务仍然沿 `put_block_on` 的搜索和放置逻辑执行，只是目标从 plate 换成了 skillet，语义上更接近“放到锅上”。

### 主要可调参数

- 容器型号：`TARGET_MODEL_ID`
- 容器分层与位置：`TARGET_LAYER`、`TARGET_LAYER_SPECS`
- 容器朝向：`TARGET_LAYER_SPECS[*].qpos`
- 容器避让范围：`TARGET_PADDING`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抓放参数：`BLOCK_LAYER_SEQUENCE`、`LOWER_PLACE_*`、`DIRECT_RELEASE_*`、`UPPER_TO_LOWER_*`
- 成功阈值：`SUCCESS_XY_TOL`、`SUCCESS_Z_TOL`

### 成功判定

- block 的 functional point 需要靠近 skillet 的目标 functional point。
- `xy` 误差小于 `0.06`，`z` 误差小于 `0.05`。
- block 需要与 skillet 接触，且左右夹爪都张开。

### 已有结果

- `data_show/put_block_skillet_fan_double/demo_clean_fan_double__medium_fan180`：`5 / 6 = 83.33%`

### Tips

- 共享于_put_block_target_fan_double_base.py

## blocks_ranking_rgb_fan_double

对应文件：`envs/blocks_ranking_rgb_fan_double.py`

### 任务描述

- 在双层扇形桌面上生成红、绿、蓝三个 block。
- 红色 block 默认作为左侧锚点，绿色和蓝色 block 需要被依次搬运。
- 最终目标是让三个 block 按颜色从左到右排成红、绿、蓝。

### 主要可调参数

- 分层与采样范围：`LAYER_SPECS`、`BLOCK_SPAWN_MIN_DIST_SQ`
- 方块尺寸：`BLOCK_SIZE_RANGE`
- 方块定义：`BLOCK_DEFS`
- 目标行布局：`TARGET_LAYER`、`TARGET_ROW_SPEC`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抓放参数：`PICK_PRE_GRASP_DIS`、`PICK_GRASP_DIS`、`LOWER_PLACE_*`、`UPPER_PICK_*`、`DIRECT_RELEASE_*`
- 成功阈值：`SUCCESS_XY_TOL`、`SUCCESS_Z_TOL`

### 成功判定

- 三个 block 都要靠近各自目标位置。
- `xy` 误差小于 `0.09`，`z` 误差小于 `0.08`。
- 三个 block 需要满足从左到右的颜色顺序，并基本处在同一条圆弧带上。
- 左右夹爪都需要张开。

### 已有结果

- `data_show/blocks_ranking_rgb_fan_double/demo_clean_fan_double__medium_fan180`：`10 / 51 = 19.61%`

### Tips

- 暂无

## blocks_ranking_size_fan_double

对应文件：`envs/blocks_ranking_size_fan_double.py`

### 任务描述

- 在双层扇形桌面上生成大、中、小三个 block。
- 大 block 默认作为左侧锚点，中、小两个 block 需要被依次搬运。
- 最终目标是让三个 block 从左到右按尺寸递减排列。

### 主要可调参数

- 分层与采样范围：`LAYER_SPECS`、`BLOCK_SPAWN_MIN_DIST_SQ`
- 方块定义与尺寸区间：`BLOCK_DEFS`
- 目标行布局：`TARGET_LAYER`、`TARGET_ROW_SPEC`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抓放参数：`PICK_PRE_GRASP_DIS`、`PICK_GRASP_DIS`、`LOWER_PLACE_*`、`UPPER_PICK_*`、`DIRECT_RELEASE_*`
- 成功阈值：`SUCCESS_XY_TOL`、`SUCCESS_Z_TOL`

### 成功判定

- 三个 block 都要靠近目标序列位置。
- `xy` 误差小于 `0.09`，`z` 误差小于 `0.08`。
- 三个 block 需要满足从左到右的大、中、小顺序，并基本位于同一圆弧带上。
- 左右夹爪都需要张开。

### 已有结果

- `data_show/blocks_ranking_size_fan_double/demo_clean_fan_double__medium_fan180`：`10 / 50 = 20.00%`

### Tips

- 暂无

## place_object_basket_fan_double

对应文件：`envs/place_object_basket_fan_double.py`

### 任务描述

- 在双层扇形桌面上随机生成一个物体和一个 basket。
- 默认设置下，物体位于下层，basket 位于上层。
- 当前物体池包括 `081_playingcards` 和 `057_toycar`，任务需要完成搜索、抓取和放入 basket。

### 主要可调参数

- 物体与篮子分层：`OBJECT_LAYER`、`BASKET_LAYER`
- 物体类别池：`OBJECT_CANDIDATES`
- 篮子模型池：`BASKET_MODEL_IDS`
- 物体生成方式：`OBJECT_R_RANGE`、`OBJECT_POSE_SPECS`、`OBJECT_ROTATE_RAND`、`OBJECT_ROTATE_LIM`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抓取参数：`OBJECT_PRE_GRASP_DIS`、`PICK_LIFT_Z`、`UPPER_PICK_*`
- 放置参数：`DIRECT_RELEASE_*`、`UPPER_PLACE_LATERAL_ESCAPE_DIS`
- 成功阈值：`SUCCESS_DIST`、`SUCCESS_Z_MIN_DELTA`

### 成功判定

- 物体和 basket 中心距离小于 `SUCCESS_DIST = 0.18`。
- 物体相对初始高度提升超过 `SUCCESS_Z_MIN_DELTA = 0.015`。
- 左右夹爪都需要张开。

### 已有结果

- `data_show/place_object_basket_fan_double/demo_clean_fan_double__medium_fan180`：`10 / 46 = 21.74%`
- `data_show/place_object_basket_fan_double/demo_clean_fan_double__medium_fan180_problem`：`10 / 45 = 22.22%`

### Tips

- 暂无

## place_can_basket_fan_double

对应文件：`envs/place_can_basket_fan_double.py`

### 任务描述

- 在双层扇形桌面上随机生成一个 can 和一个 basket。
- 默认设置下，can 位于下层，basket 位于上层。
- 任务需要完成搜索、抓取，并把 can 放入 basket。

### 主要可调参数

- 物体与篮子分层：`CAN_LAYER`、`BASKET_LAYER`
- can 模型池：`CAN_MODEL_IDS`
- basket 模型池：`BASKET_MODEL_IDS`
- can 生成方式：`CAN_R_RANGE`、`CAN_POSE_SPECS`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抓取参数：`CAN_PRE_GRASP_DIS`、`PICK_LIFT_Z`、`UPPER_PICK_*`
- 放置参数：`DIRECT_RELEASE_*`、`UPPER_PLACE_LATERAL_ESCAPE_DIS`
- 成功阈值：`SUCCESS_DIST`、`SUCCESS_Z_MIN_DELTA`

### 成功判定

- can 和 basket 中心距离小于 `SUCCESS_DIST = 0.18`。
- can 相对初始高度提升超过 `SUCCESS_Z_MIN_DELTA = 0.02`。
- 左右夹爪都需要张开。

### 已有结果

- `data_show/place_can_basket_fan_double/demo_clean_fan_double__medium_fan180`：`13 / 16 = 81.25%`


## search_object

对应文件：`envs/search_object.py`

### 任务描述

- 在单层 fan 桌面上放置一个 cabinet，并把目标物体藏在 drawer 内侧。
- 当前目标物体不再只有小方块，还会在 `block`、`toy car`、`rubik's cube` 三类变体之间随机。
- 机器人先搜索目标；如果目标不可见，则需要先打开 drawer，再抓取内部物体并抬起。

### 主要可调参数

- 柜体位置：`CABINET_CYL_R`、`CABINET_CYL_THETA_DEG_RANGE`、`CABINET_CYL_Z`、`CABINET_CYL_SPIN_DEG`
- 目标物体池：`OBJECT_VARIANTS`、`OBJECT_COLOR_CANDIDATES`
- 搜索参数：`SCAN_R`、`SCAN_Z_BIAS`、`SCAN_JOINT_NAME`
- 抽屉开合参数：`DRAWER_OPEN_SUCCESS_DIS`、`DRAWER_PULL_TOTAL_DIS`、`DRAWER_PULL_STEPS`
- 柜门抓取参数：`CABINET_PRE_GRASP_DIS`、`CABINET_GRASP_DIS`、`CABINET_GRIPPER_POS`
- 物体抓取参数：`OBJECT_PRE_GRASP_DIS`、`OBJECT_GRASP_DIS`、`OBJECT_APPROACH_CLEARANCE_Z`、`OBJECT_LIFT_Z`
- 成功阈值：`SUCCESS_LIFT_Z`

### 成功判定

- 目标物体被抓住后，需要抬高到初始高度以上 `0.03m`。
- 执行抓取的那只手需要保持闭合。

### 已有结果

- `data_show/search_object/demo_clean__easy_fan150`：`20 / 77 = 25.97%`

### Tips

- 左右手同样的方法，同样的参数，因为镜像的问题，成功概率很不一样，我把两种方法放在下面了，old左手容易成功，new右手容易成功
- 左边左手抓(1.0) 柜子R=0.7，右边右手抓（-1.0）柜子R=0.6 容易成功  # CABINET_THETA_SIGN_CHOICES 确定柜子在左还是右
- 按照上面的方法，成功率高很多

- 里面物品在柜子里深度是可以调整的，如果发现抓不到or看不到都可以调整




new:
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .open_carbinet import open_carbinet
from .utils import *


class search_object(Base_Task):
    CABINET_MODEL_ID = 46653
    OBJECT_LABEL = "small block"
    OBJECT_HALF_SIZE = 0.018
    OBJECT_COLOR = (0.10, 0.80, 0.20)
    OBJECT_COLOR_CANDIDATES = (
        (0.90, 0.20, 0.20),
        (0.15, 0.72, 0.25),
        (0.20, 0.45, 0.92),
        (0.92, 0.74, 0.18),
        (0.88, 0.45, 0.16),
    )
    OBJECT_MASS = 0.03
    OBJECT_VARIANTS = (
        {
            "kind": "block",
            "label": OBJECT_LABEL,
            "outward_offset": 0.03,
            "surface_z_offset": OBJECT_HALF_SIZE + 0.002,
            "mass": OBJECT_MASS,
        },
        {
            "kind": "asset",
            "modelname": "057_toycar",
            "label": "toy car",
            "base_q": (0.7071068, 0.7071068, 0.0, 0.0),
            "outward_offset": 0.02,
            "surface_z_offset": 0.0,
            "mass": OBJECT_MASS,
        },
        {
            "kind": "asset",
            "modelname": "073_rubikscube",
            "label": "rubik's cube",
            "base_q": (0.7071068, 0.7071068, 0.0, 0.0),
            "outward_offset": 0.02,
            "surface_z_offset": 0.0,
            "mass": OBJECT_MASS,
        },
    )

    CABINET_CYL_R = 0.6 # 柜子左右随机，再line 260
    CABINET_CYL_THETA_DEG_RANGE = (8.0, 24.0)
    CABINET_CYL_Z = 0.741
    CABINET_CYL_SPIN_DEG = 0.0

    TASK_HOMESTATE = [
        [-0.11, -0.7, -0.8, 2, -0.9, 0, 0],
        [0.11, -0.7, 0.8, 2, 0.9, 0, 0],
    ]

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    RIGHT_ARM_ROTATE_LIM = (-1.0, 0.0)
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.18
    UPPER_PLACE_BODY_JOINT_NAME = "astribot_torso_joint_2"

    OBJECT_Z_BIAS = OBJECT_HALF_SIZE + 0.002
    OBJECT_OUTER_EDGE_OFFSET = 0.03
    DRAWER_OPEN_SUCCESS_DIS = open_carbinet.DRAWER_OPEN_SUCCESS_DIS
    DRAWER_PULL_TOTAL_DIS = open_carbinet.DRAWER_PULL_TOTAL_DIS
    DRAWER_PULL_STEPS = open_carbinet.DRAWER_PULL_STEPS
    CABINET_PRE_GRASP_DIS = open_carbinet.CABINET_PRE_GRASP_DIS
    CABINET_GRASP_DIS = open_carbinet.CABINET_GRASP_DIS
    CABINET_GRIPPER_POS = open_carbinet.CABINET_GRIPPER_POS
    OBJECT_PRE_GRASP_DIS = 0.12 # 
    OBJECT_GRASP_DIS = 0.01
    OBJECT_APPROACH_CLEARANCE_Z = 0.12
    OBJECT_APPROACH_PRE_GRASP_MARGIN_Z = 0.08
    OBJECT_LIFT_Z = 0.08
    SUCCESS_LIFT_Z = 0.03

    def setup_demo(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("table_shape", "fan")
        kwargs.setdefault("fan_center_on_robot", True)
        kwargs.setdefault("fan_outer_radius", 0.9)
        kwargs.setdefault("fan_inner_radius", 0.3)
        kwargs.setdefault("fan_angle_deg", 150)
        kwargs.setdefault("fan_center_deg", 90)
        right_arm_rotate_lim = kwargs.pop("right_arm_rotate_lim", self.RIGHT_ARM_ROTATE_LIM)

        for cfg_key in ["left_embodiment_config", "right_embodiment_config"]:
            if cfg_key in kwargs and kwargs[cfg_key] is not None:
                cfg = deepcopy(kwargs[cfg_key])
                cfg["homestate"] = deepcopy(self.TASK_HOMESTATE)
                if cfg_key == "right_embodiment_config":
                    cfg["rotate_lim"] = list(right_arm_rotate_lim)
                kwargs[cfg_key] = cfg

        kwargs = init_rotate_theta_bounds(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    @staticmethod
    def _quat_from_yaw(yaw_rad):
        return t3d.euler.euler2quat(0.0, 0.0, float(yaw_rad))

    def _world_xy_from_cyl(self, r, theta_deg):
        theta_rad = float(np.deg2rad(theta_deg))
        phi_world = float(self.robot_yaw + theta_rad)
        x = float(self.robot_root_xy[0] + float(r) * np.cos(phi_world))
        y = float(self.robot_root_xy[1] + float(r) * np.sin(phi_world))
        return x, y

    def _cabinet_pose_from_cyl(self, r, theta_deg, z, spin_deg):
        x, y = self._world_xy_from_cyl(r=r, theta_deg=theta_deg)
        radial_out_yaw = float(np.arctan2(y - self.robot_root_xy[1], x - self.robot_root_xy[0]))
        yaw = float(radial_out_yaw + np.deg2rad(spin_deg))
        return sapien.Pose([x, y, float(z)], self._quat_from_yaw(yaw))

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object,
                "B": self.cabinet,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "search_hidden_object",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "object_not_found",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "open_seen_cabinet",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "cabinet_opened",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "pick_object_from_cabinet",
                    "instruction_idx": 3,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": False,
                    "done_when": "object_grasped_and_lifted",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Search for {A}; if it is not visible, open {B} and pick {A} up.",
        )

    @staticmethod
    def _get_available_model_ids(modelname):
        model_dir = Path("assets/objects") / str(modelname)
        available_ids = []
        for json_path in model_dir.glob("model_data*.json"):
            suffix = json_path.stem.replace("model_data", "")
            if suffix.isdigit():
                available_ids.append(int(suffix))
        return sorted(available_ids)

    def _sample_block_color(self):
        color_idx = int(np.random.randint(len(self.OBJECT_COLOR_CANDIDATES)))
        return tuple(float(channel) for channel in self.OBJECT_COLOR_CANDIDATES[color_idx])

    def _compose_object_quat(self, drawer_pose, base_q=None):
        drawer_q = np.array(drawer_pose.q, dtype=np.float64)
        if base_q is None:
            return drawer_q
        return np.array(t3d.quaternions.qmult(drawer_q, np.array(base_q, dtype=np.float64)), dtype=np.float64)

    def _build_drawer_object_pose(self, drawer_pose, drawer_outward_dir, outward_offset, surface_z_offset, quat):
        return sapien.Pose(
            np.array(drawer_pose.p, dtype=np.float64)
            + np.array(
                [
                    float(drawer_outward_dir[0]) * float(outward_offset),
                    float(drawer_outward_dir[1]) * float(outward_offset),
                    float(surface_z_offset - self.table_z_bias),
                ],
                dtype=np.float64,
            ),
            np.array(quat, dtype=np.float64),
        )

    def _create_search_target_object(self, drawer_pose, drawer_outward_dir):
        variant = dict(self.OBJECT_VARIANTS[int(np.random.randint(len(self.OBJECT_VARIANTS)))])
        if variant["kind"] == "block":
            block_pose = self._build_drawer_object_pose(
                drawer_pose=drawer_pose,
                drawer_outward_dir=drawer_outward_dir,
                outward_offset=float(variant.get("outward_offset", self.OBJECT_OUTER_EDGE_OFFSET)),
                surface_z_offset=float(variant.get("surface_z_offset", self.OBJECT_Z_BIAS)),
                quat=self._compose_object_quat(drawer_pose),
            )
            block = create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.OBJECT_HALF_SIZE, self.OBJECT_HALF_SIZE, self.OBJECT_HALF_SIZE),
                color=self._sample_block_color(),
                name="search_object_block",
            )
            block.set_mass(float(variant.get("mass", self.OBJECT_MASS)))
            self.selected_modelname = None
            self.selected_model_id = None
            return block, str(variant.get("label", self.OBJECT_LABEL))

        modelname = str(variant["modelname"])
        available_model_ids = self._get_available_model_ids(modelname)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {modelname}")
        model_id = int(np.random.choice(available_model_ids))
        object_pose = self._build_drawer_object_pose(
            drawer_pose=drawer_pose,
            drawer_outward_dir=drawer_outward_dir,
            outward_offset=float(variant.get("outward_offset", self.OBJECT_OUTER_EDGE_OFFSET)),
            surface_z_offset=float(variant.get("surface_z_offset", 0.0)),
            quat=self._compose_object_quat(drawer_pose, base_q=variant.get("base_q", None)),
        )
        obj = create_actor(
            scene=self,
            pose=object_pose,
            modelname=modelname,
            convex=True,
            model_id=model_id,
        )
        obj.set_mass(float(variant.get("mass", self.OBJECT_MASS)))
        self.selected_modelname = modelname
        self.selected_model_id = model_id
        return obj, str(variant.get("label", modelname))

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.cabinet_opened = False
        self.object_arm_tag = None
        self.cabinet_arm_tag = None
        self.initial_object_z = None
        self.object_label = str(self.OBJECT_LABEL)
        self.selected_modelname = None
        self.selected_model_id = None

        cabinet_theta_abs_deg = float(np.random.uniform(*self.CABINET_CYL_THETA_DEG_RANGE))
        self.cabinet_theta_deg = float(cabinet_theta_abs_deg * np.random.choice([1.0, 1.0])) # np.random.choice([-1.0, 1.0]) + left - right
        cabinet_pose = self._cabinet_pose_from_cyl(
            r=self.CABINET_CYL_R,
            theta_deg=self.cabinet_theta_deg,
            z=self.CABINET_CYL_Z,
            spin_deg=self.CABINET_CYL_SPIN_DEG,
        )
        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            pose=cabinet_pose,
            modelname="036_cabinet",
            modelid=self.CABINET_MODEL_ID,
            fix_root_link=True,
        )

        drawer_pose = self.cabinet.get_functional_point(0, "pose")
        drawer_outward_dir = self._get_drawer_outward_dir_xy()
        self.object, self.object_label = self._create_search_target_object(
            drawer_pose=drawer_pose,
            drawer_outward_dir=drawer_outward_dir,
        )
        self.initial_drawer_world_point = self._get_drawer_world_point()

        self.add_prohibit_area(self.cabinet, padding=0.03)
        self.add_prohibit_area(self.object, padding=0.03)
        self._configure_rotate_subtask_plan()

    def _project_rotate_registry_object(self, object_key, camera_pose=None, camera_spec=None):
        if str(object_key) == "A" and not bool(getattr(self, "cabinet_opened", False)):
            return None
        return super()._project_rotate_registry_object(
            object_key,
            camera_pose=camera_pose,
            camera_spec=camera_spec,
        )

    def _get_cabinet_arm_tag(self):
        cabinet_cyl = world_to_robot(self.cabinet.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if float(cabinet_cyl[1]) >= 0.0 else "right")

    def _get_drawer_outward_dir_xy(self):
        cabinet_xy = np.array(self.cabinet.get_pose().p[:2], dtype=np.float64)
        robot_xy = np.array(self.robot_root_xy, dtype=np.float64)
        direction = robot_xy - cabinet_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return np.array([0.0, -1.0], dtype=np.float64)
        return direction / norm

    def _get_drawer_world_point(self):
        return np.array(self.cabinet.get_functional_point(0)[:3], dtype=np.float64)

    def _get_drawer_pull_step_xy(self):
        direction = self._get_drawer_outward_dir_xy()
        step_dis = float(self.DRAWER_PULL_TOTAL_DIS) / float(max(int(self.DRAWER_PULL_STEPS), 1))
        return (direction * step_dis).tolist()

    def _get_current_body_facing_yaw(self):
        joint_idx = self._get_preferred_torso_joint_index(
            joint_name_prefer=getattr(self, "UPPER_PLACE_BODY_JOINT_NAME", self.SCAN_JOINT_NAME)
        )
        torso_joints = list(getattr(self.robot, "torso_joints", []) or [])
        if joint_idx is not None and 0 <= joint_idx < len(torso_joints):
            joint = torso_joints[joint_idx]
            body_link = None if joint is None else getattr(joint, "child_link", None)
            if body_link is not None:
                facing_yaw, _ = self._compute_link_planar_facing_yaw(body_link)
                if facing_yaw is not None and np.isfinite(float(facing_yaw)):
                    return float(facing_yaw)
        return float(self.robot_yaw)

    def _get_upper_place_lateral_escape_xy(self, arm_tag):
        lateral_dis = float(getattr(self, "UPPER_PLACE_LATERAL_ESCAPE_DIS", 0.0))
        if lateral_dis <= 1e-9:
            return None

        body_yaw = self._get_current_body_facing_yaw()
        leftward_xy = np.array(
            [-np.sin(body_yaw), np.cos(body_yaw)],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(leftward_xy))
        if norm <= 1e-9:
            return None
        leftward_xy /= norm
        if ArmTag(arm_tag) == "right":
            leftward_xy = -leftward_xy
        return (leftward_xy * lateral_dis).tolist()

    def _retreat_cabinet_arm_after_open(self):
        if self.cabinet_arm_tag is None:
            return True
        if not self.move(self.open_gripper(self.cabinet_arm_tag)):
            return False

        lateral_xy = self._get_upper_place_lateral_escape_xy(self.cabinet_arm_tag)
        if lateral_xy is None:
            return True
        if abs(float(lateral_xy[0])) <= 1e-9 and abs(float(lateral_xy[1])) <= 1e-9:
            return True
        return bool(
            self.move(
                self.move_by_displacement(
                    arm_tag=self.cabinet_arm_tag,
                    x=float(lateral_xy[0]),
                    y=float(lateral_xy[1]),
                    move_axis="world",
                )
            )
        )

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

    def _is_cabinet_drawer_opened(self):
        if self.initial_drawer_world_point is None:
            return False
        current_drawer_world_point = self._get_drawer_world_point()
        open_dis = float(
            np.linalg.norm(
                current_drawer_world_point[:2] - np.array(self.initial_drawer_world_point[:2], dtype=np.float64)
            )
        )
        return open_dis > float(self.DRAWER_OPEN_SUCCESS_DIS)

    def _get_arm_ee_pose(self, arm_tag):
        if arm_tag == "left":
            return np.array(self.robot.get_left_ee_pose(), dtype=np.float64)
        if arm_tag == "right":
            return np.array(self.robot.get_right_ee_pose(), dtype=np.float64)
        raise ValueError(f'arm_tag must be either "left" or "right", not {arm_tag}')

    def _build_object_grasp_transition_waypoints(self, arm_tag, pre_grasp_pose):
        current_pose = self._get_arm_ee_pose(arm_tag)
        pre_grasp_pose = np.array(pre_grasp_pose, dtype=np.float64)
        cabinet_ref_z = max(
            float(self.cabinet.get_pose().p[2]),
            float(self.object.get_pose().p[2]),
            float(self._get_drawer_world_point()[2]),
        )
        safe_z = max(
            float(current_pose[2]),
            float(pre_grasp_pose[2] + self.OBJECT_APPROACH_PRE_GRASP_MARGIN_Z),
            float(cabinet_ref_z + self.OBJECT_APPROACH_CLEARANCE_Z),
        )

        lift_pose = np.array(current_pose, dtype=np.float64)
        lift_pose[2] = safe_z

        front_pose = np.array(pre_grasp_pose, dtype=np.float64)
        front_pose[2] = safe_z
        return lift_pose.tolist(), front_pose.tolist()

    def _open_cabinet_drawer(self, cabinet_key):
        self.cabinet_arm_tag = self._get_cabinet_arm_tag()
        self.object_arm_tag = self.cabinet_arm_tag.opposite
        self.initial_drawer_world_point = self._get_drawer_world_point()
        self.enter_rotate_action_stage(2, focus_object_key=(cabinet_key or "B"))
        self.face_object_with_torso(self.cabinet, joint_name_prefer=self.SCAN_JOINT_NAME)
        self.move(
            self.grasp_actor(
                self.cabinet,
                arm_tag=self.cabinet_arm_tag,
                pre_grasp_dis=self.CABINET_PRE_GRASP_DIS,
                grasp_dis=self.CABINET_GRASP_DIS,
                gripper_pos=self.CABINET_GRIPPER_POS,
            )
        )
        if not self.plan_success:
            return False

        step_xy = self._get_drawer_pull_step_xy()
        pull_step_limit = max(int(self.DRAWER_PULL_STEPS), 1)
        if not bool(getattr(self, "need_plan", True)):
            # During replay the cabinet arm is only used for drawer pulling after grasping,
            # so the remaining cached plans on that arm are the exact pull budget.
            pull_step_limit = min(
                pull_step_limit,
                int(self._get_remaining_joint_path_count(self.cabinet_arm_tag)),
            )

        executed_pull_steps = 0
        for _ in range(pull_step_limit):
            self.move(
                self.move_by_displacement(
                    arm_tag=self.cabinet_arm_tag,
                    x=float(step_xy[0]),
                    y=float(step_xy[1]),
                )
            )
            executed_pull_steps += 1
            if not self.plan_success:
                break

        drawer_opened = self._is_cabinet_drawer_opened()
        replay_pull_budget_consumed = bool(
            (not bool(getattr(self, "need_plan", True)))
            and pull_step_limit > 0
            and executed_pull_steps >= pull_step_limit
        )
        self.cabinet_opened = bool(self.plan_success and (drawer_opened or replay_pull_budget_consumed))
        if not self.cabinet_opened:
            return False
        return self._retreat_cabinet_arm_after_open()

    def _grasp_and_lift_object(self, object_key):
        if self.object_arm_tag is None:
            self.object_arm_tag = self._get_cabinet_arm_tag().opposite
        self.enter_rotate_action_stage(3, focus_object_key=(object_key or "A"))
        self.face_object_with_torso(self.object, joint_name_prefer=self.SCAN_JOINT_NAME)
        self.initial_object_z = float(self.object.get_pose().p[2])
        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            self.object,
            arm_tag=self.object_arm_tag,
            pre_dis=self.OBJECT_PRE_GRASP_DIS,
            target_dis=self.OBJECT_GRASP_DIS,
        )
        if pre_grasp_pose is None or grasp_pose is None:
            self.plan_success = False
            return False

        lift_pose, front_pose = self._build_object_grasp_transition_waypoints(
            self.object_arm_tag,
            pre_grasp_pose,
        )
        self.move(self.move_to_pose(arm_tag=self.object_arm_tag, target_pose=lift_pose))
        if not self.plan_success:
            return False

        self.move(self.move_to_pose(arm_tag=self.object_arm_tag, target_pose=front_pose))
        if not self.plan_success:
            return False

        self.move(
            (
                self.object_arm_tag,
                [
                    Action(self.object_arm_tag, "move", target_pose=pre_grasp_pose),
                    Action(
                        self.object_arm_tag,
                        "move",
                        target_pose=grasp_pose,
                        constraint_pose=[1, 1, 1, 0, 0, 0],
                    ),
                    Action(self.object_arm_tag, "close", target_gripper_pos=0.0),
                ],
            )
        )
        if not self.plan_success:
            return False
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=self.object_arm_tag, z=self.OBJECT_LIFT_Z))
        if not self.plan_success:
            return False
        self.delay(2)
        return True

    def _build_info(self):
        if self.cabinet_arm_tag is None:
            self.cabinet_arm_tag = self._get_cabinet_arm_tag()
        if self.object_arm_tag is None:
            self.object_arm_tag = self.cabinet_arm_tag.opposite
        return {
            "{A}": str(getattr(self, "object_label", self.OBJECT_LABEL)),
            "{B}": "036_cabinet/base0",
            "{a}": str(self.object_arm_tag),
            "{b}": str(self.cabinet_arm_tag),
        }

    def play_once(self):
        scan_z = float(self.SCAN_Z_BIAS + self.table_z_bias)
        self._reset_head_to_home_pose(save_freq=None)

        object_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is not None:
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(1, carried_after=[])

        self._reset_head_to_home_pose(save_freq=None)
        cabinet_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if cabinet_key is None or not self._open_cabinet_drawer(cabinet_key):
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(2, carried_after=[])

        self._reset_head_to_home_pose(save_freq=None)
        object_key = self.search_and_focus_rotate_subtask(
            3,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is None or not self._grasp_and_lift_object(object_key):
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(3, carried_after=["A"])

        self.info["info"] = self._build_info()
        return self.info

    def check_success(self):
        if self.initial_object_z is None or self.object_arm_tag is None:
            return False
        object_z = float(self.object.get_pose().p[2])
        gripper_close = (
            self.is_left_gripper_close()
            if self.object_arm_tag == "left"
            else self.is_right_gripper_close()
        )
        return bool(object_z > self.initial_object_z + self.SUCCESS_LIFT_Z and gripper_close)

old:
from copy import deepcopy

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .open_carbinet import open_carbinet
from .utils import *


class search_object(Base_Task):
    CABINET_MODEL_ID = 46653
    OBJECT_LABEL = "small block"
    OBJECT_HALF_SIZE = 0.018
    OBJECT_COLOR = (0.10, 0.80, 0.20)
    OBJECT_MASS = 0.03

    CABINET_CYL_R = 0.6
    CABINET_CYL_THETA_DEG_RANGE = (8.0, 24.0)
    CABINET_CYL_Z = 0.741
    CABINET_CYL_SPIN_DEG = 0.0

    TASK_HOMESTATE = [
        [-0.11, -0.7, -0.8, 2, -0.9, 0, 0],
        [0.11, -0.7, 0.8, 2, 0.9, 0, 0],
    ]

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"

    OBJECT_Z_BIAS = OBJECT_HALF_SIZE + 0.002
    OBJECT_OUTER_EDGE_OFFSET = 0.03
    DRAWER_OPEN_SUCCESS_DIS = open_carbinet.DRAWER_OPEN_SUCCESS_DIS
    DRAWER_PULL_TOTAL_DIS = open_carbinet.DRAWER_PULL_TOTAL_DIS
    DRAWER_PULL_STEPS = open_carbinet.DRAWER_PULL_STEPS
    CABINET_PRE_GRASP_DIS = open_carbinet.CABINET_PRE_GRASP_DIS
    CABINET_GRASP_DIS = open_carbinet.CABINET_GRASP_DIS
    CABINET_GRIPPER_POS = open_carbinet.CABINET_GRIPPER_POS
    OBJECT_PRE_GRASP_DIS = 0.09
    OBJECT_GRASP_DIS = 0.01
    OBJECT_APPROACH_CLEARANCE_Z = 0.12
    OBJECT_APPROACH_PRE_GRASP_MARGIN_Z = 0.08
    OBJECT_LIFT_Z = 0.08
    SUCCESS_LIFT_Z = 0.03

    def setup_demo(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("table_shape", "fan")
        kwargs.setdefault("fan_center_on_robot", True)
        kwargs.setdefault("fan_outer_radius", 0.9)
        kwargs.setdefault("fan_inner_radius", 0.3)
        kwargs.setdefault("fan_angle_deg", 150)
        kwargs.setdefault("fan_center_deg", 90)

        for cfg_key in ["left_embodiment_config", "right_embodiment_config"]:
            if cfg_key in kwargs and kwargs[cfg_key] is not None:
                cfg = deepcopy(kwargs[cfg_key])
                cfg["homestate"] = deepcopy(self.TASK_HOMESTATE)
                kwargs[cfg_key] = cfg

        kwargs = init_rotate_theta_bounds(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    @staticmethod
    def _quat_from_yaw(yaw_rad):
        return t3d.euler.euler2quat(0.0, 0.0, float(yaw_rad))

    def _world_xy_from_cyl(self, r, theta_deg):
        theta_rad = float(np.deg2rad(theta_deg))
        phi_world = float(self.robot_yaw + theta_rad)
        x = float(self.robot_root_xy[0] + float(r) * np.cos(phi_world))
        y = float(self.robot_root_xy[1] + float(r) * np.sin(phi_world))
        return x, y

    def _cabinet_pose_from_cyl(self, r, theta_deg, z, spin_deg):
        x, y = self._world_xy_from_cyl(r=r, theta_deg=theta_deg)
        radial_out_yaw = float(np.arctan2(y - self.robot_root_xy[1], x - self.robot_root_xy[0]))
        yaw = float(radial_out_yaw + np.deg2rad(spin_deg))
        return sapien.Pose([x, y, float(z)], self._quat_from_yaw(yaw))

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object,
                "B": self.cabinet,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "search_hidden_object",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "object_not_found",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "open_seen_cabinet",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "cabinet_opened",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "pick_object_from_cabinet",
                    "instruction_idx": 3,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": False,
                    "done_when": "object_grasped_and_lifted",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Search for {A}; if it is not visible, open {B} and pick {A} up.",
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.cabinet_opened = False
        self.object_arm_tag = None
        self.cabinet_arm_tag = None
        self.initial_object_z = None

        cabinet_theta_abs_deg = float(np.random.uniform(*self.CABINET_CYL_THETA_DEG_RANGE))
        self.cabinet_theta_deg = float(cabinet_theta_abs_deg * np.random.choice([-1.0, 1.0]))
        cabinet_pose = self._cabinet_pose_from_cyl(
            r=self.CABINET_CYL_R,
            theta_deg=self.cabinet_theta_deg,
            z=self.CABINET_CYL_Z,
            spin_deg=self.CABINET_CYL_SPIN_DEG,
        )
        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            pose=cabinet_pose,
            modelname="036_cabinet",
            modelid=self.CABINET_MODEL_ID,
            fix_root_link=True,
        )

        drawer_pose = self.cabinet.get_functional_point(0, "pose")
        drawer_outward_dir = self._get_drawer_outward_dir_xy()
        object_pose = sapien.Pose(
            np.array(drawer_pose.p, dtype=np.float64)
            + np.array(
                [
                    float(drawer_outward_dir[0]) * float(self.OBJECT_OUTER_EDGE_OFFSET),
                    float(drawer_outward_dir[1]) * float(self.OBJECT_OUTER_EDGE_OFFSET),
                    float(self.OBJECT_Z_BIAS - self.table_z_bias),
                ],
                dtype=np.float64,
            ),
            np.array(drawer_pose.q, dtype=np.float64),
        )
        self.object = create_box(
            scene=self,
            pose=object_pose,
            half_size=(self.OBJECT_HALF_SIZE, self.OBJECT_HALF_SIZE, self.OBJECT_HALF_SIZE),
            color=self.OBJECT_COLOR,
            name="search_object_block",
        )
        self.object.set_mass(float(self.OBJECT_MASS))
        self.initial_drawer_world_point = self._get_drawer_world_point()

        self.add_prohibit_area(self.cabinet, padding=0.03)
        self.add_prohibit_area(self.object, padding=0.03)
        self._configure_rotate_subtask_plan()

    def _project_rotate_registry_object(self, object_key, camera_pose=None, camera_spec=None):
        if str(object_key) == "A" and not bool(getattr(self, "cabinet_opened", False)):
            return None
        return super()._project_rotate_registry_object(
            object_key,
            camera_pose=camera_pose,
            camera_spec=camera_spec,
        )

    def _get_cabinet_arm_tag(self):
        cabinet_cyl = world_to_robot(self.cabinet.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if float(cabinet_cyl[1]) >= 0.0 else "right")

    def _get_drawer_outward_dir_xy(self):
        cabinet_xy = np.array(self.cabinet.get_pose().p[:2], dtype=np.float64)
        robot_xy = np.array(self.robot_root_xy, dtype=np.float64)
        direction = robot_xy - cabinet_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return np.array([0.0, -1.0], dtype=np.float64)
        return direction / norm

    def _get_drawer_world_point(self):
        return np.array(self.cabinet.get_functional_point(0)[:3], dtype=np.float64)

    def _get_drawer_pull_step_xy(self):
        direction = self._get_drawer_outward_dir_xy()
        step_dis = float(self.DRAWER_PULL_TOTAL_DIS) / float(max(int(self.DRAWER_PULL_STEPS), 1))
        return (direction * step_dis).tolist()

    def _is_cabinet_drawer_opened(self):
        if self.initial_drawer_world_point is None:
            return False
        current_drawer_world_point = self._get_drawer_world_point()
        open_dis = float(
            np.linalg.norm(
                current_drawer_world_point[:2] - np.array(self.initial_drawer_world_point[:2], dtype=np.float64)
            )
        )
        return open_dis > float(self.DRAWER_OPEN_SUCCESS_DIS)

    def _get_arm_ee_pose(self, arm_tag):
        if arm_tag == "left":
            return np.array(self.robot.get_left_ee_pose(), dtype=np.float64)
        if arm_tag == "right":
            return np.array(self.robot.get_right_ee_pose(), dtype=np.float64)
        raise ValueError(f'arm_tag must be either "left" or "right", not {arm_tag}')

    def _build_object_grasp_transition_waypoints(self, arm_tag, pre_grasp_pose):
        current_pose = self._get_arm_ee_pose(arm_tag)
        pre_grasp_pose = np.array(pre_grasp_pose, dtype=np.float64)
        cabinet_ref_z = max(
            float(self.cabinet.get_pose().p[2]),
            float(self.object.get_pose().p[2]),
            float(self._get_drawer_world_point()[2]),
        )
        safe_z = max(
            float(current_pose[2]),
            float(pre_grasp_pose[2] + self.OBJECT_APPROACH_PRE_GRASP_MARGIN_Z),
            float(cabinet_ref_z + self.OBJECT_APPROACH_CLEARANCE_Z),
        )

        lift_pose = np.array(current_pose, dtype=np.float64)
        lift_pose[2] = safe_z

        front_pose = np.array(pre_grasp_pose, dtype=np.float64)
        front_pose[2] = safe_z
        return lift_pose.tolist(), front_pose.tolist()

    def _open_cabinet_drawer(self, cabinet_key):
        self.cabinet_arm_tag = self._get_cabinet_arm_tag()
        self.object_arm_tag = self.cabinet_arm_tag.opposite
        self.initial_drawer_world_point = self._get_drawer_world_point()
        self.enter_rotate_action_stage(2, focus_object_key=(cabinet_key or "B"))
        self.face_object_with_torso(self.cabinet, joint_name_prefer=self.SCAN_JOINT_NAME)
        self.move(
            self.grasp_actor(
                self.cabinet,
                arm_tag=self.cabinet_arm_tag,
                pre_grasp_dis=self.CABINET_PRE_GRASP_DIS,
                grasp_dis=self.CABINET_GRASP_DIS,
                gripper_pos=self.CABINET_GRIPPER_POS,
            )
        )
        if not self.plan_success:
            return False

        step_xy = self._get_drawer_pull_step_xy()
        pull_step_limit = max(int(self.DRAWER_PULL_STEPS), 1)
        if not bool(getattr(self, "need_plan", True)):
            # During replay the cabinet arm is only used for drawer pulling after grasping,
            # so the remaining cached plans on that arm are the exact pull budget.
            pull_step_limit = min(
                pull_step_limit,
                int(self._get_remaining_joint_path_count(self.cabinet_arm_tag)),
            )

        executed_pull_steps = 0
        for _ in range(pull_step_limit):
            self.move(
                self.move_by_displacement(
                    arm_tag=self.cabinet_arm_tag,
                    x=float(step_xy[0]),
                    y=float(step_xy[1]),
                )
            )
            executed_pull_steps += 1
            if not self.plan_success:
                break
            if self.need_plan and self._is_cabinet_drawer_opened():
                break

        drawer_opened = self._is_cabinet_drawer_opened()
        replay_pull_budget_consumed = bool(
            (not bool(getattr(self, "need_plan", True)))
            and pull_step_limit > 0
            and executed_pull_steps >= pull_step_limit
        )
        self.cabinet_opened = bool(self.plan_success and (drawer_opened or replay_pull_budget_consumed))
        return self.cabinet_opened

    def _grasp_and_lift_object(self, object_key):
        if self.object_arm_tag is None:
            self.object_arm_tag = self._get_cabinet_arm_tag().opposite
        self.enter_rotate_action_stage(3, focus_object_key=(object_key or "A"))
        self.face_object_with_torso(self.object, joint_name_prefer=self.SCAN_JOINT_NAME)
        self.initial_object_z = float(self.object.get_pose().p[2])
        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            self.object,
            arm_tag=self.object_arm_tag,
            pre_dis=self.OBJECT_PRE_GRASP_DIS,
            target_dis=self.OBJECT_GRASP_DIS,
        )
        if pre_grasp_pose is None or grasp_pose is None:
            self.plan_success = False
            return False

        lift_pose, front_pose = self._build_object_grasp_transition_waypoints(
            self.object_arm_tag,
            pre_grasp_pose,
        )
        self.move(self.move_to_pose(arm_tag=self.object_arm_tag, target_pose=lift_pose))
        if not self.plan_success:
            return False

        self.move(self.move_to_pose(arm_tag=self.object_arm_tag, target_pose=front_pose))
        if not self.plan_success:
            return False

        self.move(
            (
                self.object_arm_tag,
                [
                    Action(self.object_arm_tag, "move", target_pose=pre_grasp_pose),
                    Action(
                        self.object_arm_tag,
                        "move",
                        target_pose=grasp_pose,
                        constraint_pose=[1, 1, 1, 0, 0, 0],
                    ),
                    Action(self.object_arm_tag, "close", target_gripper_pos=0.0),
                ],
            )
        )
        if not self.plan_success:
            return False
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=self.object_arm_tag, z=self.OBJECT_LIFT_Z))
        if not self.plan_success:
            return False
        self.delay(2)
        return True

    def _build_info(self):
        if self.cabinet_arm_tag is None:
            self.cabinet_arm_tag = self._get_cabinet_arm_tag()
        if self.object_arm_tag is None:
            self.object_arm_tag = self.cabinet_arm_tag.opposite
        return {
            "{A}": self.OBJECT_LABEL,
            "{B}": "036_cabinet/base0",
            "{a}": str(self.object_arm_tag),
            "{b}": str(self.cabinet_arm_tag),
        }

    def play_once(self):
        scan_z = float(self.SCAN_Z_BIAS + self.table_z_bias)
        self._reset_head_to_home_pose(save_freq=None)

        object_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is not None:
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(1, carried_after=[])

        self._reset_head_to_home_pose(save_freq=None)
        cabinet_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if cabinet_key is None or not self._open_cabinet_drawer(cabinet_key):
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(2, carried_after=[])

        self._reset_head_to_home_pose(save_freq=None)
        object_key = self.search_and_focus_rotate_subtask(
            3,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is None or not self._grasp_and_lift_object(object_key):
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(3, carried_after=["A"])

        self.info["info"] = self._build_info()
        return self.info

    def check_success(self):
        if self.initial_object_z is None or self.object_arm_tag is None:
            return False
        object_z = float(self.object.get_pose().p[2])
        gripper_close = (
            self.is_left_gripper_close()
            if self.object_arm_tag == "left"
            else self.is_right_gripper_close()
        )
        return bool(object_z > self.initial_object_z + self.SUCCESS_LIFT_Z and gripper_close)

