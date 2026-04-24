from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d


class put_block_on(Base_Task):
    # 坐标约定：
    # 这里的 cyl 参数都使用机器人根部为圆心的柱坐标。
    # r 表示水平半径，theta_deg=0 表示机器人初始正前方，z 表示世界坐标高度。
    #
    # block 生成参数：
    # BLOCK_COUNT 手动控制生成 1/2/3 个 block。
    # BLOCK_LAYER_SEQUENCE 显式决定每个 block 的层，长度必须等于 BLOCK_COUNT。
    # BLOCK_LAYER_SPECS 还支持直接覆盖采样区间：
    #   BLOCK_LAYER_SPECS["lower"]["r_range"] = [0.42, 0.58]
    #   BLOCK_LAYER_SPECS["lower"]["theta_deg_range"] = [15, 70]
    #   BLOCK_LAYER_SPECS["upper"]["r_range"] = [0.66, 0.70]
    #   BLOCK_LAYER_SPECS["upper"]["theta_deg_range"] = [-12, 12]
    BLOCK_COUNT = 2  # 生成的 block 数量，支持 1/2/3。
    BLOCK_LAYER_SEQUENCE = ("upper", "upper")  # 每个 block 所在层，按生成顺序对应。
    BLOCK_SIZE_RANGE = (0.015, 0.025)  # 单个 block half_size 的随机范围，单位米。
    BLOCK_COLOR = (0.10, 0.80, 0.20)  # block 默认颜色，RGB 归一化。
    BLOCK_SPAWN_MIN_DIST_SQ = 0.01  # block 与 block 的最小平面距离平方。
    PLATE_BLOCK_SPAWN_MIN_DIST_SQ = 0.0255  # block 与 plate anchor 的最小平面距离平方。
    BLOCK_LAYER_SPECS = {
        "lower": {
            "r_range": [0.4, 0.5],  # 下层 block 的默认半径随机区间，单位米。
            "theta_deg_range": [15.0, 60.0],  # 下层 block 的默认角度随机区间，单位度。
        },
        "upper": {
            "r_range": [0.66, 0.70],  # 上层 block 的默认半径随机区间，单位米。
            "theta_deg_range": [-6.0, 6.0],  # 上层 block 的默认角度随机区间，单位度。
        },
    }

    # plate anchor 参数：
    # plate 的 z 由对应桌面 top_z + z_offset 计算，避免和 fan_double_layer_gap 不一致。
    PLATE_MODEL_ID = 0  # 使用的 plate 资产 model_id。
    PLATE_LAYER = "lower"  # plate 所在层，会影响 block 需要避让哪一层。
    PLATE_LAYER_SPECS = {
        "lower": {
            "r": 0.5,  # lower plate 的柱坐标半径。
            "theta_deg": -20,  # lower plate 的柱坐标角度，0 表示机器人正前方。
            "z_offset": 0.0,  # lower plate 相对该层桌面的额外高度偏置。
            "qpos": [0.5, 0.5, 0.5, 0.5],  # lower plate 初始姿态四元数。
            "scale": [0.025, 0.025, 0.025],  # lower plate 模型缩放。
        },
        "upper": {
            "r": 0.70,  # upper plate 的柱坐标半径。
            "theta_deg": 0,  # upper plate 的柱坐标角度。
            "z_offset": 0.0,  # upper plate 相对该层桌面的额外高度偏置。
            "qpos": [0.5, 0.5, 0.5, 0.5],  # upper plate 初始姿态四元数。
            "scale": [0.025, 0.025, 0.025],  # upper plate 模型缩放。
        },
    }
    # plate 内部的目标槽位，单位为相对 plate 中心的 [radial_offset, tangential_offset] 米。
    # radial: 机器人到 plate 的径向；tangential: 桌面内与其垂直的切向。
    PLATE_PLACE_SLOT_OFFSETS = {
        1: ((0.0, 0.0),),  # 单 block 时放在 plate 中心。
        2: ((0.0, -0.03), (0.0, 0.03)),  # 双 block 时沿切向分开。
        3: ((-0.045, 0.0), (0.015, 0.050), (0.015, -0.050)),  # 三 block 时优先拉开前两次放置的间距。
    }

    # 搜索参数：
    # scan_r/scan_z 决定模拟搜索时腰部对准的观察点，stage 规则在 Base_Task 中统一实现。
    SCAN_R = 0.62  # 搜索时观察点的柱坐标半径。
    SCAN_Z_BIAS = 0.90  # 搜索时观察点相对世界的默认高度。
    SCAN_JOINT_NAME = "astribot_torso_joint_2"  # 搜索优先使用的腰部关节。
    PLACE_PLATE_UPPER_HEAD_JOINT2_TARGET = 0.8  # 看 upper plate 时 head_joint2 的固定值。
    PLACE_PLATE_LOWER_HEAD_JOINT2_TARGET = None  # 看 lower plate 时 head_joint2 的固定值；None 表示沿用默认。
    REQUIRE_PLATE_VISIBLE_BEFORE_PLACE = True  # 放置前是否必须重新看到 plate。
    # 重新低头搜索时保存 head 运动过程，避免视频里相机视角瞬移。
    HEAD_RESET_SAVE_FREQ = -1  # 头部复位录制频率；-1 表示按 move 默认保存。

    # 抓取参数：
    # 先到 pre-grasp，再前进到 grasp，闭合后竖直抬升，尽量保证拿稳。
    PICK_PRE_GRASP_DIS = 0.09  # lower grasp_actor 的预抓取后退距离。
    PICK_GRASP_DIS = 0.01  # lower grasp_actor 最后推进的抓取距离。
    PICK_LIFT_Z = 0.10  # 抓取后首次抬升的高度。
    POST_GRASP_EXTRA_LIFT_Z = 0.04  # 抓稳后额外再抬升的高度。

    # 初始手臂姿态微调，降低抓取后处在极限姿态的概率。
    INITIAL_LEFT_ARM_JOINT1 = -0.110  # 左臂 homestate 的 joint1 微调值。
    INITIAL_RIGHT_ARM_JOINT1 = 0.110  # 右臂 homestate 的 joint1 微调值。

    # direct release 参数：
    # TCP 是夹爪工作点；planner target 需要沿 TCP 局部 x 轴后退 DIRECT_RELEASE_TCP_BACKOFF。
    # release/entry/approach 都基于当前 plate 的柱坐标和层高度动态生成。
    DIRECT_RELEASE_TCP_BACKOFF = 0.12  # direct release 时 planner pose 相对 TCP 沿局部 x 轴的后退量。
    DIRECT_RELEASE_ENTRY_TCP_CYL_R = None  # direct release 的入口半径；None 表示自动推算。
    DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER = 0.08  # 自动推算入口半径时，相对 upper 内圈保留的裕量。
    DIRECT_RELEASE_TCP_Z_OFFSET = 0.08  # 松手位相对目标点的 TCP 高度偏置。
    DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET = 0.10  # 入口位相对目标点的 TCP 高度偏置。
    DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET = 0.10  # approach 位相对目标点的 TCP 高度偏置。
    DIRECT_RELEASE_RETREAT_Z = 0.08  # 松手后先竖直后撤的高度。
    DIRECT_RELEASE_R_OFFSETS = (0.0, -0.03, 0.03)  # 枚举候选 release pose 的半径偏移。
    DIRECT_RELEASE_THETA_OFFSETS_DEG = (0.0, -3.0, 3.0)  # 枚举候选 release pose 的角度偏移。
    DIRECT_RELEASE_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0)  # 枚举候选 release pose 的末端 yaw 偏移。
    PLACE_RETREAT_Z = 0.12  # lower place_actor 放置后沿 arm 轴后撤的距离。
    LOWER_PLACE_PRE_DIS = 0.18  # lower place_actor 的预放置距离，恢复到最初的桌面 place_actor 配置。
    LOWER_PLACE_DIS = 0.03  # lower place_actor 的最终推进距离，恢复到最初配置。
    LOWER_PLACE_CONSTRAIN = "free"  # lower place_actor 的姿态约束模式，恢复到最初的 align 放置。
    LOWER_PLACE_PRE_DIS_AXIS = "fp"  # lower place_actor 的接近轴，恢复到最初沿当前抓取关系接近。
    UPPER_TO_LOWER_TOP_DOWN_PLACE_ENABLED = False  # 上层抓取后放下层时，是否先尝试空中转成 top-down 再竖直下放。
    UPPER_TO_LOWER_TOP_DOWN_CARRY_Z_OFFSET = 0.15  # upper->lower top-down 过渡位相对目标点的额外高度。
    UPPER_TO_LOWER_TOP_DOWN_PRE_DIS = 0.10  # upper->lower top-down 预放置高度。
    UPPER_TO_LOWER_TOP_DOWN_DIS = 0.03  # upper->lower top-down 最终下放高度。
    UPPER_TO_LOWER_TOP_DOWN_YAW_OFFSETS_DEG = (0.0, 90.0, -90.0, 180.0)  # upper->lower top-down 枚举的法向 yaw 候选。

    # 上层 block 抓取参数：
    # 由于上层较远，采用和 direct release 一致的 TCP->planner pose 语义直接 move。
    UPPER_PICK_ENTRY_Z_OFFSET = 0.08  # upper pick 的 entry 位相对抓取点的高度偏置。
    UPPER_PICK_PRE_GRASP_DIS = 0.10  # upper pick 的预抓取后退距离。
    UPPER_PICK_GRASP_Z_BIAS = 0.0  # upper pick 抓取点的额外 z 偏置。
    UPPER_PICK_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)  # upper pick 枚举的末端 yaw 候选。
    UPPER_PICK_GRIPPER_POS = -0.01  # upper pick 闭夹爪时的目标 gripper 开度。
    UPPER_PICK_POST_ENTRY_RETREAT_DIS = 0.05  # upper pick 回到 entry 后再水平后撤的距离。

    # 放置后是否直接用 move_joint 回到 homestate。
    # True: 松手后先竖直抬起，再 back_to_origin(left/right)。
    # False: 松手后只竖直抬起，不执行 homestate 回收。
    POST_PLACE_LATERAL_ESCAPE_DIS = 0.2  # upper place 后的侧向逃逸距离。
    RETURN_TO_HOMESTATE_AFTER_PLACE = True  # 放置完成后是否回双臂 home。

    # 成功判定参数：
    # 多 block 时要求每个 block 都落到 plate functional point 附近，且夹爪打开。
    KNOWN_FIXED_TARGET_KEYS = ()  # 开局就标记为“已知”的固定目标 key。
    SUCCESS_EPS = np.array([0.08, 0.08, 0.08], dtype=np.float64)  # 成功判定时 block 到 plate 中心的容差。

    def setup_demo(self, **kwargs):
        self._apply_block_spawn_overrides_from_kwargs(kwargs)
        kwargs.setdefault("table_shape", "fan_double")
        kwargs.setdefault("fan_center_on_robot", True)
        kwargs.setdefault("fan_double_lower_outer_radius", 0.9)
        kwargs.setdefault("fan_double_lower_inner_radius", 0.3)
        kwargs.setdefault("fan_double_upper_outer_radius", 0.8)
        kwargs.setdefault("fan_double_upper_inner_radius", 0.6)
        kwargs.setdefault("fan_double_layer_gap", 0.35)
        kwargs.setdefault("fan_double_upper_theta_start_deg", -30.0)
        kwargs.setdefault("fan_double_upper_theta_end_deg", 30.0)
        kwargs.setdefault("fan_double_support_theta_deg", -40.0)
        kwargs.setdefault("fan_angle_deg", 150)
        kwargs.setdefault("fan_center_deg", 90)
        kwargs = init_rotate_theta_bounds(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _apply_block_spawn_overrides_from_kwargs(self, kwargs):
        for layer_name in ("lower", "upper"):
            for suffix in ("r_range", "theta_deg_range"):
                attr_name = f"block_spawn_{layer_name}_{suffix}"
                range_values = kwargs.get(attr_name, None)
                if range_values is None:
                    continue
                normalized = np.array(range_values, dtype=np.float64).reshape(-1).tolist()
                setattr(self, attr_name, normalized)

    def _get_block_count(self):
        block_count = int(self.BLOCK_COUNT)
        if block_count not in (1, 2, 3):
            raise ValueError(f"BLOCK_COUNT must be 1, 2, or 3, got {block_count}")
        return block_count

    def _normalize_layer(self, layer_name):
        layer_name = str(layer_name).lower()
        if layer_name not in ("lower", "upper"):
            raise ValueError(f"Layer must be 'lower' or 'upper', got {layer_name}")
        return layer_name

    def _get_block_layers(self):
        block_count = self._get_block_count()
        layers = tuple(self._normalize_layer(layer) for layer in self.BLOCK_LAYER_SEQUENCE)
        if len(layers) != block_count:
            raise ValueError(
                f"BLOCK_LAYER_SEQUENCE length must equal BLOCK_COUNT: "
                f"{len(layers)} != {block_count}"
            )
        return layers

    def _get_plate_layer(self):
        return self._normalize_layer(self.PLATE_LAYER)

    def _configure_rotate_subtask_plan(self):
        object_registry = {key: block for key, block in zip(self.block_keys, self.blocks)}
        object_registry["B"] = self.plate

        subtask_defs = []
        for block_idx, block_key in enumerate(self.block_keys):
            pick_subtask_id = 2 * block_idx + 1
            place_subtask_id = pick_subtask_id + 1
            next_subtask_id = place_subtask_id + 1 if block_idx < len(self.block_keys) - 1 else -1
            subtask_defs.extend(
                [
                    {
                        "id": pick_subtask_id,
                        "name": f"pick_remaining_block_{block_idx}",
                        "instruction_idx": 1,
                        "search_target_keys": list(self.block_keys),
                        "action_target_keys": list(self.block_keys),
                        "required_carried_keys": [],
                        "carry_keys_after_done": [],
                        "allow_stage2_from_memory": False,
                        "done_when": "selected_block_grasped",
                        "next_subtask_id": place_subtask_id,
                    },
                    {
                        "id": place_subtask_id,
                        "name": f"place_selected_block_{block_idx}_into_plate",
                        "instruction_idx": 2,
                        "search_target_keys": ["B"],
                        "action_target_keys": ["B"],
                        "required_carried_keys": [],
                        "carry_keys_after_done": [],
                        "allow_stage2_from_memory": True,
                        "done_when": "selected_block_in_plate",
                        "next_subtask_id": next_subtask_id,
                    },
                ]
            )

        self.configure_rotate_subtask_plan(
            object_registry=object_registry,
            subtask_defs=subtask_defs,
            task_instruction="Put the block into {B}." if len(self.block_keys) == 1 else "Put all blocks into {B}.",
        )

    def _prepare_dynamic_pick_subtask(self, subtask_idx, remaining_block_keys):
        remaining_block_keys = [str(key) for key in remaining_block_keys]
        subtask_def = self._get_rotate_subtask_def(subtask_idx)
        if subtask_def is None:
            raise ValueError(f"Unknown rotate subtask id: {subtask_idx}")
        subtask_def["search_target_keys"] = remaining_block_keys
        subtask_def["action_target_keys"] = remaining_block_keys
        subtask_def["required_carried_keys"] = []
        subtask_def["carry_keys_after_done"] = []
        subtask_def["allow_stage2_from_memory"] = len(remaining_block_keys) == 1
        return subtask_def

    def _prepare_dynamic_place_subtask(self, subtask_idx, block_key):
        block_key = str(block_key)
        subtask_def = self._get_rotate_subtask_def(subtask_idx)
        if subtask_def is None:
            raise ValueError(f"Unknown rotate subtask id: {subtask_idx}")
        subtask_def["search_target_keys"] = ["B"]
        subtask_def["action_target_keys"] = [block_key, "B"]
        subtask_def["required_carried_keys"] = [block_key]
        subtask_def["carry_keys_after_done"] = []
        subtask_def["allow_stage2_from_memory"] = True
        return subtask_def

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    def _normalize_closed_range(self, range_values, range_name, min_allowed=None):
        if range_values is None:
            return None
        arr = np.array(range_values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 2 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{range_name} must contain exactly 2 finite values, got {range_values}")
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if min_allowed is not None and lo < float(min_allowed):
            raise ValueError(f"{range_name} must be >= {float(min_allowed)}, got {range_values}")
        return [lo, hi]

    def _get_block_spawn_range_override(self, layer_name, block_spec, attr_suffix, spec_key, min_allowed=None):
        layer_name = self._normalize_layer(layer_name)
        attr_name = f"block_spawn_{layer_name}_{attr_suffix}"
        range_values = getattr(self, attr_name, None)
        if range_values is None:
            range_values = block_spec.get(spec_key, None)
        return self._normalize_closed_range(range_values, attr_name if range_values is not None else spec_key, min_allowed)

    def _get_layer_spec(self, layer_name):
        layer_name = self._normalize_layer(layer_name)
        if layer_name == "upper":
            inner_radius = float(
                getattr(self, "rotate_fan_double_upper_inner_radius", getattr(self, "rotate_fan_inner_radius", 0.3))
            )
            outer_radius = float(
                getattr(self, "rotate_fan_double_upper_outer_radius", getattr(self, "rotate_fan_outer_radius", 0.8))
            )
            top_z = float(getattr(self, "rotate_table_top_z", 0.74)) + float(
                getattr(self, "rotate_fan_double_layer_gap", 0.40)
            )
            upper_theta_start = getattr(self, "rotate_fan_double_upper_theta_start_world_rad", None)
            upper_theta_end = getattr(self, "rotate_fan_double_upper_theta_end_world_rad", None)
        else:
            inner_radius = float(
                getattr(self, "rotate_fan_double_lower_inner_radius", getattr(self, "rotate_fan_inner_radius", 0.3))
            )
            outer_radius = float(
                getattr(self, "rotate_fan_double_lower_outer_radius", getattr(self, "rotate_fan_outer_radius", 0.8))
            )
            top_z = float(getattr(self, "rotate_table_top_z", 0.74))
            upper_theta_start = None
            upper_theta_end = None

        block_spec = dict(self.BLOCK_LAYER_SPECS.get(layer_name, {}))
        inner_margin = float(block_spec.get("inner_margin", 0.10))
        outer_margin = float(block_spec.get("outer_margin", 0.10))
        max_cyl_r = float(block_spec.get("max_cyl_r", outer_radius - outer_margin))
        theta_shrink = float(block_spec.get("theta_shrink", 0.92))
        direct_r_range = self._get_block_spawn_range_override(
            layer_name,
            block_spec,
            attr_suffix="r_range",
            spec_key="r_range",
            min_allowed=0.0,
        )
        direct_theta_deg_range = self._get_block_spawn_range_override(
            layer_name,
            block_spec,
            attr_suffix="theta_deg_range",
            spec_key="theta_deg_range",
        )

        if direct_r_range is not None:
            r_min, r_max = direct_r_range
        else:
            r_min = min(max(inner_radius + inner_margin, inner_radius + 0.05), outer_radius - 0.08)
            r_cap = min(max_cyl_r, outer_radius - outer_margin)
            r_max = max(r_min, r_cap)

        if direct_theta_deg_range is not None:
            thetalim = [float(np.deg2rad(direct_theta_deg_range[0])), float(np.deg2rad(direct_theta_deg_range[1]))]
        elif (
            layer_name == "upper"
            and upper_theta_start is not None
            and upper_theta_end is not None
            and hasattr(self, "robot_yaw")
        ):
            theta_start = float(self._wrap_to_pi(float(upper_theta_start) - float(self.robot_yaw)))
            theta_end = float(self._wrap_to_pi(float(upper_theta_end) - float(self.robot_yaw)))
            thetalim = [min(theta_start, theta_end), max(theta_start, theta_end)]
        else:
            theta_half = float(rotate_theta_half(self)) * theta_shrink
            if theta_half <= 1e-3:
                theta_half = float(getattr(self, "rotate_object_theta_half_rad", np.deg2rad(45.0))) * 0.8
            thetalim = [-float(theta_half), float(theta_half)]

        return {
            "layer": layer_name,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "rlim": [float(r_min), float(r_max)],
            "thetalim": thetalim,
            "top_z": top_z,
        }

    def _apply_task_initial_homestate(self):
        left_homestate = list(getattr(self.robot, "left_homestate", []))
        right_homestate = list(getattr(self.robot, "right_homestate", []))
        if len(left_homestate) > 0:
            left_homestate[0] = float(self.INITIAL_LEFT_ARM_JOINT1)
            self.robot.left_homestate = left_homestate
        if len(right_homestate) > 0:
            right_homestate[0] = float(self.INITIAL_RIGHT_ARM_JOINT1)
            self.robot.right_homestate = right_homestate

    @staticmethod
    def _valid_spacing(new_pose, existing_pose_lst, min_dist_sq):
        for pose in existing_pose_lst:
            if np.sum(np.square(new_pose.p[:2] - pose.p[:2])) < float(min_dist_sq):
                return False
        return True

    def _is_valid_block_spawn_pose(
        self,
        block_pose,
        existing_pose_lst=None,
        avoid_pose_lst=None,
        avoid_min_dist_sq=None,
    ):
        if existing_pose_lst is None:
            existing_pose_lst = []
        if avoid_pose_lst is None:
            avoid_pose_lst = []
        if not self._valid_spacing(block_pose, existing_pose_lst, self.BLOCK_SPAWN_MIN_DIST_SQ):
            return False
        if len(avoid_pose_lst) > 0:
            min_dist_sq = self.PLATE_BLOCK_SPAWN_MIN_DIST_SQ if avoid_min_dist_sq is None else avoid_min_dist_sq
            if not self._valid_spacing(block_pose, avoid_pose_lst, min_dist_sq):
                return False
        return True

    def _sample_block_pose(self, layer_name, size, existing_pose_lst=None, avoid_pose_lst=None, avoid_min_dist_sq=None):
        if existing_pose_lst is None:
            existing_pose_lst = []
        if avoid_pose_lst is None:
            avoid_pose_lst = []
        layer_spec = self._get_layer_spec(layer_name)
        spawn_rlim = [float(layer_spec["rlim"][0]), float(layer_spec["rlim"][1])]
        spawn_thetalim = [float(layer_spec["thetalim"][0]), float(layer_spec["thetalim"][1])]
        spawn_z = float(layer_spec["top_z"] + float(size))
        for _ in range(120):
            block_pose = rand_pose_cyl(
                rlim=spawn_rlim,
                thetalim=spawn_thetalim,
                zlim=[spawn_z, spawn_z],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.75],
            )
            block_cyl = world_to_robot(block_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if block_cyl[0] < spawn_rlim[0] - 1e-6 or block_cyl[0] > spawn_rlim[1] + 1e-6:
                continue
            if not self._is_valid_block_spawn_pose(
                block_pose,
                existing_pose_lst=existing_pose_lst,
                avoid_pose_lst=avoid_pose_lst,
                avoid_min_dist_sq=avoid_min_dist_sq,
            ):
                continue
            return block_pose

        fallback_r_candidates = [
            float(np.clip(np.mean(spawn_rlim), spawn_rlim[0], spawn_rlim[1])),
            float(spawn_rlim[0]),
            float(spawn_rlim[1]),
        ]
        fallback_theta_candidates = [
            float(np.clip(np.mean(spawn_thetalim), spawn_thetalim[0], spawn_thetalim[1])),
            float(spawn_thetalim[0]),
            float(spawn_thetalim[1]),
        ]
        for fallback_r in fallback_r_candidates:
            for fallback_theta in fallback_theta_candidates:
                block_pose = rand_pose_cyl(
                    rlim=[fallback_r, fallback_r],
                    thetalim=[fallback_theta, fallback_theta],
                    zlim=[spawn_z, spawn_z],
                    robot_root_xy=self.robot_root_xy,
                    robot_yaw_rad=self.robot_yaw,
                    qpos=[1, 0, 0, 0],
                    rotate_rand=False,
                )
                if self._is_valid_block_spawn_pose(
                    block_pose,
                    existing_pose_lst=existing_pose_lst,
                    avoid_pose_lst=avoid_pose_lst,
                    avoid_min_dist_sq=avoid_min_dist_sq,
                ):
                    return block_pose

        return rand_pose_cyl(
            rlim=[fallback_r_candidates[0], fallback_r_candidates[0]],
            thetalim=[fallback_theta_candidates[0], fallback_theta_candidates[0]],
            zlim=[spawn_z, spawn_z],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

    def _get_plate_layer_spec(self, layer_name=None):
        layer_name = self._get_plate_layer() if layer_name is None else self._normalize_layer(layer_name)
        plate_spec = dict(self.PLATE_LAYER_SPECS.get(layer_name, {}))
        if len(plate_spec) == 0:
            raise ValueError(f"Missing PLATE_LAYER_SPECS entry for layer: {layer_name}")
        layer_spec = self._get_layer_spec(layer_name)
        return {
            "layer": layer_name,
            "r": float(plate_spec.get("r", 0.70 if layer_name == "upper" else 0.55)),
            "theta_deg": float(plate_spec.get("theta_deg", 0.0)),
            "z": float(layer_spec["top_z"]) + float(plate_spec.get("z_offset", 0.0)),
            "qpos": list(plate_spec.get("qpos", [0.5, 0.5, 0.5, 0.5])),
            "scale": list(plate_spec.get("scale", [0.025, 0.025, 0.025])),
        }

    def _get_plate_anchor_pose(self, layer_name=None):
        plate_spec = self._get_plate_layer_spec(layer_name)
        return place_pose_cyl(
            [
                float(plate_spec["r"]),
                float(np.deg2rad(float(plate_spec["theta_deg"]))),
                float(plate_spec["z"]),
            ] + list(plate_spec["qpos"]),
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="pose",
        )

    def _get_block_spawn_avoid_pose_lst(self, layer_name):
        layer_name = self._normalize_layer(layer_name)
        plate_layer = self._get_plate_layer()
        if layer_name != plate_layer:
            return []
        return [self._get_plate_anchor_pose(plate_layer)]

    def _create_plate_anchor(self):
        self.plate_layer = self._get_plate_layer()
        plate_spec = self._get_plate_layer_spec(self.plate_layer)
        self.plate_cyl_r = float(plate_spec["r"])
        self.plate_cyl_theta_deg = float(plate_spec["theta_deg"])
        self.plate_z = float(plate_spec["z"])
        self.plate_qpos = list(plate_spec["qpos"])
        self.plate_scale = list(plate_spec["scale"])

        plate_pose = self._get_plate_anchor_pose(self.plate_layer)
        self.plate = create_actor(
            self,
            pose=plate_pose,
            modelname="003_plate",
            model_id=self.PLATE_MODEL_ID,
            scale=self.plate_scale,
            is_static=True,
            convex=True,
        )
        self.plate_target_pose = self.plate.get_functional_point(0)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self._apply_task_initial_homestate()

        self.block_count = self._get_block_count()
        self.block_layers = list(self._get_block_layers())
        self.blocks = []
        self.block_keys = []
        self.block_sizes = []
        self.block_poses = []
        existing_pose_lst = []
        for block_idx in range(self.block_count):
            block_key = f"A{block_idx}"
            block_layer = self.block_layers[block_idx]
            block_size = float(np.random.uniform(*self.BLOCK_SIZE_RANGE))
            avoid_pose_lst = self._get_block_spawn_avoid_pose_lst(block_layer)
            block_pose = self._sample_block_pose(
                layer_name=block_layer,
                size=block_size,
                existing_pose_lst=existing_pose_lst,
                avoid_pose_lst=avoid_pose_lst,
                avoid_min_dist_sq=self.PLATE_BLOCK_SPAWN_MIN_DIST_SQ,
            )
            block = create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_size, block_size, block_size),
                color=self.BLOCK_COLOR,
                name=f"block_{block_idx}",
            )
            block.set_mass(0.03)
            self.blocks.append(block)
            self.block_keys.append(block_key)
            self.block_sizes.append(block_size)
            self.block_poses.append(block_pose)
            existing_pose_lst.append(block_pose)

        # 兼容旧调试代码：单 block 入口仍指向第一个 block。
        self.block = self.blocks[0]
        self.block_size = self.block_sizes[0]
        self.block_pose = self.block_poses[0]

        self._create_plate_anchor()
        self.plate_place_slot_assignments = {}
        self.placed_block_keys = []
        self.object_layers = {key: layer for key, layer in zip(self.block_keys, self.block_layers)}
        self.object_layers["B"] = self.plate_layer
        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.05)
        self.add_prohibit_area(self.plate, padding=0.08)
        self._configure_rotate_subtask_plan()
        self._prime_known_target_cache()
        self.pending_block_keys_for_search_snapshot = list(self.block_keys)
        self.pending_block_search_snapshots = {}
        self.pending_block_search_snapshot_seq = 0
        self.last_pending_block_search_snapshot = None

    def _get_object_arm_tag(self, obj):
        if not bool(getattr(self, "need_plan", True)):
            left_remaining = len(getattr(self, "left_joint_path", []) or []) - int(getattr(self, "left_cnt", 0))
            right_remaining = len(getattr(self, "right_joint_path", []) or []) - int(getattr(self, "right_cnt", 0))
            if left_remaining > 0 and right_remaining <= 0:
                return ArmTag("left")
            if right_remaining > 0 and left_remaining <= 0:
                return ArmTag("right")

        obj_cyl = world_to_robot(obj.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if obj_cyl[1] >= 0 else "right")

    def _mark_registry_target_discovered(self, object_key):
        key = str(object_key)
        if key not in self.object_registry or key not in self.discovered_objects:
            return False

        obj = self.object_registry.get(key, None)
        world_point = None
        if obj is not None:
            try:
                world_point = self._resolve_object_world_point(obj=obj)
            except Exception:
                world_point = None

        state = self.discovered_objects[key]
        state["discovered"] = True
        state["visible_now"] = False
        state["last_seen_subtask"] = int(self.current_subtask_idx)
        state["last_seen_stage"] = int(self.current_stage)
        if world_point is not None:
            state["last_world_point"] = np.array(world_point, dtype=np.float64).reshape(-1).tolist()

        if key in self.visible_objects:
            self.visible_objects[key] = False
        return True

    def _prime_known_target_cache(self):
        for key in self.KNOWN_FIXED_TARGET_KEYS:
            self._mark_registry_target_discovered(key)

    def _after_rotate_visibility_refresh(self, visibility_map):
        pending_block_keys = [
            str(key)
            for key in getattr(self, "pending_block_keys_for_search_snapshot", [])
            if str(key) in self.object_registry
        ]
        if len(pending_block_keys) == 0:
            return None
        visible_pending_block_keys = [key for key in pending_block_keys if bool(self.visible_objects.get(key, False))]
        if len(visible_pending_block_keys) == 0:
            return None
        snapshot = self._capture_rotate_search_snapshot()
        if snapshot is None:
            return None
        snapshot_seq = int(getattr(self, "pending_block_search_snapshot_seq", 0)) + 1
        self.pending_block_search_snapshot_seq = snapshot_seq
        snapshot["pending_block_keys"] = list(visible_pending_block_keys)
        snapshot["refresh_seq"] = snapshot_seq
        stored_snapshot = {
            **snapshot,
            "pending_block_keys": list(snapshot["pending_block_keys"]),
        }
        self.last_pending_block_search_snapshot = stored_snapshot
        snapshot_map = getattr(self, "pending_block_search_snapshots", None)
        if not isinstance(snapshot_map, dict):
            snapshot_map = {}
            self.pending_block_search_snapshots = snapshot_map
        for key in visible_pending_block_keys:
            snapshot_map[str(key)] = {
                **stored_snapshot,
                "pending_block_keys": list(stored_snapshot["pending_block_keys"]),
            }
        return None

    def _has_discovered_pending_block(self, pending_block_keys):
        pending_block_keys = [str(key) for key in pending_block_keys]
        return any(bool(self.discovered_objects.get(key, {}).get("discovered", False)) for key in pending_block_keys)

    def _restore_block_search_snapshot_or_default(self, scan_z):
        pending_block_keys = [
            str(key)
            for key in getattr(self, "pending_block_keys_for_search_snapshot", [])
            if str(key) in self.object_registry
        ]
        snapshot = None
        snapshot_map = getattr(self, "pending_block_search_snapshots", None)
        if isinstance(snapshot_map, dict):
            candidate_snapshots = [
                snapshot_map.get(key, None) for key in pending_block_keys if snapshot_map.get(key, None) is not None
            ]
            if len(candidate_snapshots) > 0:
                snapshot = max(candidate_snapshots, key=lambda item: int(item.get("refresh_seq", -1)))
        if snapshot is None:
            last_snapshot = getattr(self, "last_pending_block_search_snapshot", None)
            if last_snapshot is not None:
                snapshot_keys = [str(key) for key in last_snapshot.get("pending_block_keys", [])]
                if any(key in pending_block_keys for key in snapshot_keys):
                    snapshot = last_snapshot
        if snapshot is None:
            snapshot = self._get_default_rotate_search_snapshot()
        restored = False
        if snapshot is not None:
            restored = self._restore_rotate_search_snapshot(
                snapshot,
                scan_r=self.SCAN_R,
                scan_z=scan_z,
                joint_name_prefer=self.SCAN_JOINT_NAME,
                head_joint2_name=getattr(self, "rotate_head_joint2_name", "astribot_head_joint_2"),
            )
        if not restored:
            default_snapshot = self._get_default_rotate_search_snapshot()
            if default_snapshot is not None:
                restored = self._restore_rotate_search_snapshot(
                    default_snapshot,
                    scan_r=self.SCAN_R,
                    scan_z=scan_z,
                    joint_name_prefer=self.SCAN_JOINT_NAME,
                    head_joint2_name=getattr(self, "rotate_head_joint2_name", "astribot_head_joint_2"),
                )
        if not restored:
            self.plan_success = False
        return restored

    def _ensure_action_target_visible(self, subtask_idx, object_key):
        focus_key = str(object_key)
        subtask_def = self._get_rotate_subtask_def(subtask_idx) or {}
        focused_key = self._align_rotate_registry_target_with_torso_and_head_joint2(
            focus_key,
            subtask_idx=subtask_idx,
            target_keys=[str(k) for k in subtask_def.get("search_target_keys", [focus_key])],
            action_target_keys=[str(k) for k in subtask_def.get("action_target_keys", [focus_key])],
            joint_name_prefer=self.SCAN_JOINT_NAME,
            head_joint2_name=getattr(self, "rotate_head_joint2_name", "astribot_head_joint_2"),
        )
        if focused_key is None:
            self.plan_success = False
            return False
        return True

    def _lift_block_to_place_ready_pose(self, arm_tag):
        if not self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.PICK_LIFT_Z)):
            return False
        if float(self.POST_GRASP_EXTRA_LIFT_Z) > 1e-9 and self.plan_success:
            return bool(self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.POST_GRASP_EXTRA_LIFT_Z)))
        return True

    def _planner_pose_from_tcp_pose(self, tcp_pose):
        tcp_pose = np.array(tcp_pose, dtype=np.float64).reshape(-1)
        planner_pos = tcp_pose[:3] - t3d.quaternions.quat2mat(tcp_pose[3:]) @ np.array(
            [float(self.DIRECT_RELEASE_TCP_BACKOFF), 0.0, 0.0],
            dtype=np.float64,
        )
        return planner_pos.tolist() + tcp_pose[3:].tolist()

    def _get_direct_release_entry_r(self, release_r):
        explicit_entry_r = getattr(self, "DIRECT_RELEASE_ENTRY_TCP_CYL_R", None)
        if explicit_entry_r is not None:
            return float(explicit_entry_r)

        entry_r = float(release_r) - 0.15
        plate_layer = getattr(self, "plate_layer", None)
        if plate_layer is None:
            plate_layer = self._get_plate_layer()
        if plate_layer == "upper":
            upper_inner = getattr(self, "rotate_fan_double_upper_inner_radius", None)
            if upper_inner is not None:
                entry_r = float(upper_inner) - float(self.DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER)
        return float(max(0.35, min(entry_r, float(release_r) - 0.08)))

    def _build_horizontal_tcp_pose(self, cyl_r, cyl_theta_rad, tcp_z, yaw):
        quat = t3d.euler.euler2quat(0.0, 0.0, float(yaw)).tolist()
        return place_pose_cyl(
            [
                float(cyl_r),
                float(cyl_theta_rad),
                float(tcp_z),
            ] + quat,
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
            quat_frame="world",
        )

    def _get_plate_place_slot_offsets(self):
        slot_offset_map = dict(getattr(self, "PLATE_PLACE_SLOT_OFFSETS", {}) or {})
        block_count = int(getattr(self, "block_count", len(getattr(self, "block_keys", [])) or 1))
        slot_offsets = slot_offset_map.get(block_count, None)
        if slot_offsets is None and len(slot_offset_map) > 0:
            nearest_key = min(slot_offset_map.keys(), key=lambda key: abs(int(key) - block_count))
            slot_offsets = slot_offset_map[nearest_key]
        if slot_offsets is None or len(slot_offsets) == 0:
            return [(0.0, 0.0)]
        return [(float(offset[0]), float(offset[1])) for offset in slot_offsets]

    def _get_plate_place_slot_index(self, block_key):
        key = None if block_key is None else str(block_key)
        assignments = getattr(self, "plate_place_slot_assignments", None)
        if not isinstance(assignments, dict):
            assignments = {}
            self.plate_place_slot_assignments = assignments
        if key in assignments:
            return int(assignments[key])
        slot_offsets = self._get_plate_place_slot_offsets()
        assigned_indices = {int(idx) for idx in assignments.values()}
        remaining_indices = [idx for idx in range(len(slot_offsets)) if idx not in assigned_indices]
        if len(remaining_indices) == 0:
            next_idx = len(assignments) % len(slot_offsets)
        elif len(assigned_indices) == 0:
            next_idx = remaining_indices[0]
        else:
            assigned_points = [np.array(slot_offsets[idx], dtype=np.float64) for idx in sorted(assigned_indices)]
            next_idx = max(
                remaining_indices,
                key=lambda idx: min(
                    float(np.linalg.norm(np.array(slot_offsets[idx], dtype=np.float64) - point))
                    for point in assigned_points
                ),
            )
        assignments[key] = int(next_idx)
        return int(next_idx)

    def _get_plate_place_target_pose(self, block_key=None):
        plate = self.object_registry.get("B", None)
        if plate is not None:
            try:
                plate_pose = plate.get_functional_point(0, "pose")
                target_pose = plate_pose.p.tolist() + plate_pose.q.tolist()
            except Exception:
                target_pose = list(self.plate_target_pose)
        else:
            target_pose = list(self.plate_target_pose)
        target_pose = np.array(target_pose, dtype=np.float64).reshape(-1)
        slot_offsets = self._get_plate_place_slot_offsets()
        slot_idx = 0 if block_key is None else self._get_plate_place_slot_index(block_key)
        radial_offset, tangential_offset = slot_offsets[int(slot_idx) % len(slot_offsets)]

        try:
            target_cyl = world_to_robot(target_pose[:3].tolist(), self.robot_root_xy, self.robot_yaw)
            target_theta = float(target_cyl[1])
        except Exception:
            target_theta = float(np.deg2rad(getattr(self, "plate_cyl_theta_deg", 0.0)))

        world_theta = float(self.robot_yaw) + target_theta
        radial_xy = np.array([np.cos(world_theta), np.sin(world_theta)], dtype=np.float64)
        tangential_xy = np.array([-np.sin(world_theta), np.cos(world_theta)], dtype=np.float64)
        target_pose[:2] += radial_offset * radial_xy + tangential_offset * tangential_xy
        return target_pose.tolist()

    def _build_upper_to_lower_top_down_place_candidates(self, block, arm_tag, target_pose):
        actor_pose_mat = block.get_pose().to_transformation_matrix()
        fp_pose_mat = np.array(block.get_functional_point(0, "matrix"), dtype=np.float64).reshape(4, 4)
        ee_pose = np.array(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(-1)

        ee_pose_mat = np.eye(4, dtype=np.float64)
        ee_pose_mat[:3, :3] = t3d.quaternions.quat2mat(ee_pose[3:])
        ee_pose_mat[:3, 3] = ee_pose[:3]

        target_pose = np.array(target_pose, dtype=np.float64).reshape(-1)
        target_pose_mat = np.eye(4, dtype=np.float64)
        target_pose_mat[:3, :3] = t3d.quaternions.quat2mat(target_pose[3:])
        target_pose_mat[:3, 3] = target_pose[:3]

        actor_to_fp = np.linalg.inv(actor_pose_mat) @ fp_pose_mat
        actor_to_ee = np.linalg.inv(actor_pose_mat) @ ee_pose_mat
        target_z = np.array(target_pose_mat[:3, 2], dtype=np.float64).reshape(3)
        target_z_norm = float(np.linalg.norm(target_z))
        if target_z_norm <= 1e-9:
            target_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            target_z /= target_z_norm

        def _axis_angle_to_matrix(axis, angle_rad):
            axis = np.array(axis, dtype=np.float64).reshape(3)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-9 or abs(float(angle_rad)) <= 1e-9:
                return np.eye(3, dtype=np.float64)
            axis /= axis_norm
            cross = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=np.float64,
            )
            return np.eye(3, dtype=np.float64) + np.sin(angle_rad) * cross + (1.0 - np.cos(angle_rad)) * (cross @ cross)

        def _pose_list_from_matrix(pose_mat):
            return pose_mat[:3, 3].astype(np.float64).tolist() + t3d.quaternions.mat2quat(pose_mat[:3, :3]).tolist()

        def _ee_pose_from_fp_pose(fp_pose_mat):
            actor_pose = fp_pose_mat @ np.linalg.inv(actor_to_fp)
            return actor_pose @ actor_to_ee

        candidates = []
        base_rot = np.array(target_pose_mat[:3, :3], dtype=np.float64).reshape(3, 3)
        for yaw_offset_deg in tuple(getattr(self, "UPPER_TO_LOWER_TOP_DOWN_YAW_OFFSETS_DEG", (0.0, 90.0, -90.0, 180.0))):
            yaw_rot = _axis_angle_to_matrix(target_z, np.deg2rad(float(yaw_offset_deg)))
            fp_rot = yaw_rot @ base_rot

            carry_fp_pose = np.eye(4, dtype=np.float64)
            carry_fp_pose[:3, :3] = fp_rot
            carry_fp_pose[:3, 3] = target_pose_mat[:3, 3] + float(self.UPPER_TO_LOWER_TOP_DOWN_CARRY_Z_OFFSET) * target_z

            pre_place_fp_pose = np.eye(4, dtype=np.float64)
            pre_place_fp_pose[:3, :3] = fp_rot
            pre_place_fp_pose[:3, 3] = target_pose_mat[:3, 3] + float(self.UPPER_TO_LOWER_TOP_DOWN_PRE_DIS) * target_z

            place_fp_pose = np.eye(4, dtype=np.float64)
            place_fp_pose[:3, :3] = fp_rot
            place_fp_pose[:3, 3] = target_pose_mat[:3, 3] + float(self.UPPER_TO_LOWER_TOP_DOWN_DIS) * target_z

            candidates.append(
                {
                    "carry_pose": _pose_list_from_matrix(_ee_pose_from_fp_pose(carry_fp_pose)),
                    "pre_place_pose": _pose_list_from_matrix(_ee_pose_from_fp_pose(pre_place_fp_pose)),
                    "place_pose": _pose_list_from_matrix(_ee_pose_from_fp_pose(place_fp_pose)),
                }
            )
        return candidates

    def _place_upper_picked_block_into_lower_plate_with_top_down_transition(self, arm_tag, subtask_idx, block_key):
        block_key = str(block_key)
        block = self.object_registry.get(block_key, None)
        if block is None:
            self.plan_success = False
            return True

        target_pose = self._get_plate_place_target_pose(block_key)
        candidates = self._build_upper_to_lower_top_down_place_candidates(block, arm_tag, target_pose)
        selected = self._select_pose_sequence_candidate(
            arm_tag,
            candidates,
            ("carry_pose", "pre_place_pose", "place_pose"),
        )
        if selected is None:
            return False

        for pose_key in ("carry_pose", "pre_place_pose", "place_pose"):
            if not self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=selected[pose_key])):
                return True
        if not self.move(self.open_gripper(arm_tag)):
            return True
        self._set_carried_object_keys([])
        if not self._retreat_after_lower_place(arm_tag):
            return True
        self.placed_block_keys.append(block_key)
        self.complete_rotate_subtask(subtask_idx, carried_after=[])
        return True

    def _build_direct_release_pose_candidates(self, arm_tag, target_pose=None):
        target_pose = self._get_plate_place_target_pose() if target_pose is None else target_pose
        target_pose = np.array(target_pose, dtype=np.float64).reshape(-1)
        target_xyz = target_pose[:3]
        target_cyl = world_to_robot(target_xyz.tolist(), self.robot_root_xy, self.robot_yaw)
        plate_xy = np.array(target_xyz[:2], dtype=np.float64)
        root_xy = np.array(self.robot_root_xy, dtype=np.float64)
        outward_yaw = float(np.arctan2(plate_xy[1] - root_xy[1], plate_xy[0] - root_xy[0]))

        yaw_candidates = [float(outward_yaw + np.deg2rad(offset)) for offset in self.DIRECT_RELEASE_YAW_OFFSETS_DEG]

        candidates = []
        release_z = float(target_xyz[2] + self.DIRECT_RELEASE_TCP_Z_OFFSET)
        entry_z = max(float(target_xyz[2] + self.DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET), release_z)
        approach_z = max(float(target_xyz[2] + self.DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET), release_z)
        entry_theta_rad = float(target_cyl[1])
        for r_offset in self.DIRECT_RELEASE_R_OFFSETS:
            release_r = float(target_cyl[0] + float(r_offset))
            entry_r = self._get_direct_release_entry_r(release_r)
            for theta_offset_deg in self.DIRECT_RELEASE_THETA_OFFSETS_DEG:
                release_theta_rad = float(target_cyl[1] + np.deg2rad(float(theta_offset_deg)))
                for yaw in yaw_candidates:
                    # Keep the gripper frame horizontal and approach from the upper-table free inner ring first.
                    entry_tcp_pose = self._build_horizontal_tcp_pose(entry_r, entry_theta_rad, entry_z, yaw)
                    approach_tcp_pose = self._build_horizontal_tcp_pose(release_r, release_theta_rad, approach_z, yaw)
                    tcp_pose = self._build_horizontal_tcp_pose(release_r, release_theta_rad, release_z, yaw)
                    candidates.append(
                        {
                            "entry_tcp_pose": entry_tcp_pose,
                            "entry_planner_pose": self._planner_pose_from_tcp_pose(entry_tcp_pose),
                            "approach_tcp_pose": approach_tcp_pose,
                            "approach_planner_pose": self._planner_pose_from_tcp_pose(approach_tcp_pose),
                            "tcp_pose": tcp_pose,
                            "planner_pose": self._planner_pose_from_tcp_pose(tcp_pose),
                        }
                    )
        return candidates

    def _select_direct_release_pose(self, arm_tag, candidates=None, pose_key="planner_pose"):
        plan_path = self.robot.left_plan_path if ArmTag(arm_tag) == "left" else self.robot.right_plan_path
        if candidates is None:
            candidates = self._build_direct_release_pose_candidates(arm_tag)
        if not bool(getattr(self, "need_plan", True)):
            for candidate in candidates:
                if candidate.get(pose_key, None) is not None:
                    return candidate
            return None
        for candidate in candidates:
            planner_pose = candidate.get(pose_key, None)
            if planner_pose is None:
                continue
            plan_res = plan_path(planner_pose)
            if isinstance(plan_res, dict) and plan_res.get("status", None) == "Success":
                return candidate
        return None

    def _select_direct_release_pose_sequence_candidate(self, arm_tag, candidates=None):
        if candidates is None:
            candidates = self._build_direct_release_pose_candidates(arm_tag)
        return self._select_pose_sequence_candidate(
            arm_tag,
            candidates,
            ("entry_planner_pose", "approach_planner_pose", "planner_pose"),
        )

    def _move_to_first_direct_release_pose(self, arm_tag, candidates, pose_key, selected_candidate=None):
        selected = selected_candidate
        if selected is None:
            selected = self._select_direct_release_pose(arm_tag, candidates=candidates, pose_key=pose_key)
        if selected is None:
            self.plan_success = False
            return False
        return bool(self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=selected[pose_key])))

    def _expand_active_plan_qpos_to_entity_qpos(self, arm_tag, active_qpos):
        active_qpos = np.array(active_qpos, dtype=np.float64).reshape(-1)
        if ArmTag(arm_tag) == "left":
            entity = getattr(self.robot, "left_entity", None)
            planner = getattr(self.robot, "left_planner", None)
        else:
            entity = getattr(self.robot, "right_entity", None)
            planner = getattr(self.robot, "right_planner", None)
        if entity is None or planner is None:
            return None

        try:
            full_qpos = np.array(entity.get_qpos(), dtype=np.float64).reshape(-1)
        except Exception:
            return None

        active_joint_names = list(getattr(planner, "active_joints_name", []) or [])
        all_joint_names = list(getattr(planner, "all_joints", []) or [])
        if len(active_joint_names) != active_qpos.shape[0] or len(all_joint_names) == 0:
            return None

        for active_idx, joint_name in enumerate(active_joint_names):
            if joint_name not in all_joint_names:
                return None
            full_idx = all_joint_names.index(joint_name)
            if full_idx >= full_qpos.shape[0]:
                return None
            full_qpos[full_idx] = active_qpos[active_idx]
        return [float(value) for value in full_qpos.astype(np.float32).tolist()]

    def _plan_path_for_sequence_check(self, plan_path, planner_pose, last_entity_qpos=None):
        try:
            if last_entity_qpos is None:
                return plan_path(planner_pose)
            return plan_path(planner_pose, last_qpos=last_entity_qpos)
        except TypeError:
            try:
                return plan_path(planner_pose)
            except Exception as exc:
                return {"status": "Fail", "error": str(exc)}
        except (IndexError, RuntimeError) as exc:
            # last_qpos is only used for candidate filtering; never let it crash data collection.
            if last_entity_qpos is None:
                return {"status": "Fail", "error": str(exc)}
            try:
                return plan_path(planner_pose)
            except Exception as retry_exc:
                return {"status": "Fail", "error": str(retry_exc)}

    def _select_pose_sequence_candidate(self, arm_tag, candidates, pose_keys):
        if not bool(getattr(self, "need_plan", True)):
            for candidate in candidates:
                if all(candidate.get(key, None) is not None for key in pose_keys):
                    return candidate
            return None

        plan_path = self.robot.left_plan_path if ArmTag(arm_tag) == "left" else self.robot.right_plan_path
        for candidate in candidates:
            last_entity_qpos = None
            ok = True
            for pose_key in pose_keys:
                planner_pose = candidate.get(pose_key, None)
                if planner_pose is None:
                    ok = False
                    break
                plan_res = self._plan_path_for_sequence_check(
                    plan_path,
                    planner_pose,
                    last_entity_qpos=last_entity_qpos,
                )
                if not (isinstance(plan_res, dict) and plan_res.get("status", None) == "Success"):
                    ok = False
                    break
                position = plan_res.get("position", None)
                if position is not None and len(position) > 0:
                    expanded_qpos = self._expand_active_plan_qpos_to_entity_qpos(arm_tag, position[-1])
                    last_entity_qpos = expanded_qpos
            if ok:
                return candidate
        return None

    def _build_upper_pick_pose_candidates(self, block, arm_tag):
        block_pos = np.array(block.get_pose().p, dtype=np.float64).reshape(3)
        root_xy = np.array(self.robot_root_xy, dtype=np.float64)
        outward_yaw = float(np.arctan2(block_pos[1] - root_xy[1], block_pos[0] - root_xy[0]))

        candidates = []
        for yaw_offset_deg in self.UPPER_PICK_YAW_OFFSETS_DEG:
            yaw = float(outward_yaw + np.deg2rad(float(yaw_offset_deg)))
            quat = t3d.euler.euler2quat(0.0, 0.0, yaw).tolist()
            local_x = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)

            grasp_pos = np.array(block_pos, dtype=np.float64)
            grasp_pos[2] += float(self.UPPER_PICK_GRASP_Z_BIAS)
            pre_grasp_pos = grasp_pos - float(self.UPPER_PICK_PRE_GRASP_DIS) * local_x
            entry_pos = np.array(pre_grasp_pos, dtype=np.float64)
            entry_pos[2] += float(self.UPPER_PICK_ENTRY_Z_OFFSET)
            post_grasp_retreat_pos = np.array(pre_grasp_pos, dtype=np.float64)
            post_entry_retreat_pos = entry_pos - float(self.UPPER_PICK_POST_ENTRY_RETREAT_DIS) * local_x

            entry_tcp_pose = entry_pos.tolist() + quat
            pre_grasp_tcp_pose = pre_grasp_pos.tolist() + quat
            grasp_tcp_pose = grasp_pos.tolist() + quat
            post_grasp_retreat_tcp_pose = post_grasp_retreat_pos.tolist() + quat
            post_entry_retreat_tcp_pose = post_entry_retreat_pos.tolist() + quat
            candidates.append(
                {
                    "entry_tcp_pose": entry_tcp_pose,
                    "entry_planner_pose": self._planner_pose_from_tcp_pose(entry_tcp_pose),
                    "pre_grasp_tcp_pose": pre_grasp_tcp_pose,
                    "pre_grasp_planner_pose": self._planner_pose_from_tcp_pose(pre_grasp_tcp_pose),
                    "grasp_tcp_pose": grasp_tcp_pose,
                    "grasp_planner_pose": self._planner_pose_from_tcp_pose(grasp_tcp_pose),
                    "post_grasp_retreat_tcp_pose": post_grasp_retreat_tcp_pose,
                    "post_grasp_retreat_planner_pose": self._planner_pose_from_tcp_pose(post_grasp_retreat_tcp_pose),
                    "post_entry_retreat_tcp_pose": post_entry_retreat_tcp_pose,
                    "post_entry_retreat_planner_pose": self._planner_pose_from_tcp_pose(post_entry_retreat_tcp_pose),
                }
            )
        return candidates

    def _get_post_place_escape_theta(self, arm_tag):
        try:
            ee_pose = np.array(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(-1)
        except Exception:
            ee_pose = None
        if ee_pose is not None and ee_pose.shape[0] >= 3 and np.all(np.isfinite(ee_pose[:3])):
            try:
                ee_cyl = world_to_robot(ee_pose[:3].tolist(), self.robot_root_xy, self.robot_yaw)
                theta_rad = float(ee_cyl[1])
                if np.isfinite(theta_rad):
                    return theta_rad
            except Exception:
                pass

        plate_theta_deg = getattr(self, "plate_cyl_theta_deg", None)
        if plate_theta_deg is not None and np.isfinite(float(plate_theta_deg)):
            return float(np.deg2rad(float(plate_theta_deg)))

        return 0.0

    def _get_lateral_escape_displacement(self, arm_tag, distance=None, theta_rad=None):
        lateral_dis = float(self.POST_PLACE_LATERAL_ESCAPE_DIS if distance is None else distance)
        if lateral_dis <= 1e-9:
            return [0.0, 0.0]

        if theta_rad is None:
            theta_rad = self._get_post_place_escape_theta(arm_tag)
        theta_rad = float(theta_rad)
        if theta_rad > 1e-6:
            side_sign = 1.0
        elif theta_rad < -1e-6:
            side_sign = -1.0
        else:
            side_sign = 1.0 if ArmTag(arm_tag) == "left" else -1.0

        world_theta = float(self.robot_yaw) + theta_rad
        tangent_xy = np.array(
            [-np.sin(world_theta), np.cos(world_theta)],
            dtype=np.float64,
        )
        escape_xy = side_sign * tangent_xy
        escape_norm = float(np.linalg.norm(escape_xy))
        if escape_norm <= 1e-9:
            left_outward_xy = np.array(
                [-np.sin(float(self.robot_yaw)), np.cos(float(self.robot_yaw))],
                dtype=np.float64,
            )
            escape_xy = left_outward_xy if ArmTag(arm_tag) == "left" else -left_outward_xy
            escape_norm = float(np.linalg.norm(escape_xy))
        if escape_norm > 1e-9:
            escape_xy = escape_xy / escape_norm
        return (escape_xy * lateral_dis).tolist()

    def _retreat_then_return_both_arms_to_initial_pose(self, arm_tag):
        if not self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.DIRECT_RELEASE_RETREAT_Z, move_axis="world")):
            return False
        if not bool(self.RETURN_TO_HOMESTATE_AFTER_PLACE):
            return True
        plate_layer = getattr(self, "object_layers", {}).get("B", None)
        if plate_layer is None:
            plate_layer = getattr(self, "plate_layer", None)
        if plate_layer == "upper":
            lateral_xy = self._get_lateral_escape_displacement(arm_tag)
            if not self.move(
                self.move_by_displacement(
                    arm_tag=arm_tag,
                    x=float(lateral_xy[0]),
                    y=float(lateral_xy[1]),
                    move_axis="world",
                )
            ):
                return False
        return bool(self.move(self.back_to_origin("left"), self.back_to_origin("right")))

    def _focus_plate_before_place(self, subtask_idx, plate_key):
        plate_key = str(plate_key or "B")
        plate_visible = self._ensure_action_target_visible(subtask_idx, plate_key)
        if bool(self.REQUIRE_PLATE_VISIBLE_BEFORE_PLACE) and not plate_visible:
            self.plan_success = False
            return False
        return True

    def _pick_block(self, subtask_idx, block_key):
        block_key = str(block_key)
        block = self.object_registry[block_key]
        arm_tag = self._get_object_arm_tag(block)
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=block_key)
        block_layer = self.object_layers.get(block_key, "lower")
        if block_layer == "upper":
            return self._pick_upper_block_with_direct_move(subtask_idx, block_key, block, arm_tag)
        return self._pick_lower_block_with_grasp_actor(subtask_idx, block_key, block, arm_tag)

    def _pick_lower_block_with_grasp_actor(self, subtask_idx, block_key, block, arm_tag):
        if not self.move(
            self.grasp_actor(
                block,
                arm_tag=arm_tag,
                pre_grasp_dis=self.PICK_PRE_GRASP_DIS,
                grasp_dis=self.PICK_GRASP_DIS,
            )
        ):
            return arm_tag
        self._set_carried_object_keys([block_key])
        if not self._lift_block_to_place_ready_pose(arm_tag):
            return arm_tag
        self.complete_rotate_subtask(subtask_idx, carried_after=[block_key])
        return arm_tag

    def _pick_upper_block_with_direct_move(self, subtask_idx, block_key, block, arm_tag):
        self._sync_curobo_tabletop_collisions()
        candidates = self._build_upper_pick_pose_candidates(block, arm_tag)
        selected = self._select_pose_sequence_candidate(
            arm_tag,
            candidates,
            (
                "entry_planner_pose",
                "pre_grasp_planner_pose",
                "grasp_planner_pose",
                "post_grasp_retreat_planner_pose",
                "entry_planner_pose",
                "post_entry_retreat_planner_pose",
            ),
        )
        if selected is None:
            self.plan_success = False
            return arm_tag

        if not self.move(self.open_gripper(arm_tag)):
            return arm_tag
        for pose_key in ("entry_planner_pose", "pre_grasp_planner_pose", "grasp_planner_pose"):
            if not self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=selected[pose_key])):
                return arm_tag
        if not self.move(self.close_gripper(arm_tag, pos=self.UPPER_PICK_GRIPPER_POS)):
            return arm_tag

        self._set_carried_object_keys([block_key])
        for pose_key in (
            "post_grasp_retreat_planner_pose",
            "entry_planner_pose",
            "post_entry_retreat_planner_pose",
        ):
            if not self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=selected[pose_key])):
                return arm_tag
        self.complete_rotate_subtask(subtask_idx, carried_after=[block_key])
        return arm_tag

    def _place_block_into_plate(self, arm_tag, subtask_idx, block_key, focus_object_key):
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=(focus_object_key or "B"))
        self._sync_curobo_tabletop_collisions()
        plate_layer = getattr(self, "object_layers", {}).get("B", None)
        if plate_layer is None:
            plate_layer = self._get_plate_layer()
        block_layer = getattr(self, "object_layers", {}).get(str(block_key), None)
        if plate_layer == "lower":
            if (
                bool(getattr(self, "UPPER_TO_LOWER_TOP_DOWN_PLACE_ENABLED", True))
                and block_layer == "upper"
                and self._place_upper_picked_block_into_lower_plate_with_top_down_transition(arm_tag, subtask_idx, block_key)
            ):
                return
            return self._place_block_into_lower_plate_with_place_actor(arm_tag, subtask_idx, block_key)
        return self._place_block_into_plate_with_direct_release(arm_tag, subtask_idx, block_key=block_key)

    def _retreat_after_lower_place(self, arm_tag):
        if not self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.PLACE_RETREAT_Z, move_axis="arm")):
            return False
        if not bool(self.RETURN_TO_HOMESTATE_AFTER_PLACE):
            return True
        return bool(self.move(self.back_to_origin("left"), self.back_to_origin("right")))

    def _place_block_into_lower_plate_with_place_actor(self, arm_tag, subtask_idx, block_key):
        block_key = str(block_key)
        block = self.object_registry.get(block_key, None)
        if block is None:
            self.plan_success = False
            return

        target_pose = self._get_plate_place_target_pose(block_key)
        if not self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=self.LOWER_PLACE_PRE_DIS,
                dis=self.LOWER_PLACE_DIS,
                pre_dis_axis=self.LOWER_PLACE_PRE_DIS_AXIS,
                constrain=self.LOWER_PLACE_CONSTRAIN,
            )
        ):
            return
        self._set_carried_object_keys([])
        if not self._retreat_after_lower_place(arm_tag):
            return
        self.placed_block_keys.append(block_key)
        self.complete_rotate_subtask(subtask_idx, carried_after=[])

    def _place_block_into_plate_with_direct_release(self, arm_tag, subtask_idx, block_key=None):
        target_pose = self._get_plate_place_target_pose(block_key)
        release_candidates = self._build_direct_release_pose_candidates(arm_tag, target_pose=target_pose)
        selected_release_candidate = self._select_direct_release_pose_sequence_candidate(
            arm_tag,
            candidates=release_candidates,
        )
        if selected_release_candidate is None:
            self.plan_success = False
            return

        if not self._move_to_first_direct_release_pose(
            arm_tag,
            release_candidates,
            "entry_planner_pose",
            selected_candidate=selected_release_candidate,
        ):
            return
        if not self._move_to_first_direct_release_pose(
            arm_tag,
            release_candidates,
            "approach_planner_pose",
            selected_candidate=selected_release_candidate,
        ):
            return
        if not self._move_to_first_direct_release_pose(
            arm_tag,
            release_candidates,
            "planner_pose",
            selected_candidate=selected_release_candidate,
        ):
            return
        if not self.move(self.open_gripper(arm_tag)):
            return
        self._set_carried_object_keys([])
        if not self._retreat_then_return_both_arms_to_initial_pose(arm_tag):
            return
        if block_key is not None:
            self.placed_block_keys.append(str(block_key))
        self.complete_rotate_subtask(subtask_idx, carried_after=[])

    def _build_info(self, arm_tag):
        return {
            "{A}": "green block" if len(getattr(self, "block_keys", [])) <= 1 else "green blocks",
            "{B}": f"003_plate/base{self.PLATE_MODEL_ID}",
            "{a}": str(arm_tag),
        }

    def play_once(self):
        scan_z = float(self.SCAN_Z_BIAS + self.table_z_bias)
        last_arm_tag = ArmTag("left")
        remaining_block_keys = [str(key) for key in self.block_keys]
        self.plate_place_slot_assignments = {}
        self.placed_block_keys = []
        self.pending_block_keys_for_search_snapshot = list(remaining_block_keys)
        self.pending_block_search_snapshots = {}
        self.pending_block_search_snapshot_seq = 0
        self.last_pending_block_search_snapshot = None
        if not self._restore_block_search_snapshot_or_default(scan_z):
            self.info["info"] = self._build_info(last_arm_tag)
            return self.info

        for block_idx in range(len(self.block_keys)):
            pick_subtask_idx = 2 * block_idx + 1
            place_subtask_idx = pick_subtask_idx + 1
            self.pending_block_keys_for_search_snapshot = list(remaining_block_keys)
            self._prepare_dynamic_pick_subtask(pick_subtask_idx, remaining_block_keys)

            block_key = self.search_and_focus_rotate_and_head_subtask(
                pick_subtask_idx,
                scan_r=self.SCAN_R,
                scan_z=scan_z,
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            if block_key is None:
                self.plan_success = False
                fallback_key = remaining_block_keys[0] if len(remaining_block_keys) > 0 else self.block_keys[0]
                fallback_arm = self._get_object_arm_tag(self.object_registry[fallback_key])
                self.info["info"] = self._build_info(fallback_arm)
                return self.info
            if block_key not in remaining_block_keys:
                self.plan_success = False
                self.info["info"] = self._build_info(last_arm_tag)
                return self.info

            if not self._ensure_action_target_visible(pick_subtask_idx, block_key):
                self.info["info"] = self._build_info(last_arm_tag)
                return self.info

            arm_tag = self._pick_block(pick_subtask_idx, block_key)
            last_arm_tag = arm_tag
            if not self.plan_success:
                self.info["info"] = self._build_info(arm_tag)
                return self.info

            self.pending_block_keys_for_search_snapshot = [
                str(key) for key in remaining_block_keys if str(key) != str(block_key)
            ]
            self._prepare_dynamic_place_subtask(place_subtask_idx, block_key)
            plate_key = self.search_and_focus_rotate_and_head_subtask(
                place_subtask_idx,
                scan_r=self.SCAN_R,
                scan_z=scan_z,
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            if plate_key is None:
                self.plan_success = False
                self.info["info"] = self._build_info(arm_tag)
                return self.info
            if not self._focus_plate_before_place(place_subtask_idx, plate_key):
                self.info["info"] = self._build_info(arm_tag)
                return self.info

            self._place_block_into_plate(arm_tag, place_subtask_idx, block_key, plate_key)
            if not self.plan_success:
                self.info["info"] = self._build_info(arm_tag)
                return self.info
            remaining_block_keys.remove(block_key)
            self.pending_block_search_snapshots.pop(str(block_key), None)
            self.pending_block_keys_for_search_snapshot = list(remaining_block_keys)
            if len(remaining_block_keys) > 0 and not self._has_discovered_pending_block(remaining_block_keys):
                if not self._restore_block_search_snapshot_or_default(scan_z):
                    self.info["info"] = self._build_info(arm_tag)
                    return self.info

        self.info["info"] = self._build_info(last_arm_tag)
        return self.info

    def check_success(self):
        plate_pose = np.array(self.plate.get_functional_point(0, "pose").p, dtype=np.float64).reshape(3)
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        blocks = getattr(self, "blocks", None)
        if blocks is None:
            blocks = [self.block]
        blocks_in_plate = all(
            np.all(
                np.abs(
                    np.array(block.get_functional_point(0, "pose").p, dtype=np.float64).reshape(3) - plate_pose
                )
                < self.SUCCESS_EPS
            )
            for block in blocks
        )
        return bool(blocks_in_plate and gripper_open)
