from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d

# 回退到原始版本

class PutBlockTargetFanDoubleBase(Base_Task):
    # 坐标约定：
    # 这里的 cyl 参数都使用机器人根部为圆心的柱坐标。
    # r 表示水平半径，theta_deg=0 表示机器人初始正前方，z 表示世界坐标高度。
    #
    # block 生成参数：
    # BLOCK_COUNT 手动控制生成 1/2/3 个 block。
    # BLOCK_LAYER_SEQUENCE 显式决定每个 block 的层，长度必须等于 BLOCK_COUNT。
    BLOCK_COUNT = 2
    BLOCK_LAYER_SEQUENCE = ("lower","lower")
    BLOCK_SIZE_RANGE = (0.015, 0.025)
    BLOCK_COLOR = (0.10, 0.80, 0.20)
    BLOCK_COLOR_CANDIDATES = (
        (0.90, 0.20, 0.20),
        (0.15, 0.72, 0.25),
        (0.20, 0.45, 0.92),
        (0.92, 0.74, 0.18),
        (0.88, 0.45, 0.16),
    )
    BLOCK_SPAWN_MIN_DIST_SQ = 0.01
    PLATE_BLOCK_SPAWN_MIN_DIST_SQ = 0.0255
    BLOCK_LAYER_SPECS = {
        "lower": {
            "inner_margin": 0.12,
            "outer_margin": 0.18,
            "max_cyl_r": 0.55,
            "theta_shrink": 0.92,
        },
        "upper": {
            "inner_margin": 0.04,
            "outer_margin": 0.06,
            "max_cyl_r": 0.64,
            "theta_shrink": 0.92,
        },
    }

    # plate anchor 参数：
    # plate 的 z 由对应桌面 top_z + z_offset 计算，避免和 fan_double_layer_gap 不一致。
    PLATE_MODEL_ID = 0
    PLATE_LAYER = "upper"
    PLATE_LAYER_SPECS = {
        "lower": {
            "r": 0.45,
            "theta_deg": 20,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": [0.025, 0.025, 0.025],
        },
        "upper": {
            "r": 0.70,
            "theta_deg": -10,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": [0.025, 0.025, 0.025],
        },
    }
    # plate 内部的目标槽位，单位为相对 plate 中心的 [radial_offset, tangential_offset] 米。
    # radial: plate 相对机器人的径向；tangential: 桌面内与其垂直的切向。
    PLATE_PLACE_SLOT_OFFSETS = {
        1: ((0.0, 0.0),),
        2: ((0.0, -0.055), (0.0, 0.055)),
        3: ((0.055, 0.0), (-0.028, 0.050), (-0.028, -0.050)),
    }

    # 搜索参数：
    # scan_r/scan_z 决定模拟搜索时腰部对准的观察点，stage 规则在 Base_Task 中统一实现。
    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    PLACE_PLATE_UPPER_HEAD_JOINT2_TARGET = 0.8
    PLACE_PLATE_LOWER_HEAD_JOINT2_TARGET = None
    REQUIRE_PLATE_VISIBLE_BEFORE_PLACE = True
    FIXED_LAYER_HEAD_JOINT2_ONLY = False
    # 重新低头搜索时保存 head 运动过程，避免视频里相机视角瞬移。
    HEAD_RESET_SAVE_FREQ = -1

    # 抓取参数：
    # 先到 pre-grasp，再前进到 grasp，闭合后竖直抬升，尽量保证拿稳。
    PICK_PRE_GRASP_DIS = 0.09
    PICK_GRASP_DIS = 0.01
    PICK_LIFT_Z = 0.10
    POST_GRASP_EXTRA_LIFT_Z = 0.04

    # 初始手臂姿态微调，降低抓取后处在极限姿态的概率。
    INITIAL_LEFT_ARM_JOINT1 = -0.110
    INITIAL_RIGHT_ARM_JOINT1 = 0.110

    # direct release 参数：
    # TCP 是夹爪工作点；planner target 需要沿 TCP 局部 x 轴后退 DIRECT_RELEASE_TCP_BACKOFF。
    # release/entry/approach 都基于当前 plate 的柱坐标和层高度动态生成。
    DIRECT_RELEASE_TCP_BACKOFF = 0.12
    DIRECT_RELEASE_ENTRY_TCP_CYL_R = None
    DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER = 0.08
    DIRECT_RELEASE_TCP_Z_OFFSET = 0.06
    DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET = 0.10
    DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET = 0.10
    DIRECT_RELEASE_RETREAT_Z = 0.06
    DIRECT_RELEASE_R_OFFSETS = (0.0, -0.03, 0.03)
    DIRECT_RELEASE_THETA_OFFSETS_DEG = (0.0, -3.0, 3.0)
    DIRECT_RELEASE_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0)

    # lower place_actor 参数：
    # 下层到下层时改回 place_actor 放置，便于直接按任务配置调整靠近和松手姿态。
    LOWER_PLACE_FUNCTIONAL_POINT_ID = 0
    LOWER_PLACE_PRE_DIS = 0.18
    LOWER_PLACE_DIS = 0.03
    LOWER_PLACE_CONSTRAIN = "free"
    LOWER_PLACE_PRE_DIS_AXIS = "fp"
    LOWER_PLACE_IS_OPEN = True
    LOWER_PLACE_RETREAT_Z = 0.12
    LOWER_PLACE_RETREAT_MOVE_AXIS = "arm"

    # upper pick -> lower plate 参数：
    # 先为 block 计算一个“落下后仍在 plate 内”的安全底面中心，再反推 hover TCP pose。
    UPPER_TO_LOWER_HOVER_Z_OFFSETS = (0.06, 0.08, 0.10)
    UPPER_TO_LOWER_DROP_PLATE_INNER_MARGIN = 0.02
    UPPER_TO_LOWER_DROP_YAW_OFFSETS_DEG = (0.0, 90.0, -90.0, 180.0)
    UPPER_TO_LOWER_RELEASE_DELAY_STEPS = 15
    UPPER_TO_LOWER_RELEASE_RETREAT_Z = 0.08

    # 上层 block 抓取参数：
    # 由于上层较远，采用和 direct release 一致的 TCP->planner pose 语义直接 move。
    UPPER_PICK_ENTRY_Z_OFFSET = 0.10
    UPPER_PICK_PRE_GRASP_DIS = 0.10
    UPPER_PICK_GRASP_Z_BIAS = 0.02
    UPPER_PICK_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
    UPPER_PICK_GRIPPER_POS = -0.02

    # 上层放置后先沿当前机器人本体左右方向侧向撤离，再回 homestate。
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.18
    UPPER_PLACE_BODY_JOINT_NAME = "astribot_torso_joint_2"

    # 放置后是否直接用 move_joint 回到 homestate。
    # True: 松手后先竖直抬起，再 back_to_origin(left/right)。
    # False: 松手后只竖直抬起，不执行 homestate 回收。
    RETURN_TO_HOMESTATE_AFTER_PLACE = True

    # 成功判定参数：
    # 多 block 时要求每个 block 都落到 plate functional point 附近，且夹爪打开。
    KNOWN_FIXED_TARGET_KEYS = ()
    SUCCESS_EPS = np.array([0.08, 0.08, 0.08], dtype=np.float64)

    def setup_demo(self, **kwargs):
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
        self.fixed_layer_head_joint2_only = bool(
            kwargs.get(
                "fixed_layer_head_joint2_only",
                getattr(self, "FIXED_LAYER_HEAD_JOINT2_ONLY", False),
            )
        )
        if "place_plate_upper_head_joint2_target" in kwargs:
            self.PLACE_PLATE_UPPER_HEAD_JOINT2_TARGET = float(kwargs["place_plate_upper_head_joint2_target"])
        if "place_plate_lower_head_joint2_target" in kwargs:
            lower_target = kwargs["place_plate_lower_head_joint2_target"]
            self.PLACE_PLATE_LOWER_HEAD_JOINT2_TARGET = None if lower_target is None else float(lower_target)
        if "place_target_upper_head_joint2_target" in kwargs:
            self.PLACE_TARGET_UPPER_HEAD_JOINT2_TARGET = float(kwargs["place_target_upper_head_joint2_target"])
        if "place_target_lower_head_joint2_target" in kwargs:
            lower_target = kwargs["place_target_lower_head_joint2_target"]
            self.PLACE_TARGET_LOWER_HEAD_JOINT2_TARGET = None if lower_target is None else float(lower_target)
        kwargs = init_rotate_theta_bounds(self, kwargs)
        super()._init_task_env_(**kwargs)

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

    def _sample_block_colors(self, block_count):
        palette = [tuple(float(channel) for channel in color) for color in self.BLOCK_COLOR_CANDIDATES]
        if len(palette) == 0:
            return [tuple(float(channel) for channel in self.BLOCK_COLOR)] * int(block_count)
        color_order = np.random.permutation(len(palette))
        return [palette[int(color_order[idx % len(color_order)])] for idx in range(int(block_count))]

    def _get_target_object_key(self):
        return "B"

    def _get_target_object(self):
        target_object = getattr(self, "target_object", None)
        if target_object is not None:
            return target_object
        return getattr(self, "plate", None)

    def _get_target_layer(self):
        return self._get_plate_layer()

    def _get_target_layer_spec(self, layer_name=None):
        return self._get_plate_layer_spec(layer_name)

    def _get_target_anchor_pose(self, layer_name=None):
        return self._get_plate_anchor_pose(layer_name)

    def _create_target_anchor(self):
        self._create_plate_anchor()
        self.target_object = self.plate
        self.target_layer = self.plate_layer
        self.target_place_pose = list(self.plate_target_pose)
        return self.target_object

    def _get_target_place_target_pose(self, block_key=None):
        return self._get_plate_place_target_pose(block_key)

    def _get_task_instruction(self):
        return "Put the block into {B}." if len(self.block_keys) == 1 else "Put all blocks into {B}."

    def _get_target_padding(self):
        return 0.08

    def _configure_rotate_subtask_plan(self):
        object_registry = {key: block for key, block in zip(self.block_keys, self.blocks)}
        object_registry[self._get_target_object_key()] = self._get_target_object()

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
            task_instruction=self._get_task_instruction(),
        )

    def _update_rotate_subtask_targets(
        self,
        subtask_idx,
        search_target_keys=None,
        action_target_keys=None,
        required_carried_keys=None,
        carry_keys_after_done=None,
        allow_stage2_from_memory=None,
    ):
        subtask_def = self._get_rotate_subtask_def(subtask_idx)
        if subtask_def is None:
            raise ValueError(f"Unknown rotate subtask id: {subtask_idx}")

        if search_target_keys is not None:
            subtask_def["search_target_keys"] = [str(key) for key in search_target_keys]
        if action_target_keys is not None:
            subtask_def["action_target_keys"] = [str(key) for key in action_target_keys]
        if required_carried_keys is not None:
            subtask_def["required_carried_keys"] = [str(key) for key in required_carried_keys]
        if carry_keys_after_done is not None:
            subtask_def["carry_keys_after_done"] = [str(key) for key in carry_keys_after_done]
        if allow_stage2_from_memory is not None:
            subtask_def["allow_stage2_from_memory"] = bool(allow_stage2_from_memory)
        return subtask_def

    def _prepare_dynamic_pick_subtask(self, subtask_idx, remaining_block_keys):
        remaining_block_keys = [str(key) for key in remaining_block_keys]
        return self._update_rotate_subtask_targets(
            subtask_idx,
            search_target_keys=remaining_block_keys,
            action_target_keys=remaining_block_keys,
            required_carried_keys=[],
            carry_keys_after_done=[],
            allow_stage2_from_memory=(len(remaining_block_keys) == 1),
        )

    def _prepare_dynamic_place_subtask(self, subtask_idx, block_key):
        block_key = str(block_key)
        return self._update_rotate_subtask_targets(
            subtask_idx,
            search_target_keys=["B"],
            action_target_keys=[block_key, "B"],
            required_carried_keys=[block_key],
            carry_keys_after_done=[],
            allow_stage2_from_memory=True,
        )

    def _get_subtask_search_layers(self, subtask_idx):
        subtask_def = self._get_rotate_subtask_def(subtask_idx) or {}
        search_target_keys = [str(key) for key in subtask_def.get("search_target_keys", [])]
        if len(search_target_keys) == 0:
            return None

        object_registry = getattr(self, "object_registry", {}) or {}
        object_layers = getattr(self, "object_layers", {}) or {}
        resolved_layers = set()
        for key in search_target_keys:
            if key not in object_registry:
                return None
            layer_name = object_layers.get(key, None)
            if layer_name is None:
                return None
            resolved_layers.add(self._normalize_layer(layer_name))
        return resolved_layers if len(resolved_layers) > 0 else None

    def _subtask_requires_head_home_reset(self, subtask_idx, prev_subtask_idx=None):
        current_layers = self._get_subtask_search_layers(subtask_idx)
        if prev_subtask_idx is None or current_layers is None or len(current_layers) != 1:
            return True

        prev_layers = self._get_subtask_search_layers(prev_subtask_idx)
        if prev_layers is None or len(prev_layers) != 1:
            return True

        return next(iter(current_layers)) != next(iter(prev_layers))

    def _get_subtask_search_target_keys(self, subtask_idx):
        subtask_def = self._get_rotate_subtask_def(subtask_idx) or {}
        return [str(key) for key in subtask_def.get("search_target_keys", [])]

    def _get_subtask_upper_search_target_keys(self, subtask_idx):
        object_layers = getattr(self, "object_layers", {}) or {}
        upper_keys = []
        for key in self._get_subtask_search_target_keys(subtask_idx):
            layer_name = object_layers.get(key, None)
            if layer_name is None:
                continue
            if self._normalize_layer(layer_name) == "upper":
                upper_keys.append(str(key))
        return upper_keys

    def _should_search_lower_before_upper_for_subtask(self, subtask_idx):
        first_upper_idx = self._get_rotate_first_upper_search_state_index()
        if first_upper_idx is None:
            return False
        return bool(len(self._get_subtask_upper_search_target_keys(subtask_idx)) > 0)

    def _has_unfinished_lower_search_phase(self):
        first_upper_idx = self._get_rotate_first_upper_search_state_index()
        if first_upper_idx is None:
            return False

        state_idx = getattr(self, "search_cursor_state_index", None)
        if state_idx is None:
            return True
        try:
            state_idx = int(state_idx)
        except (TypeError, ValueError):
            return True
        return bool(state_idx < int(first_upper_idx))

    def _clear_rotate_target_search_history(self, object_key):
        key = str(object_key)
        state = self.discovered_objects.get(key, None)
        if state is not None:
            state.update(
                {
                    "discovered": False,
                    "visible_now": False,
                    "first_seen_frame": None,
                    "last_seen_frame": None,
                    "last_seen_subtask": 0,
                    "last_seen_stage": 0,
                    "last_uv_norm": None,
                    "last_world_point": None,
                }
            )
        if key in self.visible_objects:
            self.visible_objects[key] = False

    def _prepare_subtask_rotate_search(self, subtask_idx):
        upper_target_keys = self._get_subtask_upper_search_target_keys(subtask_idx)
        for key in upper_target_keys:
            self._clear_rotate_target_search_history(key)
        if len(upper_target_keys) == 0:
            return
        if self._has_unfinished_lower_search_phase():
            return
        first_upper_idx = self._get_rotate_first_upper_search_state_index()
        if first_upper_idx is not None:
            self._set_rotate_search_cursor(state_idx=first_upper_idx, layer_name="upper")

    def _should_enforce_rotate_stage1_search_order(self, subtask_idx, subtask_def=None):
        return bool(self._should_search_lower_before_upper_for_subtask(subtask_idx))

    def _should_skip_rotate_head_home_reset(self, subtask_idx, prev_subtask_idx=None):
        return bool(
            self._should_search_lower_before_upper_for_subtask(subtask_idx)
            and self._has_unfinished_lower_search_phase()
        )

    def _after_rotate_visibility_refresh(self, visibility_map):
        if int(getattr(self, "current_stage", 0)) != 1:
            return None
        subtask_idx = int(getattr(self, "current_subtask_idx", 0))
        if subtask_idx <= 0 or (not self._should_search_lower_before_upper_for_subtask(subtask_idx)):
            return None
        current_layer = self._normalize_layer(getattr(self, "search_cursor_layer", "lower") or "lower")
        if current_layer != "lower":
            return None

        for key in self._get_subtask_upper_search_target_keys(subtask_idx):
            self._clear_rotate_target_search_history(key)
            if isinstance(visibility_map, dict) and key in visibility_map:
                visibility_map[key] = {
                    "visible": False,
                    "u_norm": None,
                    "v_norm": None,
                    "world_point": None,
                }
        return None

    def _get_layer_fixed_head_joint2_target(self, layer_name):
        layer_name = self._normalize_layer(layer_name)
        if layer_name == "upper":
            return float(
                getattr(
                    self,
                    "PLACE_TARGET_UPPER_HEAD_JOINT2_TARGET",
                    getattr(
                        self,
                        "PLACE_PLATE_UPPER_HEAD_JOINT2_TARGET",
                        getattr(self, "rotate_stage1_upper_head_joint2_rad", 0.8),
                    ),
                )
            )

        lower_target = getattr(
            self,
            "PLACE_TARGET_LOWER_HEAD_JOINT2_TARGET",
            getattr(self, "PLACE_PLATE_LOWER_HEAD_JOINT2_TARGET", None),
        )
        if lower_target is None:
            lower_target = getattr(self, "rotate_stage1_lower_head_joint2_rad", 1.22)
        return float(lower_target)

    def _maybe_reset_head_to_home_for_subtask(self, subtask_idx, prev_subtask_idx=None):
        if bool(self._should_skip_rotate_head_home_reset(subtask_idx, prev_subtask_idx=prev_subtask_idx)):
            if bool(getattr(self, "fixed_layer_head_joint2_only", False)):
                return self._move_head_to_rotate_search_layer(
                    "lower",
                    save_freq=self.HEAD_RESET_SAVE_FREQ,
                )
            return True
        if self._subtask_requires_head_home_reset(subtask_idx, prev_subtask_idx=prev_subtask_idx):
            if bool(getattr(self, "fixed_layer_head_joint2_only", False)):
                current_layers = self._get_subtask_search_layers(subtask_idx)
                if current_layers is not None and len(current_layers) == 1:
                    return self._move_head_to_rotate_search_layer(
                        next(iter(current_layers)),
                        save_freq=self.HEAD_RESET_SAVE_FREQ,
                    )
            return self._reset_head_to_home_pose(save_freq=self.HEAD_RESET_SAVE_FREQ)
        return True

    def _get_robot_root_xy_yaw(self):
        root_xy = self.robot.left_entity_origion_pose.p[:2].tolist()
        yaw = float(t3d.euler.quat2euler(self.robot.left_entity_origion_pose.q)[2])
        return root_xy, yaw

    @staticmethod
    def _pose_to_matrix(pose_like):
        if isinstance(pose_like, sapien.Pose):
            return pose_like.to_transformation_matrix()
        pose_arr = np.array(pose_like, dtype=np.float64).reshape(-1)
        if pose_arr.shape[0] != 7:
            raise ValueError(f"pose_like must contain 7 values, got shape {pose_arr.shape}")
        return sapien.Pose(pose_arr[:3], pose_arr[3:]).to_transformation_matrix()

    @staticmethod
    def _matrix_to_pose_list(matrix):
        matrix = np.array(matrix, dtype=np.float64).reshape(4, 4)
        quat = t3d.quaternions.mat2quat(matrix[:3, :3])
        return matrix[:3, 3].tolist() + quat.tolist()

    @staticmethod
    def _project_xy_into_disk(target_xy, center_xy, radius):
        target_xy = np.array(target_xy, dtype=np.float64).reshape(2)
        center_xy = np.array(center_xy, dtype=np.float64).reshape(2)
        radius = float(radius)
        if radius <= 1e-9:
            return center_xy.copy()

        delta_xy = target_xy - center_xy
        delta_norm = float(np.linalg.norm(delta_xy))
        if delta_norm <= radius or delta_norm <= 1e-9:
            return target_xy
        return center_xy + (radius / delta_norm) * delta_xy

    def _get_block_size_for_key(self, block_key=None, block=None):
        key = None if block_key is None else str(block_key)
        block_keys = list(getattr(self, "block_keys", []) or [])
        block_sizes = list(getattr(self, "block_sizes", []) or [])
        if key is not None and key in block_keys:
            block_idx = block_keys.index(key)
            if 0 <= block_idx < len(block_sizes):
                return float(block_sizes[block_idx])

        config = getattr(block, "config", {}) if block is not None else {}
        scale = np.array(config.get("scale", []), dtype=np.float64).reshape(-1)
        if scale.shape[0] >= 3:
            return float(np.max(np.abs(scale[:3])))
        return float(np.mean(np.array(self.BLOCK_SIZE_RANGE, dtype=np.float64)))

    def _get_plate_drop_inner_radius(self, block_key=None, block=None):
        plate_radius = 0.11
        plate = self.object_registry.get("B", None)
        if plate is not None:
            plate_cfg = getattr(plate, "config", {}) or {}
            extents = np.array(plate_cfg.get("extents", []), dtype=np.float64).reshape(-1)
            scale = np.array(plate_cfg.get("scale", []), dtype=np.float64).reshape(-1)
            if extents.shape[0] >= 3 and scale.shape[0] >= 3:
                plate_radius = float(max(abs(extents[0] * scale[0]), abs(extents[2] * scale[2])) / 2.0)

        block_size = self._get_block_size_for_key(block_key=block_key, block=block)
        block_footprint_radius = float(np.sqrt(2.0) * block_size)
        inner_margin = float(getattr(self, "UPPER_TO_LOWER_DROP_PLATE_INNER_MARGIN", 0.010))
        return max(0.01, plate_radius - block_footprint_radius - inner_margin)

    def _get_safe_plate_drop_target_pose(self, block_key=None, block=None):
        target_pose = np.array(self._get_plate_place_target_pose(block_key), dtype=np.float64).reshape(-1)
        plate = self.object_registry.get("B", None)
        if plate is None:
            return target_pose.tolist()

        try:
            plate_center_xy = np.array(plate.get_functional_point(0, "pose").p[:2], dtype=np.float64).reshape(2)
        except Exception:
            plate_center_xy = np.array(self.plate_target_pose[:2], dtype=np.float64).reshape(2)

        safe_radius = self._get_plate_drop_inner_radius(block_key=block_key, block=block)
        target_pose[:2] = self._project_xy_into_disk(target_pose[:2], plate_center_xy, safe_radius)
        return target_pose.tolist()

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

        r_min = min(max(inner_radius + inner_margin, inner_radius + 0.05), outer_radius - 0.08)
        r_cap = min(max_cyl_r, outer_radius - outer_margin)
        r_max = max(r_min, r_cap)
        if (
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
        for _ in range(120):
            block_pose = rand_pose_cyl(
                rlim=layer_spec["rlim"],
                thetalim=layer_spec["thetalim"],
                zlim=[layer_spec["top_z"] + float(size), layer_spec["top_z"] + float(size)],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.75],
            )
            if not self._is_valid_block_spawn_pose(
                block_pose,
                existing_pose_lst=existing_pose_lst,
                avoid_pose_lst=avoid_pose_lst,
                avoid_min_dist_sq=avoid_min_dist_sq,
            ):
                continue
            return block_pose

        fallback_r_candidates = [
            float(np.mean(layer_spec["rlim"])),
            float(layer_spec["rlim"][0]),
            float(layer_spec["rlim"][1]),
        ]
        fallback_theta_candidates = [
            float(np.mean(layer_spec["thetalim"])),
            float(layer_spec["thetalim"][0]),
            float(layer_spec["thetalim"][1]),
        ]
        spawn_z = float(layer_spec["top_z"] + float(size))
        fallback_pose = None
        for fallback_r in fallback_r_candidates:
            for fallback_theta in fallback_theta_candidates:
                candidate_pose = rand_pose_cyl(
                    rlim=[fallback_r, fallback_r],
                    thetalim=[fallback_theta, fallback_theta],
                    zlim=[spawn_z, spawn_z],
                    robot_root_xy=self.robot_root_xy,
                    robot_yaw_rad=self.robot_yaw,
                    qpos=[1, 0, 0, 0],
                    rotate_rand=False,
                )
                fallback_pose = candidate_pose
                if self._is_valid_block_spawn_pose(
                    candidate_pose,
                    existing_pose_lst=existing_pose_lst,
                    avoid_pose_lst=avoid_pose_lst,
                    avoid_min_dist_sq=avoid_min_dist_sq,
                ):
                    return candidate_pose
        return fallback_pose

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
        target_layer = self._get_target_layer()
        if layer_name != target_layer:
            return []
        return [self._get_target_anchor_pose(target_layer)]

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

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self._apply_task_initial_homestate()

        self.block_count = self._get_block_count()
        self.block_layers = list(self._get_block_layers())
        self.blocks = []
        self.block_keys = []
        self.block_sizes = []
        self.block_poses = []
        self.block_colors = self._sample_block_colors(self.block_count)
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
                color=self.block_colors[block_idx],
                name=f"block_{block_idx}",
            )
            block.set_mass(0.02)
            self.blocks.append(block)
            self.block_keys.append(block_key)
            self.block_sizes.append(block_size)
            self.block_poses.append(block_pose)
            existing_pose_lst.append(block_pose)

        # 兼容旧调试代码：单 block 入口仍指向第一个 block。
        self.block = self.blocks[0]
        self.block_size = self.block_sizes[0]
        self.block_pose = self.block_poses[0]

        self._create_target_anchor()
        self.plate_place_slot_assignments = {}
        self.object_layers = {key: layer for key, layer in zip(self.block_keys, self.block_layers)}
        self.object_layers[self._get_target_object_key()] = self._get_target_layer()
        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.05)
        self.add_prohibit_area(self._get_target_object(), padding=self._get_target_padding())
        self._configure_rotate_subtask_plan()
        self._prime_known_target_cache()

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

    def _lift_block_to_place_ready_pose(self, arm_tag):
        if not self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.PICK_LIFT_Z)):
            return False
        if float(self.POST_GRASP_EXTRA_LIFT_Z) > 1e-9 and self.plan_success:
            return bool(self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.POST_GRASP_EXTRA_LIFT_Z)))
        return True

    def _get_tcp_pose(self, arm_tag):
        if ArmTag(arm_tag) == "left":
            return np.array(self.robot.get_left_tcp_pose(), dtype=np.float64).reshape(-1)
        return np.array(self.robot.get_right_tcp_pose(), dtype=np.float64).reshape(-1)

    def _planner_pose_from_tcp_pose(self, tcp_pose):
        tcp_pose = np.array(tcp_pose, dtype=np.float64).reshape(-1)
        planner_pos = tcp_pose[:3] - t3d.quaternions.quat2mat(tcp_pose[3:]) @ np.array(
            [float(self.DIRECT_RELEASE_TCP_BACKOFF), 0.0, 0.0],
            dtype=np.float64,
        )
        return planner_pos.tolist() + tcp_pose[3:].tolist()

    def _get_direct_release_entry_r(self, release_r, target_layer=None):
        explicit_entry_r = getattr(self, "DIRECT_RELEASE_ENTRY_TCP_CYL_R", None)
        if explicit_entry_r is not None:
            return float(explicit_entry_r)

        entry_r = float(release_r) - 0.15
        if target_layer is None:
            target_layer = getattr(self, "plate_layer", None)
        if target_layer is None:
            target_layer = self._get_plate_layer()
        if str(target_layer).lower() == "upper":
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

    def _build_release_pose_candidates_for_target(
        self,
        target_pose,
        target_layer,
        tcp_z_offset,
        entry_tcp_z_offset,
        approach_tcp_z_offset,
        r_offsets,
        theta_offsets_deg,
        yaw_offsets_deg,
    ):
        target_pose = np.array(target_pose, dtype=np.float64).reshape(-1)
        target_xyz = np.array(target_pose[:3], dtype=np.float64)
        target_cyl = world_to_robot(target_xyz.tolist(), self.robot_root_xy, self.robot_yaw)
        root_xy = np.array(self.robot_root_xy, dtype=np.float64)
        outward_yaw = float(np.arctan2(target_xyz[1] - root_xy[1], target_xyz[0] - root_xy[0]))
        yaw_candidates = [float(outward_yaw + np.deg2rad(offset)) for offset in yaw_offsets_deg]

        candidates = []
        release_z = float(target_xyz[2] + tcp_z_offset)
        entry_z = max(float(target_xyz[2] + entry_tcp_z_offset), release_z)
        approach_z = max(float(target_xyz[2] + approach_tcp_z_offset), release_z)
        entry_theta_rad = float(target_cyl[1])
        for r_offset in r_offsets:
            release_r = float(target_cyl[0] + float(r_offset))
            entry_r = self._get_direct_release_entry_r(release_r, target_layer=target_layer)
            for theta_offset_deg in theta_offsets_deg:
                release_theta_rad = float(target_cyl[1] + np.deg2rad(float(theta_offset_deg)))
                for yaw in yaw_candidates:
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

    def _build_direct_release_pose_candidates(self, arm_tag, block_key=None):
        plate_layer = getattr(self, "object_layers", {}).get("B", None)
        if plate_layer is None:
            plate_layer = self._get_plate_layer()
        return self._build_release_pose_candidates_for_target(
            target_pose=self._get_plate_place_target_pose(block_key),
            target_layer=plate_layer,
            tcp_z_offset=self.DIRECT_RELEASE_TCP_Z_OFFSET,
            entry_tcp_z_offset=self.DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET,
            approach_tcp_z_offset=self.DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET,
            r_offsets=self.DIRECT_RELEASE_R_OFFSETS,
            theta_offsets_deg=self.DIRECT_RELEASE_THETA_OFFSETS_DEG,
            yaw_offsets_deg=self.DIRECT_RELEASE_YAW_OFFSETS_DEG,
        )

    def _build_upper_to_lower_hover_release_pose_candidates(self, block, arm_tag, block_key):
        target_pose = np.array(self._get_plate_place_target_pose(block_key), dtype=np.float64).reshape(-1)
        safe_target_pose = np.array(
            self._get_safe_plate_drop_target_pose(block_key=block_key, block=block),
            dtype=np.float64,
        ).reshape(-1)
        plate = self.object_registry.get("B", None)
        if plate is not None:
            try:
                plate_center_xy = np.array(plate.get_functional_point(0, "pose").p[:2], dtype=np.float64).reshape(2)
            except Exception:
                plate_center_xy = np.array(self.plate_target_pose[:2], dtype=np.float64).reshape(2)
        else:
            plate_center_xy = np.array(safe_target_pose[:2], dtype=np.float64).reshape(2)

        safe_radius = float(self._get_plate_drop_inner_radius(block_key=block_key, block=block))
        actor_pose_mat = block.get_pose().to_transformation_matrix()
        fp_pose_mat = np.array(block.get_functional_point(0, "matrix"), dtype=np.float64).reshape(4, 4)
        ee_pose = np.array(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(-1)
        ee_pose_mat = self._pose_to_matrix(ee_pose)
        actor_to_fp = np.linalg.inv(actor_pose_mat) @ fp_pose_mat
        # move_to_pose() consumes the arm EE/planner pose, not the TCP pose.
        # Preserve the current block->EE transform so the solved hover pose lands
        # the block functional point at the configured plate slot without the
        # extra TCP front offset.
        actor_to_ee = np.linalg.inv(actor_pose_mat) @ ee_pose_mat

        base_x_axis = np.array(actor_pose_mat[:2, 0], dtype=np.float64).reshape(2)
        base_x_axis_norm = float(np.linalg.norm(base_x_axis))
        if base_x_axis_norm <= 1e-9:
            base_x_axis = np.array(safe_target_pose[:2], dtype=np.float64) - plate_center_xy
            base_x_axis_norm = float(np.linalg.norm(base_x_axis))
        if base_x_axis_norm <= 1e-9:
            base_yaw = 0.0
        else:
            base_yaw = float(np.arctan2(base_x_axis[1], base_x_axis[0]))

        candidates = []
        yaw_offsets_deg = tuple(getattr(self, "UPPER_TO_LOWER_DROP_YAW_OFFSETS_DEG", (0.0, 90.0, -90.0, 180.0)))
        hover_z_offsets = tuple(getattr(self, "UPPER_TO_LOWER_HOVER_Z_OFFSETS", (0.10, 0.12, 0.14)))
        actor_to_fp_inv = np.linalg.inv(actor_to_fp)
        for yaw_offset_deg in yaw_offsets_deg:
            actor_yaw = float(base_yaw + np.deg2rad(float(yaw_offset_deg)))
            actor_rot = t3d.euler.euler2mat(0.0, 0.0, actor_yaw)
            fp_rot = actor_rot @ actor_to_fp[:3, :3]
            for hover_z in hover_z_offsets:
                target_fp_pose_mat = np.eye(4, dtype=np.float64)
                target_fp_pose_mat[:3, :3] = fp_rot
                target_fp_pose_mat[:3, 3] = np.array(safe_target_pose[:3], dtype=np.float64) + np.array(
                    [0.0, 0.0, float(hover_z)],
                    dtype=np.float64,
                )

                hover_actor_pose_mat = target_fp_pose_mat @ actor_to_fp_inv
                predicted_fp_pose_mat = hover_actor_pose_mat @ actor_to_fp
                predicted_drop_xy = np.array(predicted_fp_pose_mat[:2, 3], dtype=np.float64).reshape(2)
                if float(np.linalg.norm(predicted_drop_xy - plate_center_xy)) > safe_radius + 1e-9:
                    continue

                hover_ee_pose = self._matrix_to_pose_list(hover_actor_pose_mat @ actor_to_ee)
                candidates.append(
                    {
                        "hover_pose": hover_ee_pose,
                        "predicted_drop_xy_error": float(
                            np.linalg.norm(predicted_drop_xy - np.array(target_pose[:2], dtype=np.float64))
                        ),
                        "predicted_drop_inner_margin": float(
                            safe_radius - np.linalg.norm(predicted_drop_xy - plate_center_xy)
                        ),
                    }
                )

        if len(candidates) == 0:
            fallback_candidates = []
            current_fp_rot = np.array(fp_pose_mat[:3, :3], dtype=np.float64).reshape(3, 3)
            for hover_z in hover_z_offsets:
                target_fp_pose_mat = np.eye(4, dtype=np.float64)
                target_fp_pose_mat[:3, :3] = current_fp_rot
                target_fp_pose_mat[:3, 3] = np.array(safe_target_pose[:3], dtype=np.float64) + np.array(
                    [0.0, 0.0, float(hover_z)],
                    dtype=np.float64,
                )
                hover_actor_pose_mat = target_fp_pose_mat @ np.linalg.inv(actor_to_fp)
                hover_ee_pose = self._matrix_to_pose_list(hover_actor_pose_mat @ actor_to_ee)
                fallback_candidates.append({"hover_pose": hover_ee_pose})
            return fallback_candidates

        candidates.sort(
            key=lambda candidate: (
                float(candidate.get("predicted_drop_xy_error", 0.0)),
                -float(candidate.get("predicted_drop_inner_margin", 0.0)),
            )
        )
        return [{"hover_pose": candidate["hover_pose"]} for candidate in candidates]

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

            entry_tcp_pose = entry_pos.tolist() + quat
            pre_grasp_tcp_pose = pre_grasp_pos.tolist() + quat
            grasp_tcp_pose = grasp_pos.tolist() + quat
            candidates.append(
                {
                    "entry_tcp_pose": entry_tcp_pose,
                    "entry_planner_pose": self._planner_pose_from_tcp_pose(entry_tcp_pose),
                    "pre_grasp_tcp_pose": pre_grasp_tcp_pose,
                    "pre_grasp_planner_pose": self._planner_pose_from_tcp_pose(pre_grasp_tcp_pose),
                    "grasp_tcp_pose": grasp_tcp_pose,
                    "grasp_planner_pose": self._planner_pose_from_tcp_pose(grasp_tcp_pose),
                }
            )
        return candidates

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

    def _retreat_after_lower_place(self, arm_tag):
        if not self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=self.LOWER_PLACE_RETREAT_Z,
                move_axis=self.LOWER_PLACE_RETREAT_MOVE_AXIS,
            )
        ):
            return False
        if not bool(self.RETURN_TO_HOMESTATE_AFTER_PLACE):
            return True
        return self._return_both_arms_to_initial_pose()

    def _retreat_after_upper_to_lower_drop_release(self, arm_tag):
        if not self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=self.UPPER_TO_LOWER_RELEASE_RETREAT_Z,
                move_axis="world",
            )
        ):
            return False
        if not bool(self.RETURN_TO_HOMESTATE_AFTER_PLACE):
            return True
        return self._return_both_arms_to_initial_pose()

    def _retreat_then_return_both_arms_to_initial_pose(self, arm_tag):
        if not self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.DIRECT_RELEASE_RETREAT_Z, move_axis="world")):
            return False
        plate_layer = getattr(self, "object_layers", {}).get("B", None)
        if plate_layer is None:
            plate_layer = getattr(self, "plate_layer", None)
        if plate_layer == "upper":
            lateral_xy = self._get_upper_place_lateral_escape_xy(arm_tag)
            if lateral_xy is not None and (
                abs(float(lateral_xy[0])) > 1e-9 or abs(float(lateral_xy[1])) > 1e-9
            ):
                if not self.move(
                    self.move_by_displacement(
                        arm_tag=arm_tag,
                        x=float(lateral_xy[0]),
                        y=float(lateral_xy[1]),
                        move_axis="world",
                    )
                ):
                    return False
        if not bool(self.RETURN_TO_HOMESTATE_AFTER_PLACE):
            return True
        return self._return_both_arms_to_initial_pose()

    def _return_both_arms_to_initial_pose(self):
        return bool(self.move(self.back_to_origin("left"), self.back_to_origin("right")))

    def _prepare_target_subtask(self, subtask_idx, scan_z):
        return self.search_and_focus_rotate_and_head_subtask(
            subtask_idx,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )

    def _prepare_plate_subtask(self, subtask_idx, scan_z):
        return self._prepare_target_subtask(subtask_idx, scan_z)

    def _get_target_focus_world_point(self, target_key):
        obj = self.object_registry.get(str(target_key), None)
        if obj is not None:
            try:
                return np.array(self._resolve_object_world_point(obj=obj), dtype=np.float64).reshape(3)
            except Exception:
                pass
        target_pose = getattr(self, "target_place_pose", None)
        if target_pose is not None:
            return np.array(target_pose[:3], dtype=np.float64).reshape(3)
        if getattr(self, "plate_target_pose", None) is not None:
            return np.array(self.plate_target_pose[:3], dtype=np.float64).reshape(3)
        target_object = self._get_target_object()
        if target_object is not None:
            try:
                return np.array(self._resolve_object_world_point(obj=target_object), dtype=np.float64).reshape(3)
            except Exception:
                pass
        return np.zeros(3, dtype=np.float64)

    def _get_plate_focus_world_point(self, plate_key):
        return self._get_target_focus_world_point(plate_key)

    def _move_head_joint2_for_target_focus(self, target_key, subtask_idx):
        head_joint2_name = getattr(self, "rotate_head_joint2_name", "astribot_head_joint_2")
        head_joint2_idx = self._get_head_joint2_index(head_joint2_name=head_joint2_name)
        head_now = self._get_head_joint_state_now()
        if head_joint2_idx is None or head_now is None:
            self._refresh_rotate_discovery_from_current_view()
            return bool(self.visible_objects.get(str(target_key), False))

        world_point = self._get_target_focus_world_point(target_key)
        solve_res = self.solve_head_lookat_joint_target(world_point=world_point)
        head_target = np.array(head_now, dtype=np.float64)
        target_layer = getattr(self, "object_layers", {}).get(str(target_key), None)
        if target_layer is None:
            target_layer = self._get_target_layer()
        target_joint2 = self._get_layer_fixed_head_joint2_target(target_layer)
        if (not bool(getattr(self, "fixed_layer_head_joint2_only", False))) and solve_res is not None:
            solved_head_target = np.array(solve_res.get("target", []), dtype=np.float64).reshape(-1)
            if solved_head_target.shape[0] > head_joint2_idx:
                if target_layer == "upper":
                    # For upper-layer targets, do not leave head_joint2 lower than the tested upper-view pose.
                    target_joint2 = min(float(solved_head_target[head_joint2_idx]), target_joint2)
                else:
                    target_joint2 = float(solved_head_target[head_joint2_idx])
        head_target[head_joint2_idx] = target_joint2
        clipped_target = self._clip_head_target_to_limits(head_target, default_now=head_now)
        if clipped_target is None:
            clipped_target = head_target

        subtask_def = self._get_rotate_subtask_def(subtask_idx) or {}
        self._set_rotate_subtask_state(
            subtask_idx=subtask_idx,
            stage=2,
            focus_object_key=str(target_key),
            search_target_keys=[str(k) for k in subtask_def.get("search_target_keys", [target_key])],
            action_target_keys=[str(k) for k in subtask_def.get("action_target_keys", [target_key])],
            info_complete=1,
            camera_mode=2,
            camera_target_theta=float(self._get_current_scan_camera_theta() or 0.0),
        )
        if not self.move_head_to(clipped_target, settle_steps=getattr(self, "rotate_stage1_head_settle_steps", 12)):
            return False
        self._refresh_rotate_discovery_from_current_view()
        return bool(self.visible_objects.get(str(target_key), False))

    def _move_head_joint2_for_plate_focus(self, plate_key, subtask_idx):
        return self._move_head_joint2_for_target_focus(plate_key, subtask_idx)

    def _focus_target_before_place(self, subtask_idx, target_key):
        target_key = str(target_key or self._get_target_object_key())
        subtask_def = self._get_rotate_subtask_def(subtask_idx) or {}
        self._align_rotate_registry_target_with_torso_and_head_joint2(
            target_key,
            subtask_idx=subtask_idx,
            target_keys=[str(k) for k in subtask_def.get("search_target_keys", [target_key])],
            action_target_keys=[str(k) for k in subtask_def.get("action_target_keys", [target_key])],
            joint_name_prefer=self.SCAN_JOINT_NAME,
            head_joint2_name=getattr(self, "rotate_head_joint2_name", "astribot_head_joint_2"),
        )
        target_visible = self._move_head_joint2_for_target_focus(target_key, subtask_idx)
        require_visible = getattr(
            self,
            "REQUIRE_TARGET_VISIBLE_BEFORE_PLACE",
            getattr(self, "REQUIRE_PLATE_VISIBLE_BEFORE_PLACE", True),
        )
        if bool(require_visible) and not target_visible:
            self.plan_success = False
            return False
        return True

    def _focus_plate_before_place(self, subtask_idx, plate_key):
        return self._focus_target_before_place(subtask_idx, plate_key)

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
        candidates = self._build_upper_pick_pose_candidates(block, arm_tag)
        selected = self._select_pose_sequence_candidate(
            arm_tag,
            candidates,
            ("entry_planner_pose", "pre_grasp_planner_pose", "grasp_planner_pose"),
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
        if not self._lift_block_to_place_ready_pose(arm_tag):
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
            if block_layer == "upper":
                return self._place_upper_picked_block_into_lower_plate_with_drop_release(arm_tag, subtask_idx, block_key)
            return self._place_block_into_lower_plate_with_place_actor(arm_tag, subtask_idx, block_key)
        return self._place_block_into_upper_plate_with_direct_release(arm_tag, subtask_idx, block_key)

    def _place_block_into_upper_plate_with_direct_release(self, arm_tag, subtask_idx, block_key):
        return self._place_block_into_plate_with_direct_release(arm_tag, subtask_idx, block_key=block_key)

    def _place_upper_picked_block_into_lower_plate_with_drop_release(self, arm_tag, subtask_idx, block_key):
        block = self.object_registry.get(str(block_key), None)
        if block is None:
            self.plan_success = False
            return

        hover_candidates = self._build_upper_to_lower_hover_release_pose_candidates(block, arm_tag, block_key)
        selected_hover_candidate = self._select_pose_sequence_candidate(
            arm_tag,
            hover_candidates,
            ("hover_pose",),
        )
        if selected_hover_candidate is None:
            self.plan_success = False
            return

        if not self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=selected_hover_candidate["hover_pose"])):
            return
        if not self.move(self.open_gripper(arm_tag)):
            return
        release_delay_steps = int(max(getattr(self, "UPPER_TO_LOWER_RELEASE_DELAY_STEPS", 0), 0))
        if release_delay_steps > 0:
            self.delay(release_delay_steps)
        self._set_carried_object_keys([])
        if not self._retreat_after_upper_to_lower_drop_release(arm_tag):
            return
        self.complete_rotate_subtask(subtask_idx, carried_after=[])

    def _place_block_into_lower_plate_with_place_actor(self, arm_tag, subtask_idx, block_key):
        block = self.object_registry.get(str(block_key), None)
        if block is None:
            self.plan_success = False
            return

        target_pose = self._get_plate_place_target_pose(block_key)
        if not self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=self.LOWER_PLACE_FUNCTIONAL_POINT_ID,
                pre_dis=self.LOWER_PLACE_PRE_DIS,
                dis=self.LOWER_PLACE_DIS,
                pre_dis_axis=self.LOWER_PLACE_PRE_DIS_AXIS,
                constrain=self.LOWER_PLACE_CONSTRAIN,
                is_open=bool(self.LOWER_PLACE_IS_OPEN),
            )
        ):
            return
        self._set_carried_object_keys([])
        if not self._retreat_after_lower_place(arm_tag):
            return
        self.complete_rotate_subtask(subtask_idx, carried_after=[])

    def _place_block_into_plate_with_direct_release(self, arm_tag, subtask_idx, block_key=None):
        release_candidates = self._build_direct_release_pose_candidates(arm_tag, block_key=block_key)
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
        self.complete_rotate_subtask(subtask_idx, carried_after=[])

    def _place_block_into_target(self, arm_tag, subtask_idx, block_key, focus_object_key):
        return self._place_block_into_plate(arm_tag, subtask_idx, block_key, focus_object_key)

    def _build_info(self, arm_tag):
        return {
            "{A}": "block" if len(getattr(self, "block_keys", [])) <= 1 else "blocks",
            "{B}": f"003_plate/base{self.PLATE_MODEL_ID}",
            "{a}": str(arm_tag),
        }

    def play_once(self):
        scan_z = float(self.SCAN_Z_BIAS + self.table_z_bias)
        last_arm_tag = ArmTag("left")
        remaining_block_keys = [str(key) for key in self.block_keys]
        prev_subtask_idx = None

        for block_idx in range(len(self.block_keys)):
            pick_subtask_idx = 2 * block_idx + 1
            place_subtask_idx = pick_subtask_idx + 1
            self._prepare_dynamic_pick_subtask(pick_subtask_idx, remaining_block_keys)

            self._prepare_subtask_rotate_search(pick_subtask_idx)
            self._maybe_reset_head_to_home_for_subtask(pick_subtask_idx, prev_subtask_idx=prev_subtask_idx)
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

            arm_tag = self._pick_block(pick_subtask_idx, block_key)
            last_arm_tag = arm_tag
            if not self.plan_success:
                self.info["info"] = self._build_info(arm_tag)
                return self.info
            prev_subtask_idx = pick_subtask_idx

            self._prepare_dynamic_place_subtask(place_subtask_idx, block_key)
            self._prepare_subtask_rotate_search(place_subtask_idx)
            self._maybe_reset_head_to_home_for_subtask(place_subtask_idx, prev_subtask_idx=prev_subtask_idx)
            target_key = self._prepare_target_subtask(place_subtask_idx, scan_z)
            if target_key is None:
                self.plan_success = False
                self.info["info"] = self._build_info(arm_tag)
                return self.info
            if not self._focus_target_before_place(place_subtask_idx, target_key):
                self.info["info"] = self._build_info(arm_tag)
                return self.info

            self._place_block_into_target(arm_tag, place_subtask_idx, block_key, target_key)
            if not self.plan_success:
                self.info["info"] = self._build_info(arm_tag)
                return self.info
            prev_subtask_idx = place_subtask_idx
            remaining_block_keys.remove(block_key)

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


class PutSingleBlockTargetFanDoubleBase(PutBlockTargetFanDoubleBase):
    BLOCK_COUNT = 1
    BLOCK_LAYER_SEQUENCE = ("lower",)
    BLOCK_SIZE_RANGE = (0.018, 0.022)
    BLOCK_COLOR = (0.10, 0.80, 0.20)
    BLOCK_COLOR_CANDIDATES = ((0.10, 0.80, 0.20),)
    FIXED_LAYER_HEAD_JOINT2_ONLY = True
    LOWER_PLACE_PRE_DIS = 0.12
    LOWER_PLACE_DIS = 0.02
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.25

    TARGET_MODEL_NAME = None
    TARGET_MODEL_ID = 0
    TARGET_MODEL_IDS = ()
    TARGET_LAYER = "lower"
    TARGET_LAYER_SPECS = {
        "lower": {
            "r": 0.48,
            "theta_deg": 20.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": None,
        },
        "upper": {
            "r": 0.68,
            "theta_deg": 0.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": None,
        },
    }
    TARGET_FUNCTIONAL_POINT_ID = 0
    TARGET_MASS = 0.05
    TARGET_IS_STATIC = True
    TARGET_PADDING = 0.08
    TARGET_TASK_PREPOSITION = "into"

    PLACE_TARGET_UPPER_HEAD_JOINT2_TARGET = 0.8
    PLACE_TARGET_LOWER_HEAD_JOINT2_TARGET = None
    REQUIRE_TARGET_VISIBLE_BEFORE_PLACE = True

    SUCCESS_XY_TOL = 0.08
    SUCCESS_Z_TOL = 0.06

    def _sample_target_model_id(self):
        if len(getattr(self, "TARGET_MODEL_IDS", ()) or ()) > 0:
            return int(np.random.choice(list(self.TARGET_MODEL_IDS)))
        return int(self.TARGET_MODEL_ID)

    def _get_task_instruction(self):
        preposition = str(getattr(self, "TARGET_TASK_PREPOSITION", "into")).strip() or "into"
        return f"Put the block {preposition} {{B}}."

    def _get_plate_layer(self):
        return self._normalize_layer(getattr(self, "TARGET_LAYER", "lower"))

    def _get_target_layer(self):
        return self._get_plate_layer()

    def _get_plate_layer_spec(self, layer_name=None):
        layer_name = self._get_plate_layer() if layer_name is None else self._normalize_layer(layer_name)
        layer_specs = dict(getattr(self, "TARGET_LAYER_SPECS", {}) or {})
        target_spec = dict(layer_specs.get(layer_name, {}))
        if len(target_spec) == 0:
            raise ValueError(f"Missing TARGET_LAYER_SPECS entry for layer: {layer_name}")
        layer_spec = self._get_layer_spec(layer_name)
        return {
            "layer": layer_name,
            "r": float(target_spec.get("r", 0.48 if layer_name == "lower" else 0.68)),
            "theta_deg": float(target_spec.get("theta_deg", 0.0)),
            "z": float(layer_spec["top_z"]) + float(target_spec.get("z_offset", 0.0)),
            "qpos": list(target_spec.get("qpos", [0.5, 0.5, 0.5, 0.5])),
            "scale": target_spec.get("scale", None),
        }

    def _get_target_layer_spec(self, layer_name=None):
        return self._get_plate_layer_spec(layer_name)

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

    def _get_target_anchor_pose(self, layer_name=None):
        return self._get_plate_anchor_pose(layer_name)

    def _get_target_padding(self):
        return float(getattr(self, "TARGET_PADDING", 0.08))

    def _create_plate_anchor(self):
        self.plate_layer = self._get_plate_layer()
        target_spec = self._get_plate_layer_spec(self.plate_layer)
        self.target_name = str(getattr(self, "TARGET_MODEL_NAME", "")).strip()
        if not self.target_name:
            raise ValueError("TARGET_MODEL_NAME must be set for single-target block tasks.")
        self.target_id = self._sample_target_model_id()
        self.plate_cyl_r = float(target_spec["r"])
        self.plate_cyl_theta_deg = float(target_spec["theta_deg"])
        self.plate_z = float(target_spec["z"])
        self.plate_qpos = list(target_spec["qpos"])
        target_scale = target_spec.get("scale", None)
        self.plate_scale = None if target_scale is None else list(target_scale)

        create_kwargs = {
            "scene": self,
            "pose": self._get_plate_anchor_pose(self.plate_layer),
            "modelname": self.target_name,
            "model_id": self.target_id,
            "convex": True,
            "is_static": bool(getattr(self, "TARGET_IS_STATIC", True)),
        }
        if target_scale is not None:
            create_kwargs["scale"] = list(target_scale)

        self.plate = create_actor(**create_kwargs)
        self.plate.set_mass(float(getattr(self, "TARGET_MASS", 0.05)))
        self.plate_target_pose = self._get_plate_place_target_pose()
        self.target_object = self.plate
        self.target_layer = self.plate_layer
        self.target_place_pose = list(self.plate_target_pose)
        return self.plate

    def _create_target_anchor(self):
        return super()._create_target_anchor()

    def _get_plate_place_target_pose(self, block_key=None):
        plate = self.object_registry.get("B", None)
        if plate is None:
            plate = getattr(self, "plate", None)
        if plate is not None:
            try:
                plate_pose = plate.get_functional_point(int(getattr(self, "TARGET_FUNCTIONAL_POINT_ID", 0)), "pose")
                target_pose = plate_pose.p.tolist() + plate_pose.q.tolist()
            except Exception:
                plate_pose = plate.get_pose()
                target_pose = plate_pose.p.tolist() + plate_pose.q.tolist()
        else:
            target_pose = list(getattr(self, "plate_target_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

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

    def _get_target_place_target_pose(self, block_key=None):
        return self._get_plate_place_target_pose(block_key)

    def _place_block_into_target(self, arm_tag, subtask_idx, block_key, focus_object_key):
        return super()._place_block_into_target(arm_tag, subtask_idx, block_key, focus_object_key)

    def _build_info(self, arm_tag):
        return {
            "{A}": "green block",
            "{B}": f"{self.target_name}/base{self.target_id}",
            "{a}": str(arm_tag),
        }

    def check_success(self):
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        block = self.blocks[0] if len(getattr(self, "blocks", [])) > 0 else getattr(self, "block", None)
        target_object = self._get_target_object()
        if block is None or target_object is None:
            return False

        block_pose = np.array(block.get_functional_point(0, "pose").p, dtype=np.float64).reshape(3)
        target_pose = np.array(
            target_object.get_functional_point(int(getattr(self, "TARGET_FUNCTIONAL_POINT_ID", 0)), "pose").p,
            dtype=np.float64,
        ).reshape(3)
        xy_ok = float(np.linalg.norm(block_pose[:2] - target_pose[:2])) < float(self.SUCCESS_XY_TOL)
        z_ok = float(abs(block_pose[2] - target_pose[2])) < float(self.SUCCESS_Z_TOL)
        on_target = self.check_actors_contact(block.get_name(), target_object.get_name())
        return bool(gripper_open and xy_ok and z_ok and on_target)
