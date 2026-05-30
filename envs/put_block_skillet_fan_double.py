from ._base_task import Base_Task, PutBlockFanDoubleMixin
from .utils import *
import numpy as np


class put_block_skillet_fan_double(PutBlockFanDoubleMixin, Base_Task):
    ROTATE_TABLE_SHAPE = "fan_double"
    ROTATE_TABLE_CONFIG_KEY = "fan_double_left_support"
    BLOCK_COUNT = 2
    BLOCK_LAYER_SEQUENCE = ("lower", "lower")
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

    def _get_target_object_key(self):
        return "B"

    def _get_target_object(self):
        target_object = getattr(self, "target_object", None)
        if target_object is not None:
            return target_object
        return getattr(self, "plate", None)

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

    def _get_block_spawn_avoid_pose_lst(self, layer_name):
        layer_name = self._normalize_layer(layer_name)
        target_layer = self._get_target_layer()
        if layer_name != target_layer:
            return []
        return [self._get_target_anchor_pose(target_layer)]

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

    def _get_target_layer_spec(self, layer_name=None):
        return self._get_plate_layer_spec(layer_name)

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
        return self._create_plate_anchor()

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
        return self._place_block_into_plate(arm_tag, subtask_idx, block_key, focus_object_key)

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

    TARGET_THETA_JITTER_DEG = 5.0
    TARGET_MODEL_NAME = "106_skillet"
    TARGET_MODEL_ID = 0
    TARGET_PADDING = 0.06
    TARGET_TASK_PREPOSITION = "on"
    TARGET_LAYER = "upper"
    TARGET_LAYER_SPECS = {
        "lower": {
            "r": 0.47,
            "theta_deg": 0.0,
            "z_offset": 0.0,
            "qpos": [0.0, 0.0, 0.70710678, 0.70710678],
            "scale": None,
        },
        "upper": {
            "r": 0.68,
            "theta_deg": 5.0,
            "z_offset": 0.0,
            "qpos": [0.0, 0.0, 0.70710678, 0.70710678],
            "scale": None,
        },
    }
    SUCCESS_XY_TOL = 0.06
    SUCCESS_Z_TOL = 0.05

    def setup_demo(self, **kwargs):
        self._target_theta_deg_jitter_cache = {}
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
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
        super()._init_task_env_(**kwargs)

    def _get_plate_layer_spec(self, layer_name=None):
        layer_name = self._get_plate_layer() if layer_name is None else self._normalize_layer(layer_name)
        layer_specs = dict(getattr(self, "TARGET_LAYER_SPECS", {}) or {})
        target_spec = dict(layer_specs.get(layer_name, {}))
        if len(target_spec) == 0:
            raise ValueError(f"Missing TARGET_LAYER_SPECS entry for layer: {layer_name}")
        layer_spec = self._get_layer_spec(layer_name)
        target_spec = {
            "layer": layer_name,
            "r": float(target_spec.get("r", 0.48 if layer_name == "lower" else 0.68)),
            "theta_deg": float(target_spec.get("theta_deg", 0.0)),
            "z": float(layer_spec["top_z"]) + float(target_spec.get("z_offset", 0.0)),
            "qpos": list(target_spec.get("qpos", [0.5, 0.5, 0.5, 0.5])),
            "scale": target_spec.get("scale", None),
        }
        theta_cache = getattr(self, "_target_theta_deg_jitter_cache", None)
        if not isinstance(theta_cache, dict):
            theta_cache = {}
            self._target_theta_deg_jitter_cache = theta_cache
        cache_key = str(target_spec["layer"])
        if cache_key not in theta_cache:
            theta_cache[cache_key] = float(target_spec["theta_deg"]) + float(
                np.random.uniform(-self.TARGET_THETA_JITTER_DEG, self.TARGET_THETA_JITTER_DEG)
            )
        target_spec["theta_deg"] = float(theta_cache[cache_key])
        return target_spec
