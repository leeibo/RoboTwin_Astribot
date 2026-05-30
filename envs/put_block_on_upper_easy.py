from ._base_task import Base_Task, PutBlockFanDoubleMixin
from .utils import *
import numpy as np


class put_block_on_upper_easy(PutBlockFanDoubleMixin, Base_Task):
    ROTATE_TABLE_SHAPE = "fan_double"
    ROTATE_TABLE_CONFIG_KEY = "fan_double_left_support"
    BLOCK_COUNT = 2
    BLOCK_LAYER_SEQUENCE = ("upper", "lower")
    BLOCK_SIZE_RANGE = (0.015, 0.025)
    BLOCK_COLOR = (0.10, 0.80, 0.20)
    BLOCK_COLOR_CANDIDATES = (
        (0.10, 0.80, 0.20),
        (0.90, 0.20, 0.20),
        (0.20, 0.45, 0.92),
        (0.92, 0.74, 0.18),
        (0.88, 0.45, 0.16),
    )
    BLOCK_SPAWN_MIN_DIST_SQ = 0.01
    PLATE_BLOCK_SPAWN_MIN_DIST_SQ = 0.0255
    BLOCK_LAYER_SPECS = {
        "lower": {
            "inner_margin": 0.10,
            "outer_margin": 0.20,
            "max_cyl_r": 0.5,
            "theta_shrink": 0.92,
        },
        "upper": {
            "inner_margin": 0.02,
            "outer_margin": 0.04,
            "max_cyl_r": 0.64,
            "theta_shrink": 0.92,
        },
    }

    # plate anchor 参数：
    # plate 的 z 由对应桌面 top_z + z_offset 计算，避免和 fan_double_layer_gap 不一致。

    # 注意：这里的 theta_deg 只是占位/回退值，不能在这个表里调盘子角度；
    # 实际 lower 角度在 _get_plate_theta_deg() 里采样，upper 角度跟随 fan_double_support_theta_deg。
    PLATE_MODEL_ID = 0
    PLATE_LAYER = "upper"
    PLATE_LAYER_SPECS = {
        "lower": {
            "r": 0.55,
            "theta_deg": -60,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": [0.025, 0.025, 0.025],
        },
        "upper": {
            "r": 0.70,
            "theta_deg": 0.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": [0.025, 0.025, 0.025],
        },
    }
    # plate 内部的目标槽位，单位为相对 plate 中心的 [radial_offset, tangential_offset] 米。
    # radial: plate 相对机器人的径向；tangential: 桌面内与其垂直的切向。
    PLATE_PLACE_SLOT_OFFSETS = {
        1: ((0.0, 0.0),),
        2: ((0.04, 0), (-0.04, 0)),
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
    LOWER_PLACE_PRE_DIS = 0.15
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
    UPPER_PICK_ENTRY_Z_OFFSET = 0.06
    UPPER_PICK_PRE_GRASP_DIS = 0.06
    UPPER_PICK_GRASP_Z_BIAS = 0.02
    PLATE_PLACE_SLOT_OFFSETS = 0.025
    UPPER_PICK_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
    UPPER_PICK_GRIPPER_POS = -0.02

    # 上层放置后先沿当前机器人本体左右方向侧向撤离，再回 homestate。
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.2
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
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _get_plate_layer(self):
        return self._normalize_layer(self.PLATE_LAYER)

    def _get_plate_theta_deg(self, layer_name, plate_spec):
        layer_name = self._normalize_layer(layer_name)
        theta_cache = getattr(self, "_plate_theta_deg_cache", None)
        if not isinstance(theta_cache, dict):
            theta_cache = {}
            self._plate_theta_deg_cache = theta_cache
        if layer_name in theta_cache:
            return float(theta_cache[layer_name])

        if layer_name == "lower":
            theta_deg = float(np.random.choice((-55.0, 55.0)))
        elif layer_name == "upper":
            support_theta_world_rad = getattr(self, "rotate_fan_double_support_theta_world_rad", None)
            if support_theta_world_rad is not None:
                theta_deg = float(
                    np.rad2deg(self._wrap_to_pi(float(support_theta_world_rad) - float(self.robot_yaw)))
                )
            else:
                theta_deg = float(plate_spec.get("theta_deg", 0.0))
        else:
            theta_deg = float(plate_spec.get("theta_deg", 0.0))

        theta_cache[layer_name] = float(theta_deg)
        return float(theta_deg)

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

    def _maybe_reset_head_to_home_for_subtask(self, subtask_idx, prev_subtask_idx=None):
        if self._subtask_requires_head_home_reset(subtask_idx, prev_subtask_idx=prev_subtask_idx):
            return self._reset_head_to_home_pose(save_freq=self.HEAD_RESET_SAVE_FREQ)
        return True

    def _get_plate_layer_spec(self, layer_name=None):
        layer_name = self._get_plate_layer() if layer_name is None else self._normalize_layer(layer_name)
        plate_spec = dict(self.PLATE_LAYER_SPECS.get(layer_name, {}))
        if len(plate_spec) == 0:
            raise ValueError(f"Missing PLATE_LAYER_SPECS entry for layer: {layer_name}")
        layer_spec = self._get_layer_spec(layer_name)
        return {
            "layer": layer_name,
            "r": float(plate_spec.get("r", 0.70 if layer_name == "upper" else 0.55)),
            "theta_deg": self._get_plate_theta_deg(layer_name, plate_spec),
            "z": float(layer_spec["top_z"]) + float(plate_spec.get("z_offset", 0.0)),
            "qpos": list(plate_spec.get("qpos", [0.5, 0.5, 0.5, 0.5])),
            "scale": list(plate_spec.get("scale", [0.025, 0.025, 0.025])),
        }

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

    def _get_plate_place_slot_offsets(self):
        block_count = int(getattr(self, "block_count", len(getattr(self, "block_keys", [])) or 1))
        raw_offsets = getattr(self, "PLATE_PLACE_SLOT_OFFSETS", {}) or {}
        if np.isscalar(raw_offsets):
            spacing = max(float(raw_offsets), 0.0)
            if block_count <= 1 or spacing <= 1e-9:
                return [(0.0, 0.0)]
            if block_count == 2:
                return [(spacing, 0.0), (-spacing, 0.0)]
            angles = np.linspace(0.0, 2.0 * np.pi, block_count, endpoint=False)
            return [(float(spacing * np.cos(angle)), float(spacing * np.sin(angle))) for angle in angles]

        slot_offset_map = dict(raw_offsets)
        slot_offsets = slot_offset_map.get(block_count, None)
        if slot_offsets is None and len(slot_offset_map) > 0:
            nearest_key = min(slot_offset_map.keys(), key=lambda key: abs(int(key) - block_count))
            slot_offsets = slot_offset_map[nearest_key]
        if slot_offsets is None or len(slot_offsets) == 0:
            return [(0.0, 0.0)]
        return [(float(offset[0]), float(offset[1])) for offset in slot_offsets]

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
        self._plate_theta_deg_cache = {}
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

        self._create_plate_anchor()
        self.plate_place_slot_assignments = {}
        self.object_layers = {key: layer for key, layer in zip(self.block_keys, self.block_layers)}
        self.object_layers["B"] = self.plate_layer
        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.05)
        self.add_prohibit_area(self.plate, padding=0.08)
        self._configure_rotate_subtask_plan()
        self._prime_known_target_cache()

    def _prepare_plate_subtask(self, subtask_idx, scan_z):
        return self.search_and_focus_rotate_and_head_subtask(
            subtask_idx,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )

    def _get_plate_focus_world_point(self, plate_key):
        obj = self.object_registry.get(str(plate_key), None)
        if obj is not None:
            try:
                return np.array(self._resolve_object_world_point(obj=obj), dtype=np.float64).reshape(3)
            except Exception:
                pass
        return np.array(self.plate_target_pose[:3], dtype=np.float64).reshape(3)

    def _move_head_joint2_for_plate_focus(self, plate_key, subtask_idx):
        head_joint2_name = getattr(self, "rotate_head_joint2_name", "astribot_head_joint_2")
        head_joint2_idx = self._get_head_joint2_index(head_joint2_name=head_joint2_name)
        head_now = self._get_head_joint_state_now()
        if head_joint2_idx is None or head_now is None:
            self._refresh_rotate_discovery_from_current_view()
            return bool(self.visible_objects.get(str(plate_key), False))

        world_point = self._get_plate_focus_world_point(plate_key)
        solve_res = self.solve_head_lookat_joint_target(world_point=world_point)
        head_target = np.array(head_now, dtype=np.float64)
        plate_layer = getattr(self, "object_layers", {}).get(str(plate_key), None)
        if plate_layer is None:
            plate_layer = self._get_plate_layer()
        if plate_layer == "upper":
            target_joint2 = float(self.PLACE_PLATE_UPPER_HEAD_JOINT2_TARGET)
        else:
            lower_target = self.PLACE_PLATE_LOWER_HEAD_JOINT2_TARGET
            if lower_target is None:
                head_home = np.array(getattr(self.robot, "head_homestate", []), dtype=np.float64).reshape(-1)
                lower_target = (
                    head_home[head_joint2_idx]
                    if head_home.shape[0] > head_joint2_idx
                    else head_now[head_joint2_idx]
                )
            target_joint2 = float(lower_target)
        if solve_res is not None:
            solved_head_target = np.array(solve_res.get("target", []), dtype=np.float64).reshape(-1)
            if solved_head_target.shape[0] > head_joint2_idx:
                if plate_layer == "upper":
                    # For the upper plate, do not leave head_joint2 lower than the tested upper-view pose.
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
            focus_object_key=str(plate_key),
            search_target_keys=[str(k) for k in subtask_def.get("search_target_keys", [plate_key])],
            action_target_keys=[str(k) for k in subtask_def.get("action_target_keys", [plate_key])],
            info_complete=1,
            camera_mode=2,
            camera_target_theta=float(self._get_current_scan_camera_theta() or 0.0),
        )
        if not self.move_head_to(clipped_target, settle_steps=getattr(self, "rotate_stage1_head_settle_steps", 12)):
            return False
        self._refresh_rotate_discovery_from_current_view()
        return bool(self.visible_objects.get(str(plate_key), False))

    def _focus_plate_before_place(self, subtask_idx, plate_key):
        plate_key = str(plate_key or "B")
        subtask_def = self._get_rotate_subtask_def(subtask_idx) or {}
        self._align_rotate_registry_target_with_torso_and_head_joint2(
            plate_key,
            subtask_idx=subtask_idx,
            target_keys=[str(k) for k in subtask_def.get("search_target_keys", [plate_key])],
            action_target_keys=[str(k) for k in subtask_def.get("action_target_keys", [plate_key])],
            joint_name_prefer=self.SCAN_JOINT_NAME,
            head_joint2_name=getattr(self, "rotate_head_joint2_name", "astribot_head_joint_2"),
        )
        plate_visible = self._move_head_joint2_for_plate_focus(plate_key, subtask_idx)
        if bool(self.REQUIRE_PLATE_VISIBLE_BEFORE_PLACE) and not plate_visible:
            self.plan_success = False
            return False
        return True

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
        prev_subtask_idx = None

        for block_idx in range(len(self.block_keys)):
            pick_subtask_idx = 2 * block_idx + 1
            place_subtask_idx = pick_subtask_idx + 1
            self._prepare_dynamic_pick_subtask(pick_subtask_idx, remaining_block_keys)

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
            self._maybe_reset_head_to_home_for_subtask(place_subtask_idx, prev_subtask_idx=prev_subtask_idx)
            plate_key = self._prepare_plate_subtask(place_subtask_idx, scan_z)
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

    PLATE_LAYER = "upper"
    BLOCK_COUNT = 2
    BLOCK_LAYER_SEQUENCE = ("lower", "lower")
