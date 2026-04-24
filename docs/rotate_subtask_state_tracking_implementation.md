# Rotate View 子任务状态跟踪实现说明

## 1. 这次实现的目标

这次改动的核心目标，是把原来 rotate-view 任务里“先扫完整个场景，再一口气执行整段动作”的流程，改成符合 `target.md` 描述的子任务流程：

1. 一个任务被拆成多个 `subtask`
2. 每个 `subtask` 都由两部分组成：
   - 主动感知与转动搜索
   - 对应的一段机械臂动作
3. 搜索过程需要区分阶段：
   - `stage 1`：粗搜索
   - `stage 2`：精定位
   - `stage 3`：动作执行
4. 每一帧都要带状态标签，后续可直接用于 VLA / VLM 数据构造
5. 重点先覆盖 `collect_rotate_tasks_multi_difficulty_whitelist.sh` 使用到的白名单任务

这次实现已经把白名单中的 rotate-view 任务全部接入新的子任务状态流。

## 2. 我具体实现了什么

### 2.1 在 `Base_Task` 中实现了统一的子任务运行时

文件：
- `envs/_base_task.py`

我在 `Base_Task` 里加入了一套共享的 rotate 子任务运行时，让所有白名单任务都走同一套状态机，而不是每个任务自己单独处理。

新增的核心能力有：

1. 子任务配置入口
   - `configure_rotate_subtask_plan(...)`
   - 用于在每个任务里注册：
     - `object_registry`
     - `subtask_defs`
   - 子任务文本不再直接写在 env 文件里，而是从：
     - `description/task_instruction/<task>.json`
     - `subtask_instruction_template_map`
     读取模板，再按 episode 的 `scene_info["info"]` 做占位符解析

2. 子任务切换接口
   - `begin_rotate_subtask(...)`
   - `enter_rotate_action_stage(...)`
   - `complete_rotate_subtask(...)`

3. 搜索与精定位接口
   - `search_and_focus_rotate_subtask(...)`
   - 该接口会自动处理：
     - 先检查当前画面里目标是否已经可见；只有“当前帧已看见目标”时才进入 `stage 2`
     - 若当前帧还没看到目标，则进入 `stage 1` 的离散单位角扫描
     - `stage 1` 每次只转一个固定单位角，并在每一步结束后重新检查目标是否出现
     - 若当前扫描侧的扇形桌边缘两端点已经同时进入视野，但目标仍未出现，则立即反向扫描
     - `stage 2` 不再直接用 GT 目标点，而是依据当前图像里的 `u` 偏差做细调

4. 物体记忆机制
   - 维护“已经发现过的物体”和“当前视野可见的物体”
   - 这些信息现在主要用于逐帧标注、sidecar 元数据和后续 VLM/VLA 数据构造
   - 当前这版采集中，不再因为“之前见过”就直接跳过 `stage 1`

5. 侧边元数据保存
   - `save_rotate_subtask_metadata(...)`
   - 会把子任务定义、转换日志、每帧注释信息保存成单独 JSON

### 2.2 加入了每帧子任务标注

文件：
- `envs/_base_task.py`

现在每一帧都会写入新的根级 HDF5 数据集，重点包括：

- `subtask`
- `stage`
- `subtask_instruction_idx`
- `focus_object_idx`
- `focus_object_visible`
- `info_complete`
- `camera_mode`
- `camera_target_theta`
- `visible_object_mask`
- `discovered_object_mask`
- `search_target_mask`
- `action_target_mask`
- `carried_object_mask`
- `target_uv_norm`

这些字段的目的，是把“当前帧属于哪个子任务、当前在粗搜/精搜/动作哪个阶段、现在正在找什么、手里拿着什么、目标是否在画面里”等信息直接写进训练数据。

### 2.3 加入了 episode 级 sidecar 元数据

文件：
- `envs/_base_task.py`
- `script/collect_data.py`

每个 episode 现在会额外生成：

- `data/<task>/<setting>/subtask_metadata/episodeX.json`

里面包含：

- `task_instruction`
- `object_key_to_idx`
- `object_key_to_name`
- `subtask_instruction_map`
- `subtask_instruction_template_map`
- `subtask_defs`
- `transition_log`
- `frame_annotations`
- `final_discovered_objects`

这里的 `task_instruction` 和 `subtask_instruction_map` 现在都不是 env 中手写死的最终字符串，而是按原始 instruction 生成链路解析后的 episode 级文本：

1. 先读取 `description/task_instruction/<task>.json`
2. 使用和 `description/utils/generate_episode_instructions.py` 相同的 placeholder 过滤与替换逻辑
3. 结合当前 episode 的 `scene_info["info"]`
4. 输出当前 episode 对应的具体 task / subtask instruction

同时，`scene_info.json` 里也会记录：

- `subtask_metadata_path`

这里的写法不是把 rotate 元数据整份塞进 `scene_info.json`，而是按 episode 记录索引：

- `scene_info.json -> episode_0 -> subtask_metadata_path`
- `scene_info.json -> episode_1 -> subtask_metadata_path`
- `scene_info.json -> episode_2 -> subtask_metadata_path`

真正的子任务定义、切换日志和逐帧注释，都保存在：

- `data/<task>/<setting>/subtask_metadata/episodeX.json`

这样后续无论是做数据清洗、构造 VLM 样本，还是中途停止后恢复检查，都有单独的结构化元数据可用。

### 2.4 加入了世界坐标到图像坐标的投影工具

文件：
- `envs/utils/camera_visibility.py`
- `test/test_camera_visibility.py`

新增了：

- `project_world_point_to_image_uv(...)`

作用是把目标物体投影到图像上，并得到归一化像素位置，用于：

- 判断目标是否在当前画面中
- 填写 `target_uv_norm`
- 支持后续 VLM 文本中“当前目标位于图像中的哪里”的描述

对应测试也补了：

- 中心点投影测试
- 左偏点投影测试

### 2.5 修复了一个真实运行时 bug

文件：
- `envs/_base_task.py`

一开始在真实采集时，`click_bell_rotate_view` 会报：

- `ValueError: Unknown rotate subtask id: 1`

原因不是任务定义缺失，而是生命周期顺序有问题：

1. 任务在 `load_actors()` 里已经调用了 `configure_rotate_subtask_plan(...)`
2. 但 `Base_Task._init_task_env_()` 在后面又重新调用了一次 `_init_rotate_subtask_runtime_state()`
3. 导致刚注册好的 `subtask_defs`、`subtask_def_map` 被清空

我已经把初始化顺序修正为：

1. 先初始化 rotate 子任务运行时
2. 再 `load_actors()`
3. 由每个任务在 `load_actors()` 结束时注册自己的子任务 plan

这个 bug 修完后，真实采集 smoke run 已能正常通过。

### 2.6 增加了每个 episode 的带注释视频导出

文件：
- `envs/utils/pkl2hdf5.py`
- `envs/_base_task.py`
- `script/collect_data.py`

现在 rotate 子任务 episode 在导出原始 HDF5 / 原始视频的同时，还会额外生成一个主视角注释视频：

- `video/episodeX_annotated.mp4`

这个视频直接基于 `camera_head` RGB 序列生成，画面上会叠加：

- 当前 `frame`
- 当前 `stage`
- 当前 `subtask`
- 当前 `sub task instruction`
- 当前已经找到的物体列表

实现方式是：

1. 复用现有 `saved_frame_annotations`
2. 复用 episode 级解析后的 `subtask_instruction_map`
3. 复用 `object_key_to_name`
4. 在 `process_folder_to_hdf5_video(...)` 阶段统一做带半透明信息面板的 overlay

这样做的好处是：

1. 不改原始 HDF5 结构
2. 不影响 headless 采集
3. 仍然保留原始 `episodeX.mp4` 和多视角视频
4. 额外给出一份可直接人工检查子任务切换是否正确的可视化结果

同时，下面两个位置现在也会记录注释视频路径：

- `scene_info.json -> episode_X -> annotated_video_path`
- `subtask_metadata/episodeX.json -> annotated_video_path`

## 3. 任务里具体新增了哪些字段

这一部分是本次实现里最关键的“状态和子任务切换字段”。

### 3.1 每个任务都要声明的 plan 字段

每个 rotate-view 任务现在都新增了：

- `object_registry`
  - 把任务中关心的对象注册成统一 key，例如 `A`、`B`、`C`
- `subtask_defs`
  - 逐个定义子任务

每个 rotate-view task 对应的 `description/task_instruction/<task>.json` 现在还新增了：

- `subtask_instruction_template_map`
  - 把 `instruction_idx` 映射到子任务 instruction 模板
  - 模板仍然使用 `{A}`、`{B}`、`{a}` 这类 placeholder
  - 保存 episode sidecar 时，再按原始 instruction pipeline 解析成最终文本

### 3.2 每个 `subtask_def` 的关键字段

每个子任务定义里，我统一使用了这些字段：

- `id`
  - 子任务编号
- `name`
  - 子任务名字
- `instruction_idx`
  - 当前子任务对应的 instruction 索引
- `search_target_keys`
  - 该子任务搜索阶段关注哪些对象
- `action_target_keys`
  - 该子任务动作阶段涉及哪些对象
- `required_carried_keys`
  - 执行动作前，手里必须已经携带哪些对象
- `carry_keys_after_done`
  - 子任务完成后，手里应该保留哪些对象
- `allow_stage2_from_memory`
  - 兼容保留字段；当前这版实现里不再直接用它跳过 `stage 1`
- `done_when`
  - 当前子任务的语义完成条件描述
- `next_subtask_id`
  - 当前子任务结束后切到哪个子任务

这里面真正决定“状态切换”和“子任务切换”的，是下面几个字段：

- `search_target_keys`
  - 决定当前该找谁
- `action_target_keys`
  - 决定当前动作围绕谁
- `carry_keys_after_done`
  - 决定当前 subtask 结束后，下一步是不是“带着物体继续找下一个目标”
- `next_subtask_id`
  - 决定 subtask 完成后跳到哪里
- `allow_stage2_from_memory`
  - 当前仅作为 schema 兼容字段保留，不再决定是否跳过粗搜索

### 3.3 运行时状态字段

为了真正完成状态切换，我在运行时维护了这些字段：

- `current_subtask_idx`
  - 当前属于哪个子任务
- `current_stage`
  - 当前属于 `stage 1 / 2 / 3`
- `current_instruction_idx`
  - 当前子任务 instruction 索引
- `current_focus_object_key`
  - 当前正对或动作聚焦的对象
- `current_search_target_keys`
  - 当前正在找哪些对象
- `current_action_target_keys`
  - 当前动作涉及哪些对象
- `carried_object_keys`
  - 当前抓在手里的对象
- `discovered_objects`
  - 历史上已经发现过的对象
- `visible_objects`
  - 当前帧可见对象
- `subtask_done`
  - 哪些子任务已经完成
- `transition_log`
  - 子任务切换日志
- `saved_frame_annotations`
  - 每一帧的结构化注释

### 3.4 每帧状态字段

每帧真正写入数据集的字段，负责承载训练时可直接使用的状态标签：

- `subtask`
  - 当前帧属于哪个子任务
- `stage`
  - 1=粗搜，2=精定位，3=动作执行
- `subtask_instruction_idx`
  - 当前子任务 instruction 索引
- `focus_object_idx`
  - 当前聚焦目标对象编号
- `focus_object_visible`
  - 当前聚焦目标是否可见
- `info_complete`
  - 当前信息是否已经足够
- `camera_mode`
  - 当前相机处于粗搜、精搜还是动作阶段
- `camera_target_theta`
  - 当前相机旋转目标
- `visible_object_mask`
  - 当前画面里看见了哪些对象
- `discovered_object_mask`
  - 历史上已经发现过哪些对象
- `search_target_mask`
  - 当前搜索目标集合
- `action_target_mask`
  - 当前动作目标集合
- `carried_object_mask`
  - 当前抓着哪些对象
- `target_uv_norm`
  - 当前聚焦目标的归一化图像坐标

## 4. 白名单任务覆盖情况

这次已经把 `task_config/rotate_task_whitelist.yml` 里的 26 个任务全部改完。

包括：

- `beat_block_hammer_rotate_view`
- `blocks_ranking_rgb_rotate_view`
- `blocks_ranking_size_rotate_view`
- `click_alarmclock_rotate_view`
- `click_bell_rotate_view`
- `move_pillbottle_pad_rotate_view`
- `move_stapler_pad_rotate_view`
- `open_laptop_rotate_view`
- `place_a2b_left_rotate_view`
- `place_a2b_right_rotate_view`
- `place_burger_fries_rotate_view`
- `place_cans_plasticbox_rotate_view`
- `place_container_plate_rotate_view`
- `place_empty_cup_rotate_view`
- `place_fan_rotate_view`
- `place_mouse_pad_rotate_view`
- `place_object_scale_rotate_view`
- `place_object_stand_rotate_view`
- `place_shoe_rotate_view`
- `press_stapler_rotate_view`
- `shake_bottle_horizontally_rotate_view`
- `shake_bottle_rotate_view`
- `stack_blocks_three_rotate_view`
- `stack_blocks_two_rotate_view`
- `stamp_seal_rotate_view`
- `turn_switch_rotate_view`

这些任务现在都统一具备：

1. 子任务 plan 定义
2. `search_and_focus_rotate_subtask(...)`
3. `enter_rotate_action_stage(...)`
4. `complete_rotate_subtask(...)`
5. 显式的 `carried_object_keys` 切换

## 5. 我是怎么改任务执行逻辑的

旧逻辑通常是：

1. 先 `_scan_scene_two_views(...)`
2. 然后直接抓取 / 放置 / 按压 / 排列

新逻辑统一改成：

1. `search_and_focus_rotate_subtask(subtask_id, ...)`
2. `enter_rotate_action_stage(subtask_id, ...)`
3. 执行动作
4. 如有抓取，更新 `carried_object_keys`
5. `complete_rotate_subtask(subtask_id, carried_after=...)`
6. 自动进入下一个子任务

例如两段式任务：

- `pick A`
- `place A on B`

现在会被拆成：

1. 子任务 1：找 `A`，抓 `A`
2. 子任务 2：带着 `A` 找 `B`，把 `A` 放到 `B`

例如排序 / 堆叠任务，则会变成更长的链式子任务：

- 找可移动块
- 抓取
- 找锚点块
- 放置
- 再找下一个块
- 再放置

## 6. 我做了哪些验证

### 6.1 静态验证

在 `robotwin` 环境中执行：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
python -m py_compile envs/_base_task.py envs/utils/camera_visibility.py script/collect_data.py test/test_camera_visibility.py <all whitelist rotate env files>
```

结果：通过

### 6.2 单元测试

执行：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
pytest -q test/test_camera_visibility.py
```

结果：`6 passed`

### 6.3 真实采集 smoke 验证

#### 已完整通过的单子任务 smoke

执行：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
CUDA_VISIBLE_DEVICES=0 ROBOTWIN_MAX_SEED_TRIES=10 python script/collect_data.py click_bell_rotate_view demo_clean
```

结果：

- 5 个 episode 全部成功
- 已生成 HDF5、视频、instructions、`subtask_metadata`
- `scene_info.json` 已写入 `subtask_metadata_path`

实际产物目录：

- `data/click_bell_rotate_view/demo_clean__easy_fan150/`

#### 已验证到真实多子任务切换的 smoke

执行：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
CUDA_VISIBLE_DEVICES=0 ROBOTWIN_MAX_SEED_TRIES=10 python script/collect_data.py place_empty_cup_rotate_view demo_clean
```

结果：

- seed 收集成功
- 已保存多个 episode
- 我人工中断了后续 episode，避免继续耗时
- 但 `episode0` 已经足够证明多子任务切换是正确的

在 `episode0` 的 sidecar 里可以看到：

- `subtask 1` 完成时 `carried_after = ['A']`
- 随后自动 `begin_subtask 2`
- `subtask 2` 会先以 `stage 1` 注册进入；只有当当前画面已经看到目标时，后续帧才会立刻切到 `stage 2`
- 最后 `complete_subtask 2`

说明：

1. `next_subtask_id` 生效了
2. `carry_keys_after_done` 生效了
3. 当前版本的 `stage 2` 触发依赖“当前帧可见”，而不是历史记忆
4. 每帧状态写入也生效了

### 6.4 easy 白名单 3-episode 无渲染全量采集

本次最终实际跑的是：

- `task_config/demo_clean_easy3_headless.yml`

这个配置里明确设置了：

- `episode_num: 3`
- `render_freq: 0`

实际批量采集是在 `robotwin` 环境里逐个任务执行：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
export MPLCONFIGDIR=/tmp/mpl_robotwin
export CUDA_VISIBLE_DEVICES=0
export ROBOTWIN_MAX_SEED_TRIES=50
python script/collect_data.py <task_name> demo_clean_easy3_headless
```

最终结果：

1. `task_config/rotate_task_whitelist.yml` 中 26 个 easy 任务全部完成
2. 每个任务都成功采集 3 个 episode
3. 总计完成 78 个真实 episode
4. 所有任务都生成了：
   - `data/episode0-2.hdf5`
   - `subtask_metadata/episode0-2.json`
   - `scene_info.json`
   - `instructions.json`

我额外做了两层全量校验：

1. 文件存在性校验
   - 26 个任务全部满足 `3 个 HDF5 + 3 个 subtask_metadata JSON + scene_info.json`
2. 帧数一致性校验
   - 对全部 78 个 episode 检查 `len(HDF5['subtask']) == len(frame_annotations)`
   - 结果：`78 / 78` 全部通过

我还抽查了四个代表性任务：

- `click_bell_rotate_view`
- `place_burger_fries_rotate_view`
- `stack_blocks_three_rotate_view`
- `turn_switch_rotate_view`

确认以下字段已经真实写入 HDF5：

- `subtask`
- `stage`
- `subtask_instruction_idx`
- `focus_object_idx`
- `focus_object_visible`
- `info_complete`
- `camera_mode`
- `camera_target_theta`
- `visible_object_mask`
- `discovered_object_mask`
- `search_target_mask`
- `action_target_mask`
- `carried_object_mask`
- `target_uv_norm`

同时也确认了：

1. `subtask_metadata/episodeX.json` 内存在 `subtask_defs`、`transition_log`、`frame_annotations`
2. `scene_info.json` 的每个 `episode_N` 条目都包含 `subtask_metadata_path`

真实采集中观察到的主要随机失败模式有两类：

1. 个别任务在 seed 阶段会出现 `pre_grasp_pose=None`
2. 个别任务会因为物体初始不稳定触发 `UnStableError`

这些失败都被 `ROBOTWIN_MAX_SEED_TRIES=50` 的重试机制消化掉了，没有阻止 easy 白名单任务最终完成 3-episode 收集。

## 7. 检查点文件

文件：

- `.progress/rotate_subtask_checkpoint.json`

它记录了：

- 当前项目状态
- 已完成项
- 正在进行项
- 待完成项
- 已迁移任务列表
- 新增字段列表
- 已做验证
- 下次恢复时从哪里继续

这个文件的作用是：

1. 中途停止时快速恢复上下文
2. 后续继续扩展 VLM 数据导出时不需要重新梳理一遍
3. 让“代码改到哪一步了”有明确结构化记录

## 8. 现在已经实现到什么程度

已经完成的部分：

1. 白名单任务全部完成子任务化改造
2. easy 白名单 26 个任务已经全部完成 3-episode 真实无渲染采集
3. 每帧状态标签已经落到 HDF5
4. episode 级 sidecar JSON 已生成
5. 每个 rotate episode 现在会同步导出 `episodeX_annotated.mp4`
6. `scene_info.json` 已按 episode 挂接 `subtask_metadata_path` 和 `annotated_video_path`
7. 单子任务和多子任务都已做真实采集验证
8. 78 个 episode 的 `frame_annotations` 数量已经全量验证与 HDF5 帧数一致
9. 生命周期 bug 已修复

还没有完成的部分：

1. 还没有把多帧 VLM pretrain 样本正式导出成单独数据格式
2. 还没有专门针对 seed 阶段的 `pre_grasp_pose=None` / `UnStableError` 做进一步鲁棒性优化

## 9. 如果后面继续做，建议优先级

建议下一步按这个顺序继续：

1. 基于现有 `frame_annotations + transition_log + subtask_instruction_map` 生成多帧 VLM pretrain 数据
   这里的 `subtask_instruction_map` 已经是按 episode 解析后的最终文本，`subtask_instruction_template_map` 则保留了模板来源
2. 如果后续要继续大规模采集，可以优先加固 seed 阶段的鲁棒性
3. 再考虑是否需要为个别任务补更细粒度的目标点定义

## 10. 一句话总结

这次我做的不是“给任务多加几个标签”，而是把白名单 rotate-view 任务整体改造成了一个统一的、可搜索、可切换、可追踪、可落盘的子任务状态系统，并且已经在真实链路里把 26 个 easy 白名单任务全部采集到了 3 个 headless episode。
