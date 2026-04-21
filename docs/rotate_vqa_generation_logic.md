# Rotate VQA 生成逻辑说明

日期: 2026-04-18

## 1. 文档目的

这份文档描述的是当前仓库里已经实现并实际用于导出数据的 Rotate VQA 逻辑，不是理想化设计稿。目标是把整条链路讲清楚，让别人只看这份文档也能理解:

1. 环境侧每一帧到底记录了什么标注。
2. 这些标注如何被整理成记忆帧和快照。
3. 三类 VQA 样本是如何生成的。
4. 为什么有些任务会没有 memory compression 样本。
5. randomized 白名单任务当前的数据是按什么配置生成的。

当前实现的核心代码位于:

- `script/rotate_vlm/__init__.py`
- `script/rotate_vlm/snapshots.py`
- `script/rotate_vlm/models.py`
- `script/rotate_vlm/annotated_video.py`
- `script/render_object_search_qa_video.py`
- `envs/_base_task.py`
- `envs/utils/camera_visibility.py`

## 2. 整体流程概览

整条链路可以概括为:

1. 任务执行时，环境在每一帧写入 rotate 相关标注，保存到 `subtask_metadata/episode*.json` 的 `frame_annotations` 中。
2. 导出脚本读取 `episode*.hdf5` 里的 head camera 视频帧，以及 `frame_annotations`。
3. 根据标注把 episode 切成一串 `memory slot`。
4. 对每个 slot 构造一个带历史记忆的 `snapshot`。
5. 基于这些 snapshot 和压缩事件，导出三类样本:
   - `object_search`
   - `angle_delta`
   - `memory_compression_vqa`
6. 同时导出:
   - 基础注释视频 `episode*_annotated.mp4`
   - 右侧带 object-search QA 面板的视频 `episode*_annotated_object_search_qa.mp4`

额外说明:

- 环境内部 rotate 的正负号约定与 VQA 输出不同。
- 当前 VQA 文本与 metadata 里统一使用: 向左为负，向右为正。

默认导出参数是:

- 最大上下文帧数 `max_context_frames = 16`
- action chunk 大小 `action_chunk_size = 10`

也就是说，一个 VQA 样本最多输入 16 张图，其中最后一张永远是当前视角。

## 3. 环境侧原始标注是怎么来的

### 3.1 子任务与阶段定义

rotate-view 任务在环境里会先配置一组子任务 `subtask_defs`。每个子任务通常会带这些信息:

- `id`
- `search_target_keys`
- `action_target_keys`
- `required_carried_keys`
- `carry_keys_after_done`

执行时会维护一个运行时状态:

- `subtask`
- `stage`
- `focus_object_key`
- `search_target_keys`
- `action_target_keys`
- `carried_object_keys`
- `camera_target_theta`
- `waist_heading_deg`

当前阶段语义是:

- `stage1`: 搜索阶段。目标还没被锁定，`info_complete = 0`，相机模式为扫描。
- `stage2`: 定位阶段。目标已经可见，或者已经通过历史记忆完成重定位，`info_complete = 1`。
- `stage3`: 动作阶段。已经进入操作执行，`info_complete = 1`。

其中 stage2 的当前实现非常重要:

1. 只要目标当前可见，或者历史里已经有该目标的 `last_world_point`，系统就会进入 stage2。
2. 一旦进入 stage2，VQA 侧就直接把它视为“信息充分”。
3. 如果目标不在当前帧、但在历史帧中出现过，think 会写成“在第 k 帧 (x, y) 发现了目标”，而不是“当前视角没看到”。

### 3.2 每帧保存的关键字段

每次存帧时，环境会把当前状态写入 `frame_annotations`。当前 VQA 逻辑真正会用到的字段主要有:

- `frame_idx`: episode 内绝对帧号。
- `subtask`
- `stage`
- `focus_object_key`
- `search_target_keys`
- `action_target_keys`
- `carried_object_keys`
- `visible_object_keys`
- `discovered_object_keys`
- `visible_object_uv_map`
- `discovered_last_uv_map`
- `visible_object_ratio_map`
- `target_uv_norm`
- `camera_mode`
- `waist_heading_deg`
- `camera_target_theta`

这些字段里最关键的是:

1. `visible_object_uv_map`
   表示当前帧里真正可见的物体，以及它们在图像中的归一化坐标。

2. `discovered_last_uv_map`
   表示历史上已经发现过的物体，最近一次看到它时的图像坐标。

3. `waist_heading_deg`
   当前观察方向的水平朝向，后面会用于角度差与记忆压缩。

4. `camera_target_theta`
   当前帧对应的规划目标观察角。对于搜索/定位阶段，它代表系统此时希望转向的角度。

### 3.3 “可见”是如何定义的

当前不是简单用物体中心点是否落在视锥里来定义可见，而是:

1. 先把物体的 world AABB 投影到图像平面。
2. 计算投影框与有效视野区域的重叠比例。
3. 当重叠比例不小于 `rotate_scan_aabb_visible_ratio_threshold` 时，才记为可见。

当前 randomized 配置里该阈值是:

- `rotate_scan_aabb_visible_ratio_threshold = 0.4`

所以当前“可见”的意思可以理解成:

- 目标 AABB 至少有 40% 面积暴露在有效视野区域里。

## 4. 从 frame annotation 到 memory slot

### 4.1 先对帧做分段

导出时不会直接逐帧做 VQA，而是先把 `frame_annotations` 切成一段一段的 segment。分段键是:

- `(subtask_id, stage, camera_target_theta)`

但 `stage3` 会特殊处理:

- 所有 `stage3` 帧都视为同一类 segment，不再按 `camera_target_theta` 细分。

这意味着:

1. 搜索/定位阶段里，只要子任务、阶段、目标观察角不变，就会被视为同一段。
2. 动作阶段里，一整段连续动作帧会单独拿出来再切 chunk。

### 4.2 memory slot 的构造规则

当前 memory slot 规则如下:

1. 对于 `stage1` 和 `stage2`:
   - 每个 segment 取首帧作为 `stageX_start`
   - 如果尾帧和首帧不同，再取尾帧作为 `stageX_end`

2. 对于 `stage3`:
   - 不再取首尾帧
   - 直接按 `action_chunk_size = 10` 切块
   - 每一块变成一个 `stage3_chunk` slot
   - 这个 slot 绑定的是该 chunk 的第一帧，也就是时刻 `t` 的观测帧

所以当前进入记忆的不是所有原始帧，而是:

- 搜索/定位阶段的起止关键帧
- 动作阶段按 10 步切出来的 chunk 代表帧

### 4.3 stage3 为什么要按 chunk 存

因为 stage3 的目标已经不是继续搜索，而是执行动作。当前实现里:

1. `stage3_chunk` 的 `planned_delta_deg` 直接置 0。
2. 同时会把这个 chunk 真实对应的左右臂动作序列拼出来。
3. object-search 样本如果落在 stage3，会在 `<action>` 字段里写出真实 10 步 action chunk，而不是占位字符串。
4. `stage3` 的 `current_frame_idx` 和当前输入图像都对齐到这个 chunk 的第一帧。

## 5. snapshot 是怎么构造的

每个当前 slot 都会生成一个 `EpisodeSnapshot`。它代表“当前时刻模型能看到什么历史记忆，以及最后应该回答什么”。

### 5.1 上下文窗口

当前窗口规则是:

1. 历史保留上限是 `max_context_frames - 1 = 15` 个 history slot。
2. 当前 slot 会追加为最后一张图。
3. 所以最终 prompt 最多 16 张图。

### 5.2 构造当前 prompt 时，也会先做一次压缩

即使全局历史还没有触发正式 memory compression，当前 snapshot 的 prompt 也不是直接把最近 15 个 slot 全喂进去，而是:

1. 取最近最多 15 个 history slot。
2. 把它们和当前 slot 放在一起。
3. 先跑一次 `compress_memory_slots(...)`。
4. 再把当前 slot 从结果里拿掉，剩下的作为 prompt history。
5. 最后把当前 slot 作为最后一张图接回去。

因此，最终送给模型的 prompt 本身已经是“压缩过的历史 + 当前帧”。

但 `stage3` 是一个例外。当前实现里:

1. action 阶段每 10 帧会生成一个新的 `stage3_chunk` memory slot。
2. 为了让 action VQA 真正表达“根据当前观测和历史记忆预测未来 10 步动作”，`stage3` 的 prompt 不再对最近历史做这一步压缩。
3. 换句话说，`stage3` 会直接使用最近的原始历史 slots，所以前面的 action chunks 会继续留在 prompt 里。

因此现在 action 阶段看到的是:

- 当前 chunk 的起始观测帧
- 最近若干个 stage1/stage2 关键帧
- 最近若干个已经发生过的 action chunk 记忆帧

### 5.3 evidence frame 怎么选

当前 object-search 回答需要知道“哪一张图提供了目标证据”。证据帧选择规则是:

1. 先看当前 slot 本身的目标是否可见。
2. 如果当前不可见，就倒序扫描历史 prompt slots。
3. 优先找同一个 `target_key` 的可见 UV。
4. 如果没有，再找当前目标集合与历史目标集合有交集的 candidate key。
5. 只要找到合法 UV，就把该 slot 视为 evidence slot。

这里有一个非常重要的索引约定:

- object-search 回答里的 `<frame>` 字段写的是 prompt 内第几张图，采用 1-based 编号。
- 它不是 episode 的绝对帧号。
- episode 绝对帧号会单独记录在 metadata 里的 `evidence_frame_idx`。

## 6. 当前帧压缩规则

这是当前实现里最核心的一段逻辑。

### 6.1 压缩触发时机

历史压缩事件在两种情况下触发:

1. 切换子任务时，先把旧子任务累积下来的 history 做一次压缩。
2. history slot 数达到 16 时，做一次压缩。

压缩事件会记录成 `CompressionEvent`，里面包含:

- `before_slots`
- `after_slots`
- `trigger`
- `trigger_frame_idx`

### 6.2 规则 1: 合并连续 Rotate(0,0) 区间

当前的“零旋转帧”定义是:

- `stage3_chunk`
- 或者 `abs(planned_delta_deg) <= 1e-3`

先从老到新扫描整个序列，把连续的零旋转区间折叠成一帧，只保留这一段里最新的那一帧。

这条规则对应用户要求里的:

- 连续不改变视角的内容不需要保留多张，保留最后到达的那一帧即可。

### 6.3 规则 2: 从老到新增量加入，新帧进来后反删旧帧

经过规则 1 之后，再按时间从老到新把帧依次加入队列 `kept`。

每次新帧加入后:

1. 遍历队列里更老的帧。
2. 如果某一旧帧的视野覆盖已经完全被“其余保留帧的并集”覆盖，就把这张旧帧删掉。
3. 最新刚加入的当前帧不会在这一轮被删掉。

这正对应用户最后确认的定义:

- 老到新进队
- 每来一张新帧，就检查队列中其它旧帧有没有已经冗余
- 如果冗余，删旧帧，保新帧

### 6.4 当前“空间覆盖”的实现方式

这里的“空间优先”当前不是基于图像语义重叠，也不是基于三维点云覆盖，而是一个更稳定的水平视野近似:

1. 用 `waist_heading_deg` 表示一张记忆帧的朝向。
2. 假设每张记忆帧覆盖一个固定的水平 FOV 区间。
3. 当前参数是:
   - half FOV = `35 deg`
   - coverage grid step = `2 deg`
4. 对每张帧，把 `heading +/- 35 deg` 覆盖到一个离散角度网格上。
5. 如果某一帧覆盖到的网格点被其它保留帧完全覆盖，就视为冗余。

所以当前压缩逻辑的本质是:

- 先去掉连续不转的重复观察
- 再用“水平视野覆盖并集”去做冗余删除
- 在能保持覆盖范围不变的前提下，优先保留更新的帧

## 7. 三类 VQA 的生成逻辑

## 7.1 object_search

### 输入

每个 snapshot 都会生成一个 object-search 样本。输入图像顺序是:

- 历史 prompt slots 从早到晚
- 最后一张是当前视角

### 用户问题

用户问题的模板大意是:

- 给定当前子任务 instruction
- 输入图像按时间排序
- 最后一张是当前视角
- 请输出 `<think><info><frame><camera><action>`

### 输出字段含义

- `<think>`: 自然语言推理。
- `<info>`:
  - `1` 表示当前信息足够
  - `0` 表示还需要继续搜索
- `<frame>`:
  - 写 prompt 内第几张图提供了证据
  - 1-based，相对索引
- `<camera>`:
  - 需要执行的水平旋转，格式是 `Rotate(x, 0)`
- `<action>`:
  - 只有 stage3 时会写真实 action chunk
  - 其它阶段为空字符串

### “信息是否足够”的当前判定

当前判定函数是:

1. 只要 `stage >= 2`，直接认为信息足够。
2. 否则，只有当 prompt 里能找到 evidence frame 且 UV 合法时，才认为信息足够。

所以用户之前强调的那条规则，当前实现是满足的:

- 只要已经进入 stage2，就认为信息足够。

### think 文本的当前行为

1. 如果当前帧里直接看到目标:
   - think 写“当前视角中发现了目标物体，位置约在 (x, y)”

2. 如果当前帧没看到，但历史 prompt 里某张图看到过:
   - think 写“在第 k 帧 (x, y) 中发现了目标物体”
   - 同时明确“当前阶段信息已足够”

3. 如果已经是 stage2/3，但当前样本没有具体 evidence UV:
   - think 会写“目标物体已在历史搜索阶段完成定位，当前根据已有定位信息继续执行”

4. 如果还处在信息不足的搜索阶段:
   - think 会写“当前视角中还没看到目标，需要继续搜索”

### camera 字段怎么来

当前 `<camera>` 字段逻辑是:

1. 如果是 `stage3_chunk`，固定输出 `Rotate(0, 0)`。
2. 如果证据来自历史帧，输出“evidence 帧朝向”和“当前帧朝向”的差。
3. 否则输出当前 slot 的 `planned_delta_deg`。

也就是说:

- 当前帧可见目标时，camera 基本代表下一步应该继续转多少。
- 目标只存在于历史记忆里时，camera 更像是“为了回到证据视角，应转回多少”。

### stage3 的 action 字段

如果当前 slot 是 `stage3_chunk`:

1. 会从 `joint_action/left_arm` 和 `joint_action/right_arm` 中取出对应帧的动作。
2. 左右臂动作拼成一个二维数组。
3. 若不足 10 步，用最后一步复制补齐。
4. 最终写进样本顶层的 `action` 字段，并同步写进回答里的 `<action>`。

## 7.2 angle_delta

angle-delta 只在搜索/定位阶段生成，不覆盖 stage3。

### 样本配对规则

1. 先按 `subtask_id` 分组。
2. 去掉同一绝对帧号的重复 slot。
3. 对每个子任务，固定取第一张 slot 作为 anchor。
4. 从第二张开始，构造:
   - `anchor_slot`
   - `current_slot`
   - 从 anchor 到 current 的累计水平转角

这里的累计角度不是直接看两帧朝向做差，而是把中间每一步的 `planned_delta_deg` 累加起来。

### 样本内容

- 输入: 两张图，第一张是历史帧，第二张是当前帧。
- 输出:
  - `<think>` 解释累计转角
  - `<answer>` 给出方向和角度

当前 metadata 还会保存:

- `frame_indices`
- `angle_delta_deg`
- `planned_actions`

## 7.3 memory_compression_vqa

### 基础样本来源

只要某次历史压缩触发了 `CompressionEvent`，就会尝试从这次事件里构造 compression VQA。

### 一个事件里先求“最优保留帧”

对 `event.before_slots` 重新跑一次当前压缩规则，得到:

- `optimal_slots = compress_memory_slots(before_slots)`

这组 `optimal_slots` 就是这次事件下的“最优压缩结果”。

### 不只做 16 -> 最优，还会做大量子集扩增

当前实现不是只导出“满 16 帧压缩成最优”的样本，而是会基于同一次压缩事件扩增出很多变体:

1. 样本输入长度从 `max(len(optimal_slots), 4)` 到 `min(len(before_slots), 16)`。
2. 对每个长度，保留全部最优帧。
3. 然后再从被丢弃的帧里补若干张，构造不同输入子集。
4. 对这个输入子集重新压缩。
5. 如果重新压缩后确实变短，才保留这个样本。

有一个例外:

- 如果压缩事件来自 `subtask_switch`，并且切换前只有 2 到 3 个 memory slots，
  当前会直接把这次事件本身记录成 1 条 VQA，不再受“至少 4 张图”限制，也不做子集扩增。

也就是说，当前 memory compression 数据集是在做:

- “不同长度的输入记忆 -> 相同压缩规则下的更短保留集合”

### 变体是怎么采样的

当前会优先构造几类变体:

- `oldest`
- `newest`
- `spread`
- 如果组合数不大，直接枚举全部组合
- 如果组合数太大，随机采样，单个长度最多保留 64 个变体

### 输出字段含义

memory compression 的回答也复用 `<think><info><frame><camera><action>` 格式，但这里:

- `<info>` 固定为 `1`
- `<frame>` 写的是“输入序列中应该保留第几张图”
- 也是 1-based 的相对位置，不是 episode 绝对帧号
- `<camera>` 和 `<action>` 为空

think 会明确说明:

1. 连续 `Rotate(0,0)` 先折叠。
2. 再按时间从早到晚加入新帧。
3. 新帧加入后删除被并集覆盖的旧帧。
4. 最终保留哪些输入位置，移除哪些位置。

## 8. 注释视频是怎么生成的

## 8.1 基础 annotated 视频

`episode*_annotated.mp4` 来自环境存储的 head-camera 视频，再叠加每帧 annotation 文本。

当前顶部条里主要会显示:

- `frame/subtask/stage`
- `focus/search/action`
- `visible/discovered/carried`

如果某帧额外挂载了 object-search QA 字段，也可以显示 QA 相关文本，但常规基础视频主要用于调试底层 frame annotation。

### 8.2 object-search QA 视频

`script/render_object_search_qa_video.py` 会读取:

- `video/episode*_annotated.mp4`
- `vlm/object_search.json`

然后把每条 object-search 样本按 `sample_frame_idx` 映射回视频时间轴，在视频右侧加一个 panel。

当前右侧 panel 保留的字段是:

- `Video Frame`
- `QA Step`
- `Q Images`
- `Q`
- `Q think`
- `info`
- `frame`
- `camera`

不会再额外显示:

- `A`
- `QA`
- `action`

如果右侧内容放不下，当前实现不是把字强行挤进原分辨率，而是:

1. 保留左边原视频内容。
2. 在右侧新开一块 panel。
3. 按内容需要自动拉高整张画布高度。

### 8.3 视频编码兼容性

当前导出时会优先调用 `ffmpeg`，统一转成:

- codec: `h264`
- pixel format: `yuv420p`
- `+faststart`

这是为了解决部分播放器无法直接播放 `mp4v` 中间产物的问题。

## 9. randomized 白名单数据当前是怎么生成的

本次实际使用的配置是:

- 白名单: `task_config/rotate_task_whitelist.yml`
- 任务配置: `task_config/demo_randomized_easy_ep2.yml`
- 存储目录后缀: `demo_randomized_easy_ep2__easy_fan150`

### 9.1 randomized 配置的关键点

当前 randomized easy 配置里，与 rotate VQA 最相关的参数是:

- `episode_num: 2`
- `table_shape: fan`
- `fan_outer_radius: 0.9`
- `fan_inner_radius: 0.3`
- `fan_angle_deg: 150`
- `rotate_scan_aabb_visible_ratio_threshold: 0.4`
- `rotate_clutter_min_radius: 0.55`
- `rotate_cluttered_numbers: 2`
- `clean_background_rate: 0.2`
- `random_background: true`
- `cluttered_table: true`
- `random_light: true`

### 9.2 杂物采样的当前实现

当前扇形杂物采样并不是全桌面均匀撒点，而是:

1. 只在扇形工作区内采样。
2. 要满足最小半径约束，也就是离机器人不能太近。
3. 用多个候选点做挑选，并对更靠后的区域有偏置。

同时还加了一条健壮性修复:

- 对 objaverse 杂物，只会从本地确实存在 `model.urdf` 的资产中采样，避免把缺失资产抽进流程导致采集失败。

## 10. 为什么有些任务的 memory_compression_vqa 是 0

这不是导出遗漏，主要有两种原因:

1. 这个 episode 太短，history 根本没有触发压缩事件。
   例如:
   - 没切子任务
   - 也没累计到 16 个 memory slot

2. 虽然触发了压缩事件，但压缩前后长度没有真的变短。
   当前代码会把这种“没有压缩收益”的候选样本过滤掉。

所以 `memory_compression_vqa = 0` 的含义是:

- 在当前 episode 和当前压缩规则下，没有形成有效的“长输入 -> 更短输出”训练样本。

## 11. 当前导出的数据产物

每个任务目录当前应包含:

- `data/episode*.hdf5`
- `subtask_metadata/episode*.json`
- `video/episode*_annotated.mp4`
- `video/episode*_annotated_object_search_qa.mp4`
- `vlm/object_search.json`
- `vlm/angle_delta.json`
- `vlm/memory_compression_vqa.json`
- `vlm/manifest.json`

截至本次整理时，白名单 randomized 数据的总体统计是:

- 白名单任务数: 26
- episode 数: 52
- annotated 视频数: 52
- object-search QA 视频数: 52
- `object_search` 样本数: 854
- `angle_delta` 样本数: 199
- `memory_compression_vqa` 样本数: 11685

对应的汇总文件有:

- `data/collection_reports/collect_rotate_tasks_whitelist__demo_randomized_easy_ep2__final_summary.json`
- `data/collection_reports/export_rotate_vlm_whitelist__demo_randomized_easy_ep2__summary.json`

## 12. 一句话总结当前实现

如果要把当前 Rotate VQA 的实现用一句话讲给别人听，可以这么说:

> 系统先把每个 episode 的关键观察时刻抽成记忆帧，再用“连续零旋转折叠 + 水平视野覆盖冗余删除”的规则压缩记忆；随后围绕“怎么继续找目标”“累计转了多少角度”“哪些记忆帧可以删”这三个问题，分别导出 object-search、angle-delta 和 memory-compression VQA，并把 object-search 的问答过程同步渲染到注释视频右侧。

## 13. 当前实现中的几个注意事项

最后补几个容易在沟通时说错的点:

1. object-search 的 `<frame>` 是 prompt 内相对位置，不是 episode 绝对帧号。
2. memory compression 的 `<frame>` 也是输入序列内相对位置。
3. 当前“空间覆盖”是基于 heading 的水平 FOV 近似，不是图像特征级覆盖。
4. stage2 在当前实现里一律视为信息充分。
5. stage3 的 `<action>` 已经是真实 10-step action chunk，不是占位文本。
6. 代码里保留了 `allow_stage2_from_memory` 配置位，但当前执行逻辑里并没有额外加一个分支去限制它；实际行为是只要历史发现信息足够，就会尝试走历史重定位进入 stage2。
