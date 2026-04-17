# fix.md 重构执行文档

## 1. 目标

这次不是在当前 VLM VQA 导出链路上继续打补丁，而是按 `fix.md` 的要求，结合历史指令和当前仓库实现，重新梳理一版可执行的重构清单。

目标分为两块：

1. 重构 rotate 场景的 VQA 后处理链路，正确导出：
   - `object_search.json`
   - `angle_delta.json`
   - `memory_compression_vqa.json`
2. 补齐 rotate randomized 场景的扇形桌面随机化能力，并让采集配置、可见性规则、导出逻辑保持一致。

## 2. 需求来源

本次文档只保留和修复有关的有效信息，主要来自：

1. `fix.md`
2. `/home/admin1/.codex/history_bak.jsonl` 中和本次修复直接相关的历史指令，重点包括：
   - `ts=1775896824`
   - `ts=1776161970`
   - `ts=1776234113`
   - `ts=1776248553`
   - `ts=1776410337`

这些历史要求补充了 `fix.md` 里没有写全但必须保留的约束，例如：

1. `stage1` 和 `stage2` 的开始帧、结束帧都应进入记忆。
2. `stage2` 应始终以一个 `Rotate(0, 0)` 结束，再进入 `stage3`。
3. `stage3` 需要按 `action_chunk_size=10` 离散化，并把 action 阶段的观测也纳入记忆。
4. 动作阶段的旋转记忆必须基于规划值，不能被执行抖动污染。
5. 只在确实拿着物体时描述持物状态，而且不要输出成列表。

## 3. 当前项目里相关实现的位置

### 3.1 VQA 导出链路

当前实现主要集中在：

1. `script/export_rotate_vlm_dataset.py`
2. `script/rotate_vlm/__init__.py`
3. `script/rotate_vlm/snapshots.py`
4. `script/rotate_vlm/models.py`
5. `script/rotate_vlm/annotated_video.py`

### 3.2 rotate 运行时和可见性

当前 rotate 场景的发现、对准、帧标注主要集中在：

1. `envs/_base_task.py`
2. `envs/utils/camera_visibility.py`

### 3.3 randomized 场景和杂物采样

当前 randomized 配置和杂物分布逻辑主要在：

1. `task_config/demo_randomized.yml`
2. `envs/_base_task.py`
3. `envs/utils/rand_create_cluttered_actor.py`

## 4. 当前实现与要求的主要偏差

### 4.1 object_search 导出还不是目标格式

当前 `script/rotate_vlm/__init__.py` 里的 `_render_object_search_response()` 仍然会输出：

1. 已发现物体列表
2. 当前可见物体列表
3. cross-side 扫描提示
4. carried 列表

这和最新要求不一致。当前版本在杂乱背景下会把无关信息写进 `think`，而目标要求是只保留和当前目标搜索直接有关的信息。

此外，当前 `stage3` 样本虽然已经有 `action_chunk`，但记忆组织、`frame/info` 对齐方式、以及“动作阶段也进入记忆”的规则都没有完全按历史要求落稳。

### 4.2 angle_delta 当前构造方式不对

当前 `_build_angle_delta_sample()` 是从一个子任务块里取“首尾两个非 stage3 snapshot”做差，并直接输出数值。

这和要求有三处偏差：

1. 应只从同一 `subtask` 的前两个阶段中取帧，不能跨物体、不能跨子任务。
2. 应表达“从历史帧到当前帧累计水平转了多少度”，而不是简单取块首尾。
3. 角度应来自规划记忆链路，而不是受执行抖动影响的即时值。

### 4.3 memory compression 现在只是最小实现

当前 `script/rotate_vlm/snapshots.py` 的压缩逻辑只在 `len(history_slots) >= max_context_frames` 时触发，默认等价于 16 帧触发。

当前版本的问题是：

1. 没有在“子任务切换”时触发压缩。
2. `CompressionEvent` 只记录了一次 `before -> after`，没有为大规模扩增压缩 VQA 保留足够信息。
3. 压缩规则是 bucket/语义重合启发式，不是“空间优先，其次最新优先”的明确实现。
4. 没有把“最优压缩结果可用于构造大量子集压缩 VQA”的需求落下去。
5. 当前 `_build_memory_compression_sample()` 的 prompt 和答案都过于弱，达不到训练要求。

### 4.4 目标发现规则还没有 40% AABB 暴露阈值

当前 `envs/utils/camera_visibility.py` 在 `mode="aabb"` 下，只要 AABB 投影和可视区域相交，就会认为目标在视野内。

这和要求不一致。要求是：

1. AABB mask 在画面中暴露超过 40% 才算发现。
2. AABB 只用于“是否发现”的判定。
3. 对准中心时仍然要用中心点，不应直接拿 AABB 中心替代。

也就是说，现在的“发现”门槛偏低，会把只露出一点点边缘的物体误判成已发现。

### 4.5 randomized 的扇形桌杂物分布没有真正适配 fan

虽然 rotate 任务环境里大多已经默认使用 `fan` 桌面，但当前 randomized 链路仍有明显问题：

1. `task_config/demo_randomized.yml` 没有形成一套明确的 fan randomized 采集配置。
2. `envs/_base_task.py:get_cluttered_table()` 仍然按矩形 `xlim/ylim` 投杂物。
3. `envs/utils/rand_create_cluttered_actor.py` 里也还是矩形边界约束，不是扇形环带约束。
4. 没有把杂物半径下界设成 `r >= 0.55`。
5. 没有体现“新加入的散落物体尽可能靠后”的要求。

因此当前 randomized 数据即使能采，也不是目标分布。

## 5. 我需要做的事情

下面是我后续真正要做的事情，不是泛泛而谈，而是和当前代码一一对应的重构任务。

### 5.1 重建 rotate VQA 的记忆帧组织规则

我要先改 `script/rotate_vlm/snapshots.py`，把“什么帧进入记忆、何时替换、何时压缩”重新定义清楚。

具体包括：

1. `stage1` 和 `stage2` 的开始帧、结束帧都进入记忆。
2. `stage2` 结束时必须显式形成一个 `Rotate(0, 0)` 的结束帧。
3. `stage3` 开始时不复用旧的 `stage2` 结束样本，而是按历史要求重新组织成动作 chunk 采样。
4. `stage3` 每隔 `action_chunk_size=10` 步记录一次新观测，并把这张观测帧放进记忆。
5. `stage3` 最后一个 chunk 不足 10 步时，剩余 action 用静止填充。
6. 当子任务切换时，要处理“新子任务起始帧替换上一个 stage2 结束零转角记忆”的规则。
7. 记忆链路使用规划角度，不使用执行抖动推出来的偏差。

### 5.2 重写 object_search 样本构造

我要重写 `script/rotate_vlm/__init__.py` 里 object search 的样本生成和模板渲染逻辑。

具体包括：

1. `think` 只保留和目标搜索有关的信息，不再列出所有杂物或所有可见/已发现物体。
2. 如果目标在当前帧可见：
   - 直接描述“当前视角中看到目标，位置约在 `(x,y)`”。
3. 如果目标只在历史记忆中可见：
   - 必须选“最新的、仍然有效的目标证据帧”。
   - 描述“当前视角中没看到目标，在第 k 帧看到目标，位置约在 `(x,y)`，按这张记忆帧转 xx 度”。
4. `frame` 字段必须和引用的证据帧一致。
5. `info` 字段必须和 `frame`、`camera` 保持一致。
6. `stage3` 的 `think` 保持 stage2 风格，只补一句“已经位于中心附近，正在执行动作操作”。
7. `stage3` 的 `camera` 固定为 `Rotate(0, 0)`。
8. `stage3` 的 `action` 只写左右手臂未来 10 步 chunk。
9. 若一开始就是 `stage3`，也必须有完整的 `think/info/frame/camera/action` 记录，不能缺。
10. 持物描述只在确实持物时输出，且写成单句，不写成列表。

### 5.3 重写 angle_delta 样本构造

我要重写 angle delta 的样本选择与输出模板。

具体包括：

1. 样本的两帧必须来自同一子任务。
2. 两帧必须属于该子任务的前两个阶段，不跨 `stage3`。
3. 角度差值应表示“从历史帧到当前帧累计的水平旋转量”。
4. 角度计算要基于规划链路，不基于执行抖动。
5. 输出模板要改成 `fix.md` 里要求的 `<think>...</think><answer>...</answer>` 格式。
6. 要确保不再出现为了凑数量而用 fallback 重复截取同一块数据的行为。

### 5.4 重写 memory_compression_vqa 生成逻辑

我要把 memory compression 从“事件导出”改成“压缩结果导出 + 数据扩增导出”。

具体包括：

1. 压缩触发点改为：
   - 子任务切换时
   - 记忆达到 16 帧时
2. 压缩目标明确为：
   - 第一优先级：保留尽可能大的 FOV 覆盖范围
   - 第二优先级：在覆盖不变或近似不变时优先保留新帧
3. 对动作阶段同位置重复帧，优先保留最新一帧。
4. 压缩算法要显式产出“最优保留帧集合”。
5. 在得到最优保留帧后，额外构造从 `max(最优帧数, 4)` 到 `16` 的大量子集压缩样本。
6. 这些扩增样本都应压缩到同一个最优保留帧集合。
7. `CompressionEvent` 或等价数据结构要记录生成这些样本所需的完整信息，而不是只记录一次 before/after。
8. prompt/answer 模板要改成真正可训练的压缩任务格式，而不是简单输出“保留后是哪些帧”。

### 5.5 修正“发现目标”的可见性门槛

我要修改 `envs/utils/camera_visibility.py` 和 `envs/_base_task.py` 中的发现逻辑。

具体包括：

1. 在 AABB 投影后计算当前可见区域占对象 AABB mask 的比例。
2. 只有可见比例超过 40% 才算 `visible/discovered`。
3. 对目标是否“已发现”的更新、`visible_object_keys`、`discovered_object_keys` 都要统一走这个规则。
4. 对准中心的 `yaw_error` 仍然使用中心点投影，不改成 AABB 中心纠偏。
5. 必要时把可见比例写进 frame annotation sidecar，便于后处理调试。

### 5.6 重建 fan randomized 杂物分布

我要重构 rotate randomized 的 fan 场景采样逻辑。

具体包括：

1. 明确 fan randomized 的 task config，必要时恢复或新增 `demo_randomized_easy_ep2.yml`。
2. `demo_randomized` 系列配置要显式对齐扇形桌：
   - `table_shape: fan`
   - `fan_center_on_robot`
   - `fan_inner_radius`
   - `fan_outer_radius`
   - `fan_angle_deg`
   - `fan_center_deg`
3. 杂物采样不能再只靠矩形 `xlim/ylim`，而要按扇形环带采样。
4. 采样半径下界设为 `r >= 0.55`。
5. 在满足碰撞与可放置条件下，优先把散落物体放在更靠后的区域，也就是更大的 `r`。
6. 去掉和“新圆桌”相关的偏差实现，只保留 fan 方案。
7. 让 easy 白名单 rotate 任务都能用这套 randomized 配置重新收集数据。

### 5.7 补测试，避免再次修坏

这次必须把关键约束写成测试，不然导出链路还会再次偏掉。

至少要补下面几类测试：

1. `object_search`：
   - 当前帧直接发现目标
   - 历史帧引用目标
   - `stage3` chunk 样本
   - 一开始直接进入 `stage3`
2. `angle_delta`：
   - 同一子任务、同一对象
   - 不跨子任务
   - 使用规划角度而不是执行抖动
3. `memory_compression_vqa`：
   - 16 帧触发
   - 子任务切换触发
   - 空间优先
   - 新帧替换旧帧
   - 子集扩增正确
4. 可见性：
   - AABB 暴露比例低于 40% 不算发现
   - 高于 40% 才算发现
   - 对准仍走中心点
5. fan randomized：
   - 杂物都落在扇形合法区域
   - `r >= 0.55`
   - 不再依赖矩形边界假设

## 6. 推荐实施顺序

为了避免边改边乱，后续我会按下面顺序推进：

1. 先修可见性判定和 frame annotation 字段。
2. 再重写 `snapshots.py` 的记忆帧组织规则。
3. 然后重写 `object_search`、`angle_delta`、`memory_compression_vqa` 三类导出。
4. 再改 fan randomized 的采样和配置。
5. 最后补测试、重跑 smoke case、再批量重导数据。

这个顺序的原因是：

1. 可见性门槛会直接影响“目标是否发现”和记忆证据帧。
2. 记忆帧组织决定三类 VQA 的输入。
3. randomized 分布调整放在后面，避免先采一批错误数据。

## 7. 重构后需要验证的结果

完成后我需要至少验证下面这些结果：

1. 对 `beat_block_hammer_rotate_view` 和 `blocks_ranking_rgb_rotate_view` 各重跑 1 个 episode 的导出样例。
2. 导出的 `vlm/` 目录下必须稳定产出：
   - `object_search.json`
   - `angle_delta.json`
   - `memory_compression_vqa.json`
3. `object_search` 中 action 阶段样本要正确包含 10 步 action chunk。
4. `angle_delta` 不再跨子任务、不再重复凑数。
5. `memory_compression_vqa` 不只在 16 帧时有样本，子任务切换也能产出。
6. randomized fan 数据中杂物分布满足 `r >= 0.55` 且整体偏后。
7. 白名单 rotate 任务可以按新的 randomized 配置重新收集数据。

## 8. 结论

结合当前项目状态，这次我要做的核心工作不是“修几个 prompt”，而是把以下四条链路一起重构：

1. 记忆帧生成链路
2. 三类 VQA 导出链路
3. 目标发现/可见性判定链路
4. fan randomized 杂物分布链路

只有这四块一起收敛，`fix.md` 里要求的 rotate VLM VQA 导出才会重新变得可用。
