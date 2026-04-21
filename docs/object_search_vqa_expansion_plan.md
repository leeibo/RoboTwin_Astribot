# Object Search VQA 扩充方案

日期: 2026-04-21

## 1. 目标

这份文档只讨论 `object_search` 的扩充，不讨论 `angle_delta` 和 `memory_compression_vqa`。

目标是基于当前已经实现的 rotate-view 标注与导出链路，在不改环境采集格式的前提下，把 `stage1 / stage2` 的 `object_search` 样本数量和难度都扩起来，同时尽量保持标签仍然由现有规则自动生成，而不是人工写新规则。

核心要求是:

1. 扩充后的样本仍然走当前 `object_search` 的 teacher 逻辑。
2. 当前帧仍然固定为 prompt 最后一张图。
3. 扩充重点放在“历史记忆子集变化”上，而不是像素级图像增强。
4. 扩充出来的样本要有明确的训练价值，不能只是重复拷贝。

## 2. 当前生成链路

当前链路可以概括为:

1. 环境把每帧 rotate 标注写进 `subtask_metadata/episode*.json`。
2. 导出时先把 `frame_annotations` 切成 `memory slot`。
3. `stage1 / stage2` 每个 segment 只保留:
   - `stageX_start`
   - `stageX_end`
4. 对每个 slot 只构造一个 `EpisodeSnapshot`。
5. 每个 `EpisodeSnapshot` 只导出一条 `object_search` 样本。

对应代码位置:

- `memory slot` 构造: [script/rotate_vlm/snapshots.py](/home/admin1/Desktop/RoboTwin/script/rotate_vlm/snapshots.py)
- `snapshot` 构造: [script/rotate_vlm/snapshots.py](/home/admin1/Desktop/RoboTwin/script/rotate_vlm/snapshots.py)
- `object_search` 样本导出: [script/rotate_vlm/__init__.py](/home/admin1/Desktop/RoboTwin/script/rotate_vlm/__init__.py)

## 3. 当前链路的瓶颈

当前 `object_search` 样本偏少，主要不是因为 episode 少，而是因为同一个“当前决策时刻”只导出一条 prompt。

具体瓶颈有三类:

1. `stage1 / stage2` 的 slot 太稀疏。
   每个 segment 只取首尾帧，中间许多旋转视角没有进入记忆。

2. 同一个当前 slot 只保留一个 prompt。
   当前代码会先把历史压缩成一个固定的 `prompt_history`，然后直接导出，不再对历史子集做变化。

3. 没有显式构造 hard negative。
   例如一个当前帧本来可以依赖历史证据完成定位，但如果把关键证据帧拿掉，当前链路不会额外生成“信息不足”的反事实样本。

这和 `memory_compression_vqa` 的差别很明显:

1. `memory_compression_vqa` 会围绕同一个事件生成多种输入子集。
2. `object_search` 目前没有“围绕同一个决策点做 prompt 扩增”的机制。

## 4. 扩充原则

为了让扩充样本仍然可靠，我建议遵守下面几个原则:

1. 不手写新标签。
   只改变输入 prompt 的历史图集合，标签仍然交给现有 `_make_snapshot`、`_search_info_complete`、`_render_object_search_response` 去重算。

2. 当前帧永远固定。
   不改变当前 slot，只变历史记忆。

3. 保持时间顺序。
   历史图仍然按时间从早到晚排序。

4. 不做图像像素增强。
   当前任务的 supervision 依赖 evidence frame、UV、frame index 和 camera action；像素级增强容易把语义和标签拉开。

5. 扩增要有语义目的。
   不是简单随机删帧，而是围绕“证据保留/证据删除/记忆干扰/最小支持集”来构造样本。

## 5. 总体思路

核心思路是:

把当前 `memory_compression_vqa` 的“围绕同一事件做输入子集扩增”的模式，迁移到 `object_search`，但扩充对象从 `before_slots` 改成“当前 slot 在导出前可用的历史记忆池”。

也就是说，对于同一个当前 slot:

1. 当前帧固定不动。
2. 从可用历史记忆池里构造多个有序子集。
3. 对每个子集重新生成 `snapshot`。
4. 用现有 `object_search` 逻辑重新算:
   - `info`
   - `frame`
   - `camera`
   - `think`
5. 只保留有信息量的变体。

这条线本质上是在做:

- “同一当前决策帧 + 不同历史记忆上下文 -> 不同 `object_search` supervision”

## 6. 推荐的扩充单元

建议把“一个当前 slot”作为最小扩充单元。

对于每个 `stage1 / stage2` 当前 slot，构造:

1. `base sample`
   当前已有样本。

2. `expanded samples`
   使用不同历史子集重算出来的新样本。

这样做的好处是:

1. 现有主流程不需要推翻。
2. 扩充逻辑可以局部插到 `object_search` 导出阶段。
3. 一个当前时刻的所有变体天然可以按组管理。

## 7. 历史池怎么取

这里建议不要只用已经导出的 `snapshot.prompt_slots`，因为它已经是“压缩后的唯一结果”，可扩展空间太小。

更合适的是使用“当前 slot 在生成 prompt 之前的候选历史池”。

对当前 `stage1 / stage2` 链路来说，这个池子应当来自:

- `history_slots[-history_limit:]`

也就是当前 slot 被拼进 prompt 之前、已经经过全局历史压缩但尚未做“当前 prompt 内再压缩”的历史集合。

然后对这个候选池做两步:

1. 采样不同子集。
2. 对每个子集再执行一次当前已有的 prompt 压缩逻辑。

这样生成出来的 prompt 更接近当前系统真实会见到的记忆形式，而不是离线随意拼图。

## 8. 样本扩充类型

我建议先做四类扩充，已经足够覆盖大部分训练收益。

### 8.1 保证证据存在的正样本

目的:

- 扩充“历史里有证据，当前应该输出 `info=1`”的样本。

做法:

1. 找出 baseline snapshot 的 evidence slot。
2. 在历史池里固定保留 evidence slot。
3. 其余历史帧按不同策略补若干张:
   - `oldest`
   - `newest`
   - `spread`
   - `random`
4. 重新生成 snapshot。

预期效果:

1. 同一个当前帧，可以看到“短记忆正样本”和“长记忆正样本”。
2. 模型不会把“必须看满所有历史图”误学成先验。

### 8.2 证据删除的 hard negative

目的:

- 扩充“删掉关键历史证据后，应该退化成 `info=0`”的样本。

做法:

1. 只对 baseline 中 evidence 来自历史的样本做。
2. 删掉 latest evidence slot。
3. 再删掉与它提供同等证据的冗余后继帧。
4. 重新生成 snapshot。
5. 如果重算后:
   - `info` 从 `1` 变成 `0`
   - 或 `frame` 为空
   就保留该样本。

预期效果:

1. 这类样本最有可能提升 stage1 搜索质量。
2. 它能明确告诉模型: “不是所有历史都等价，关键证据被拿掉后就不够了。”

### 8.3 最小支持集正样本

目的:

- 让模型学会“只靠最小必要记忆也能回答”。

做法:

1. 以 baseline 的正样本为起点。
2. 尽可能删掉不影响:
   - `info`
   - `frame`
   - `camera`
   的历史帧。
3. 保留最小集合。

这类样本和 8.1 的区别是:

- 8.1 更偏“多种正样本上下文”
- 8.3 更偏“最小可解释上下文”

### 8.4 干扰增强正样本

目的:

- 让模型在更长、更杂的历史记忆里仍然抓住关键证据。

做法:

1. 从 baseline 正样本出发。
2. 保留当前使用的关键证据。
3. 额外加入若干无关历史帧，尤其是:
   - 更早的搜索帧
   - 方向差异大的旋转帧
   - 不包含目标证据但视觉上相似的帧
4. 重算 snapshot 后若标签不变，则保留。

预期效果:

1. 减少模型对“历史很干净”的依赖。
2. 更贴近真实长记忆推理场景。

## 9. Stage1 / Stage2 的不同处理

虽然我们只做 `object_search`，但 `stage1` 和 `stage2` 的扩充策略不应该完全一样。

### 9.1 Stage1

优先级最高。

原因:

1. `stage1` 本来就处在“搜索是否完成”的临界区。
2. `info=0` 和 `info=1` 都可能出现。
3. 最适合做正负样本对照。

建议:

1. 优先做 8.1 和 8.2。
2. 每个当前 slot 保留更多 hard negative 配额。

### 9.2 Stage2

需要更克制。

原因:

1. 当前逻辑里 `stage >= 2` 直接判 `info=1`。
2. 如果盲目扩增，容易生成很多“标签都一样、文本也差不多”的重复样本。

建议:

1. 只保留:
   - 最小支持集正样本
   - 干扰增强正样本
2. 不主打 hard negative。
3. 更强调 history frame 的变化，而不是标签变化。

## 10. 样本筛选与去重

如果不做筛选，`object_search` 扩增很容易爆炸，并产生大量重复样本。

建议的筛选规则如下。

### 10.1 硬性去重键

按以下键去重:

- `current_frame_idx`
- `prompt_frame_indices`
- `info`
- `frame`
- `camera`

只要这些完全一致，就视为重复样本。

### 10.2 保留条件

只保留以下三类变体:

1. 标签发生变化。
   例如:
   - `info 1 -> 0`
   - `frame [k] -> []`
   - `camera` 变化

2. 标签不变，但 prompt 历史长度显著变化。
   例如:
   - 15 图正样本
   - 3 图最小支持集正样本

3. 标签不变，但历史干扰显著增加。
   例如:
   - 加入多个无关旋转帧后仍保持正确 evidence。

### 10.3 每个当前 slot 的上限

建议先限制为:

- `stage1`: 最多 8 到 12 条扩展样本
- `stage2`: 最多 4 到 6 条扩展样本

先做小规模、稳定的扩增，再看数量分布。

## 11. 推荐增加的 metadata

为了后续分析样本来源，建议给扩增样本加几项 metadata:

- `variant_group`
  - `baseline`
  - `history_positive`
  - `evidence_drop_negative`
  - `minimal_support_positive`
  - `distractor_positive`

- `variant_name`
  例如:
  - `oldest`
  - `newest`
  - `spread`
  - `random_003`
  - `drop_latest_evidence`

- `base_prompt_frame_indices`
  baseline prompt 的 frame indices

- `history_pool_frame_indices`
  扩增时可用历史池的 frame indices

- `expanded_from_current_frame_idx`
  标识这些变体属于哪个当前 slot

这样后面无论是做 viewer 分析，还是做训练 ablation，都会方便很多。

## 12. 建议的实现方式

为了不破坏现有主链路，我建议按下面方式实现。

### 12.1 先补一个中间表示

在 `EpisodeSnapshot` 或导出阶段，保留“当前 slot 的候选历史池”。

最小改动方案是:

1. 在 `build_episode_context(...)` 里，给每个 snapshot 额外保存:
   - `candidate_history_slots`
2. 这个字段只用于扩增，不影响现有 baseline 导出。

### 12.2 新增一个 object_search 扩增函数

新增类似这样的 helper:

```python
def _expand_object_search_variants(
    snapshot: EpisodeSnapshot,
    candidate_history_slots: list[MemorySlot],
    metadata: dict[str, Any],
    max_variants: int,
) -> list[EpisodeSnapshot]:
    ...
```

职责是:

1. 围绕同一个当前 slot 采样历史子集。
2. 重新跑 prompt 压缩。
3. 重新生成 snapshot。
4. 做筛选和去重。

### 12.3 在 object_search 导出阶段并入

在 baseline `object_search` 样本之外:

1. 先导 baseline。
2. 再导 expansion variants。
3. 用统一的 sample builder 输出。

这样 viewer、JSON 格式、下游训练接口都不需要变。

## 13. 推荐的落地顺序

建议分三步。

### 第一步

只做 `stage1` 的两类扩增:

1. `history_positive`
2. `evidence_drop_negative`

这是收益最大、也最容易验证正确性的部分。

### 第二步

加入:

1. `minimal_support_positive`
2. `distractor_positive`

先看样本分布和标签稳定性。

### 第三步

再决定是否进一步放宽:

1. stage2 的变体数量
2. 历史池长度
3. 随机采样上限

## 14. 不建议现在做的事

我不建议第一版就做下面这些。

1. 不建议直接对原始 frame 逐帧扩增。
   当前 slot 太稀疏的问题确实存在，但第一版先在现有 slot 上做 prompt 扩增，风险更低。

2. 不建议做图像裁剪、颜色扰动之类的视觉增强。
   这会让 evidence frame 的语义和当前 deterministic teacher 之间出现偏差。

3. 不建议同时改 `angle_delta`。
   会把评估边界搞乱。

4. 不建议一上来就无上限地枚举所有子集。
   很快就会把数据量打爆，而且重复率高。

## 15. 总结

如果只聚焦 `object_search`，最合理的扩充方向不是改标签规则，而是:

1. 固定当前帧。
2. 围绕“历史记忆子集”生成多种 prompt。
3. 用现有 teacher 逻辑重算 `info / frame / camera / think`。
4. 优先做:
   - 保证证据存在的正样本
   - 删除证据后的 hard negative
   - 最小支持集正样本
   - 带干扰的长记忆正样本

这条路和当前 `memory_compression_vqa` 的扩增思路是一致的，但监督目标更接近“给定不同记忆上下文，下一步该怎么搜、信息够不够、证据在哪一帧”。
