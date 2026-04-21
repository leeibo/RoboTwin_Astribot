# Memory Compression VQA 模板设计

日期: 2026-04-19

## 1. 目标

这份文档只讨论 `memory_compression_vqa` / 帧压缩类 VQA 的模板设计。

目标是把帧压缩任务写成真正可训练的格式，而不是只机械输出“保留哪些帧”。模板需要同时满足：

1. 风格和 `object_search`、`angle_delta` 保持一致。
2. 能表达当前压缩算法的核心规则。
3. 能显式说明为什么保留这些帧、为什么删除那些帧。
4. 能把“空间优先、最新优先、保护当前有效任务证据”写清楚。

## 2. 当前压缩算法的语义

当前帧压缩的核心语义不是“任意删帧”，而是：

1. 先折叠连续的 `Rotate(0, 0)` 区间。
2. 每个连续零转区间只保留最后一帧。
3. 然后按时间从早到晚加入剩余帧。
4. 新帧加入后，如果某个更老的帧已经能被其它保留帧的空间覆盖并集替代，就删除那个旧帧。
5. 在覆盖不变时，优先保留更新的帧。

也就是说，压缩目标是：

1. 第一优先级：保留尽可能大的有效视野覆盖。
2. 第二优先级：在覆盖不变时尽可能保留更新的帧。
3. 第三优先级：不能丢掉当前任务仍然需要的最新有效证据。

## 3. 统一输出格式

所有 `memory_compression_vqa` 样本统一使用：

```xml
<think>...</think><info>...</info><frame>...</frame><camera></camera><action></action>
```

字段语义：

1. `<think>`
   写压缩决策过程。
2. `<info>`
   固定写 `1`。
3. `<frame>`
   写最终应该保留的输入图片位置，使用 1-based 相对索引，例如 `[1, 15]`。
4. `<camera>`
   为空。
5. `<action>`
   为空。

## 4. 统一用户问题模板

推荐的 user prompt 模板是：

```text
{image_tokens}Your task is: "{task_instruction}" Please track only the useful memory for {object_phrase}. The input images are ordered from earliest to latest, and the last image is the current view. Please keep the most relevant and reliable frames, and output the filtered frames. Your response should be in the format of: <think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>.
```

其中：

1. `{image_tokens}`
   是若干个连续的 `<image>`，数量等于输入图片数。
2. `{task_instruction}`
   是总任务指令。
3. `{object_phrase}`
   是当前压缩任务需要跟踪的关键对象描述。

推荐写法：

1. 单目标：

```text
Track only the useful memory for the faucet.
```

2. 多目标：

```text
Track only the useful memory for the faucet and the hanging cloth.
```

## 5. 指令字段的定义

这里把 `Q` 和 `A` 中的 instruction 语义明确拆开：

1. user prompt 里的 `Your task is: "{task_instruction}"` 使用总任务指令。
2. assistant answer 的 `<think>` 里同时显式写总任务指令和当前子任务指令。
3. 如果当前压缩样本没有显式子任务，就令 `{subtask_instruction} = {task_instruction}`。

推荐写法固定为：

```text
The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}".
```

## 6. 推荐的 think 骨架

推荐所有 assistant answer 都复用下面的骨架：

```text
Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". This sequence mixes {sequence_summary}. Spatially, keep frames {spatial_keep}. Replacement: {replacement_summary}. Protection: {protection_summary}. Latest valid task evidence comes from frames {evidence_frames}. Keep frames {final_keep}. Info is sufficient now. The latest valid observations still cover {object_phrase}.
```

这个骨架对应当前压缩任务里最关键的 8 个判断点：

1. 输入一共有多少帧。
2. 总任务是什么，以及当前执行的子任务是什么。
3. 这段序列整体是“探索”“回访”“动作静止”还是混合。
4. 从空间覆盖角度先应该保留哪些帧。
5. 哪些旧帧被新帧替代。
6. 哪些帧受到保护不能删。
7. 当前任务的最新有效证据来自哪些帧。
8. 最终应该保留哪些帧。

## 7. 各个插槽的推荐写法

### 7.1 帧数描述

只有两种写法：

```text
Frames: current only.
```

```text
Frames: {n} total ({n-1} history + current).
```

对于压缩任务，正常情况下都应使用第二种。

### 7.2 指令字段

推荐固定写成：

```text
The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}".
```

约束：

1. 这里的 `{task_instruction}` 使用总任务指令。
2. 这里的 `{subtask_instruction}` 使用当前样本对应的子任务指令。
3. 如果没有显式子任务，就令 `{subtask_instruction} = {task_instruction}`。

### 7.3 序列概述

推荐只保留 4 类写法：

```text
This sequence is mostly exploration.
```

```text
This sequence is mostly revisits.
```

```text
This sequence is mostly stable observation.
```

```text
This sequence mixes exploration and revisits.
```

如果包含明显的动作末段静止，也可以写：

```text
This sequence mixes exploration, revisits, and stable observation.
```

### 7.4 空间保留结论

这一句只说“从空间覆盖角度先保留谁”，不讨论保护规则。

推荐格式：

```text
Spatially, keep frames {spatial_keep} for distinct coverage.
```

例如：

```text
Spatially, keep frames [1, 15] for distinct coverage.
```

### 7.5 替代说明

这一句专门解释“新帧替换旧帧”的原因。

推荐格式：

```text
Replacement: frame {newer} over frame {older} because the newer frame revisits the same area with newer valid memory.
```

如果有多组替代：

```text
Replacement: frame {a2} over frame {a1}; frame {b2} over frame {b1}; frame {c2} over frame {c1}.
```

如果没有替代：

```text
Replacement: none.
```

### 7.6 保护说明

这一句专门解释哪些帧不能删。

保护的来源可以是：

1. 当前帧。
2. 最新有效任务证据帧。
3. 当前 still-visible 的关键对象。

推荐格式：

```text
Protection: none.
```

或者：

```text
Protection: keep frame {k} as the latest valid task evidence.
```

或者：

```text
Protection: keep frames {protected_frames} as the latest valid task evidence.
```

### 7.7 最新有效证据说明

这一句明确当前任务真正依赖哪几张图。

推荐格式：

```text
Latest valid task evidence comes from frames {evidence_frames}.
```

例如：

```text
Latest valid task evidence comes from frames [15].
```

### 7.8 最终保留结论

这一句是最终答案前的显式决策。

推荐格式：

```text
Keep frames {final_keep}.
```

## 8. 主模板

下面给出 4 个核心模板，基本覆盖实际常见情况。

### 8.1 标准模板：有探索、有回访、无保护

```xml
<think>Frames: {n} total ({n-1} history + current). The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". This sequence mixes exploration and revisits. Spatially, keep frames {spatial_keep} for distinct coverage. Replacement: {replacement_summary}. Protection: none. Latest valid task evidence comes from frames {evidence_frames}. Keep frames {final_keep}. Info is sufficient now. The latest valid observations still cover {object_phrase}.</think><info>1</info><frame>{final_keep}</frame><camera></camera><action></action>
```

适用条件：

1. 这段序列既有新区域探索，也有重复回访。
2. 不存在必须额外保护的特殊帧。

### 8.2 有保护帧的模板

```xml
<think>Frames: {n} total ({n-1} history + current). The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". This sequence mixes exploration and revisits. Spatially, keep frames {spatial_keep} for distinct coverage. Replacement: {replacement_summary}. Protection: keep frames {protected_frames} as the latest valid task evidence. Latest valid task evidence comes from frames {evidence_frames}. Keep frames {final_keep}. Info is sufficient now. The latest valid observations still cover {object_phrase}.</think><info>1</info><frame>{final_keep}</frame><camera></camera><action></action>
```

适用条件：

1. 某些帧从纯空间角度可删。
2. 但它们是当前任务的最新有效证据，因此必须保留。

### 8.3 主要是稳定观测/零转回访的模板

```xml
<think>Frames: {n} total ({n-1} history + current). The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". This sequence is mostly stable observation. Spatially, keep frames {spatial_keep} for distinct coverage. Replacement: {replacement_summary}. Protection: {protection_summary}. Latest valid task evidence comes from frames {evidence_frames}. Keep frames {final_keep}. Info is sufficient now. The latest valid observations still cover {object_phrase}.</think><info>1</info><frame>{final_keep}</frame><camera></camera><action></action>
```

适用条件：

1. 输入里存在连续零转区间。
2. 这些区间被压成每段只保留最后一帧。

### 8.4 多目标保护模板

```xml
<think>Frames: {n} total ({n-1} history + current). The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". This sequence mixes exploration and revisits. Spatially, keep frames {spatial_keep} for distinct coverage. Replacement: {replacement_summary}. Protection: keep frames {protected_frames} for the latest valid observations of {object_phrase}. Latest valid task evidence comes from frames {evidence_frames}. Keep frames {final_keep}. Info is sufficient now. The latest valid observations still cover {object_phrase}.</think><info>1</info><frame>{final_keep}</frame><camera></camera><action></action>
```

适用条件：

1. 关键对象不止一个。
2. 最终保留帧不仅要覆盖空间，还要覆盖多个目标的最新有效证据。

## 9. 一个完整示例

参考你给的目标风格，推荐完整示例如下：

```xml
<think>Frames: 15 total (14 history + current). The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". This sequence mixes exploration and revisits. Spatially, keep frames [1, 15] for distinct coverage. Replacement: frame 6 over frame 5; frame 15 over frame 14 because the newer frames revisit the same area with newer valid memory. Protection: none. Latest valid task evidence comes from frames [15]. Keep frames [1, 15]. Info is sufficient now. The latest valid observations still cover the faucet and the hanging cloth.</think><info>1</info><frame>[1, 15]</frame><camera></camera><action></action>
```

## 10. 不应该出现的写法

下面这些写法不建议再出现：

1. 只写“保留帧 [1, 15]”，不解释为什么。

2. 在 user prompt 里把 `Your task is: ...` 填成压缩子任务指令，而不是总任务指令。

3. 在 `<think>` 里不能只写一个 instruction；必须同时写 `The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}".`

4. 不区分“空间覆盖”和“任务证据保护”。

5. 不说明哪些旧帧是被新帧替代掉的。

6. 在压缩任务里输出非空的 `<camera>` 或 `<action>`。

7. `think` 里说要保留一组帧，但 `<frame>` 写了另一组。

8. 使用过于空泛的句子，例如：

```text
These frames are useful and the others are redundant.
```

## 11. 推荐落地策略

如果后面按这份模板去改真实生成器，我建议把实现拆成 9 个稳定插槽：

1. `frame_summary`
2. `task_instruction`
3. `subtask_instruction`
4. `sequence_summary`
5. `spatial_keep`
6. `replacement_summary`
7. `protection_summary`
8. `evidence_frames`
9. `final_keep`

生成逻辑上则对应：

1. 先确定 user prompt 里的总任务指令。
2. 再确定 answer `think` 里的当前子任务指令。
3. 再根据压缩过程提取“零转折叠”和“新帧替代旧帧”信息。
4. 再根据当前任务对象，补出 `Protection` 和 `Latest valid task evidence`。
5. 最后把最终保留帧写入 `<frame>`。

这样模板既能贴近当前算法，也能保留你想要的训练监督信息。
