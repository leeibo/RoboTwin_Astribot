# Search VQA 模板设计

日期: 2026-04-19

## 1. 目标

这份文档只讨论 `object_search` / `search` 类 VQA 的模板设计，不讨论 `angle_delta` 和 `memory_compression_vqa`。

目标是先把模板语言规范定清楚，保证后续实现时：

1. `stage1 / stage2 / stage3` 都有明确模板。
2. 当前帧证据、历史帧证据、无显式证据三类情况都被覆盖。
3. `current only / multi-frame memory / carry / action chunk` 这些可选信息都有统一写法。
4. 文本风格尽量靠近你给的 `action` 风格：短句、英文主导、结构稳定、字段明确。

## 2. 统一输出格式

所有 `search` 样本统一使用以下输出壳：

```xml
<think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>
```

字段语义固定如下：

1. `<think>`
   只写结构化自然语言推理，不写多余解释。
2. `<info>`
   - `0`: 当前信息还不足，需要继续搜索。
   - `1`: 当前信息已经足够，可以继续对准或执行动作。
3. `<frame>`
   - 写成 `[]` 或 `[k]`。
   - `k` 是当前 prompt 内的第 `k` 张图，不是 episode 原始帧号。
   - `[]` 表示当前没有引用明确证据图。
4. `<camera>`
   - 固定写成 `Rotate(dx, 0)`。
   - `stage3` 固定是 `Rotate(0, 0)`。
5. `<action>`
   - `stage1 / stage2` 为空。
   - `stage3` 写未来一个 action chunk。
   - 内部既可以是 raw action chunk，也可以是离散 action token 序列；外层模板不变。

## 3. 统一用户问题模板

当前建议所有 `search` 样本都用同一个 user prompt：

```text
{image_tokens}Your task is: "{task_instruction}" The input images are ordered from earliest to latest, and the last image is the current view. Please think about the next action and output it. Your response should be in the format of: <think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>.
```

其中：

1. `{image_tokens}`
   由若干个连续的 `<image>` 构成，数量等于输入图片数。
2. `{task_instruction}`
   总任务指令。

## 4. 推荐的 think 骨架

推荐所有 assistant answer 都尽量复用同一个骨架：

```text
Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. {evidence_clause} {info_clause}{carry_clause}{stage3_clause} Next: Rotate({delta}, 0).
```

其中每个插槽的建议写法如下。

### 4.1 记忆描述

只有两种写法：

```text
Frames: current only.
```

```text
Frames: {n} total ({n-1} history + current).
```

说明：

1. `current only` 表示当前 prompt 里只有当前图，没有历史图。
2. `{n} total ({n-1} history + current)` 表示总共有 `n` 张图，其中 `n-1` 张历史图，1 张当前图。

### 4.2 指令字段描述

这里把 `Q` 和 `A` 中的 instruction 语义明确拆开：

1. user prompt 里的 `Your task is: "{task_instruction}"` 使用总任务指令。
2. assistant answer 的 `<think>` 里同时显式写总任务指令和当前子任务指令。
3. 如果当前样本没有显式子任务，就令 `{subtask_instruction} = {task_instruction}`。

推荐写法固定为：

```text
The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}".
```

### 4.3 证据句

推荐只保留三种核心写法：

1. 当前帧可见：

```text
The {object} is visible in the current view at ({x}, {y}).
```

2. 历史帧可见：

```text
The {object} was found in frame {k} at ({x}, {y}).
```

3. 没有明确证据：

```text
The {object} is not visible in the current memory.
```

重要约束：

1. 如果目标不在当前画面、但在历史帧中出现过，并且当前样本已经进入 `stage2` 或 `stage3`，必须使用

```text
The {object} was found in frame {k} at ({x}, {y}).
```

不能再写 “The {object} is not visible in the current view.” 这种句子作为主句。

### 4.4 信息充分性句

只保留两类：

```text
Info incomplete.
```

```text
Info sufficient.
```

### 4.5 持物句

只有确实持物时才出现：

```text
The {carried_object} is currently held by the {left_or_right} hand.
```

否则整句省略。

### 4.6 stage3 追加句

只在动作阶段出现：

```text
The robot is now executing the task.
```

额外约束：

1. 即使进入 `stage3`，也必须保留

```text
The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}".
```

2. 这句话的位置固定放在 `Frames: ...` 后面。

## 5. 各类情况的模板

下面按“当前阶段 + 证据来源”穷举核心模板。为了避免重复，默认：

1. `{frame_summary}` 已经按第 4.1 节替换。
2. `{task_instruction}` 和 `{subtask_instruction}` 已经按第 4.2 节替换。
3. `{carry_clause}` 默认可选，有持物时再拼接。
4. `{action_chunk}` 只有 `stage3` 才填，其他阶段留空。

### 5.1 Stage 1: 当前帧不可见，历史中也没有有效证据

这是最标准的搜索态模板。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} is not visible in the current memory. Info incomplete. Next: Rotate({delta}, 0).</think><info>0</info><frame>[]</frame><camera>Rotate({delta}, 0)</camera><action></action>
```

适用条件：

1. 当前 `stage1`。
2. 当前图没看到目标。
3. 历史图里也没有可用目标证据。

### 5.2 Stage 1: 当前帧直接看到目标

这是 `stage1` 的“当前图已经足够”的模板。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} is visible in the current view at ({x}, {y}). Info sufficient. Next: Rotate({delta}, 0).</think><info>1</info><frame>[{current_index}]</frame><camera>Rotate({delta}, 0)</camera><action></action>
```

适用条件：

1. 当前仍记为 `stage1`。
2. 但当前视角已经直接看到了目标。
3. 当前样本希望把这张图作为对准依据。

### 5.3 Stage 1: 当前帧没看到，但历史帧里已有目标证据

这是 `stage1` 的边界情况模板。虽然在理想流程里通常会很快进入 `stage2`，但模板上应覆盖。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} was found in frame {k} at ({x}, {y}). Info sufficient. Next: Rotate({delta}, 0).</think><info>1</info><frame>[{k}]</frame><camera>Rotate({delta}, 0)</camera><action></action>
```

适用条件：

1. 当前还是 `stage1`。
2. 当前图没看到目标。
3. 但历史图中已经存在有效目标证据。

### 5.4 Stage 2: 当前帧直接看到目标

这是最标准的定位态模板。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} is visible in the current view at ({x}, {y}). Info sufficient. Next: Rotate({delta}, 0).</think><info>1</info><frame>[{current_index}]</frame><camera>Rotate({delta}, 0)</camera><action></action>
```

### 5.5 Stage 2: 目标不在当前图，但在历史帧里出现过

这是你之前特别强调的模板。这里不能再写“当前视角没看到，所以信息不足”。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} was found in frame {k} at ({x}, {y}). Info sufficient. Next: Rotate({delta}, 0).</think><info>1</info><frame>[{k}]</frame><camera>Rotate({delta}, 0)</camera><action></action>
```

额外规则：

1. `{k}` 应该是“最新的、仍然有效的目标证据帧”。
2. `frame` 必须和 think 中引用的历史帧一致。

### 5.6 Stage 2: 没有显式 UV 证据，但已经有抽象定位信息

这是兜底模板，属于少见但必须覆盖的情况。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} has already been localized earlier. Info sufficient. Next: Rotate({delta}, 0).</think><info>1</info><frame>[]</frame><camera>Rotate({delta}, 0)</camera><action></action>
```

适用条件：

1. 当前已经进入 `stage2`。
2. 系统判定信息充分。
3. 但当前样本里没有明确的 `(x, y)` 证据可引用。

### 5.7 Stage 3: 当前帧直接看到目标，并预测未来 action chunk

动作阶段的核心模板。这里的当前图必须是 action chunk 的第一张图，也就是时刻 `t` 的观测。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} is visible in the current view at ({x}, {y}). Info sufficient. The robot is now executing the task. Next: Rotate(0, 0).</think><info>1</info><frame>[{current_index}]</frame><camera>Rotate(0, 0)</camera><action>{action_chunk}</action>
```

额外规则：

1. 这里的 `{action_chunk}` 代表未来 `[t, t+H)` 的动作。
2. `frame` 引用的是当前 chunk 的首图在 prompt 中的位置。
3. `The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}".` 必须保留，并且紧跟在 `Frames: ...` 后面。

### 5.8 Stage 3: 当前帧没看到目标，但历史帧里有证据

动作阶段如果依赖历史记忆，也必须明确引用那张历史图。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} was found in frame {k} at ({x}, {y}). Info sufficient. The robot is now executing the task. Next: Rotate(0, 0).</think><info>1</info><frame>[{k}]</frame><camera>Rotate(0, 0)</camera><action>{action_chunk}</action>
```

### 5.9 Stage 3: 没有显式 UV 证据，但动作阶段信息已经足够

这是动作阶段的兜底模板。

```xml
<think>Frames: {frame_summary}. The current task is "{task_instruction}". Now executing subtask "{subtask_instruction}". The target object is the {object}. The {object} has already been localized earlier. Info sufficient. The robot is now executing the task. Next: Rotate(0, 0).</think><info>1</info><frame>[{current_index}]</frame><camera>Rotate(0, 0)</camera><action>{action_chunk}</action>
```

说明：

1. `stage3` 一定是信息充分。
2. `camera` 一定是 `Rotate(0, 0)`。
3. `action` 一定非空。
4. 如果任务一开始就直接进入 `stage3`，也直接用这组三个模板之一。

## 6. 可选修饰槽

下面这些不是新的主模板，而是可以拼接到主模板里的可选句。

### 6.1 有持物时

把下面这句插到 `Info ...` 之后、`Next ...` 之前：

```text
The {carried_object} is currently held by the {left_or_right} hand.
```

例如：

```text
... Info sufficient. The white mug is currently held by the right hand. Next: Rotate(0, 0).
```

### 6.2 多图输入时

把 `Frames: current only.` 换成：

```text
Frames: 5 total (4 history + current).
```

### 6.3 action 的两种序列写法

如果后续继续保留 raw action：

```xml
<action>[[a11,a12,...],[a21,a22,...],...]</action>
```

如果后续改为离散 token：

```xml
<action><robot_action_913><robot_action_280>...</action>
```

这两种都可以，但推荐统一只保留一种，避免训练分布混杂。

## 7. 不应该出现的写法

下面这些属于模板禁忌，后续实现时应避免：

1. 在 `stage2 / stage3` 且目标来自历史帧时，写成

```text
The {object} is not visible in the current view, so more search is needed.
```

2. 在 `stage3` 里继续输出非零的 `camera`。

3. 在 `stage3` 里用 chunk 的最后一张图作为当前图。

4. 在 `stage3` 里把 `<action>` 留空。

5. `think` 引用了历史第 `k` 张图，但 `<frame>` 写成了别的值。

6. `stage1 / stage2` 的 `<action>` 非空。

7. `frame` 混用 prompt 相对位置和 episode 绝对帧号。

## 8. 推荐落地策略

如果后续要把这些模板真正接到代码里，我建议按下面方式实现：

1. 先固定公共骨架：
   - `frame_summary`
   - `task_instruction`
   - `subtask_instruction`
   - `object`
2. 再单独决定 `evidence_clause`：
   - current
   - history
   - none
3. 再按 `stage` 决定：
   - `info`
   - `camera`
   - `stage3_clause`
   - `action`
4. 最后再拼接可选修饰：
   - `carry_clause`

这样模板系统只需要处理少量稳定插槽，不需要为每个 case 写一大堆硬编码分支。
