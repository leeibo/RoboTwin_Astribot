# Angle Delta VQA 模板设计

日期: 2026-04-19

## 1. 目标

这份文档只讨论 `angle_delta` 类 VQA 的模板设计。

这次模板按你刚刚明确的定义重写：

1. 提问固定为 2 张图。
2. 当前推荐模板默认不显式带 task instruction；如果后续为了统一格式显式带 instruction，Q 只能使用总任务指令。
3. 输出格式固定为 `<think>...</think><camera>...</camera>`。
4. 这里的 rotation 含义不是“中间过程累计规划角”，而是两张图片视角的 `rotate` 直接作差，也就是 `rotation difference`。
5. `angle_delta` 的 assistant `<think>` 不引入 `The current task is ...` 或 `Now executing subtask ...` 这类 instruction 句，因此不参与本次 task/subtask 双句式改动。

## 2. 任务语义

`angle_delta` 的语义现在明确为：

1. 输入两张图。
2. 第一张是历史帧。
3. 第二张是当前帧。
4. 两张图按时间从早到晚排序。
5. 回答的是两张图片视角的 `rotate` 差值。

也就是说：

```text
delta_rotation = current_view_rotate - history_view_rotate
```

如果当前系统里 `rotate` 用二元形式表示，那么这里输出的也是二元差值：

```text
(dx, dy)
```

最终的控制字段统一写成：

```text
Rotate(dx, dy)
```

## 3. 统一用户问题模板

提问模板固定为：

```text
<image><image>The first image is a history frame and the second image is the current frame. The two images are ordered from earlier to later. Please estimate the rotation difference from the history frame to the current frame. Your response should be in the format of: <think>...</think><camera>...</camera>.
```

这里不再引入其它可变槽位。

额外约束：

1. 当前推荐版本仍然不显式带 instruction。
2. 如果后续为了统一其它 VQA 样本而显式加入 `Your task is: ...`，这个 instruction 也只能使用总任务指令，不能使用子任务指令。
3. assistant `<think>` 仍然只描述两张图之间的视角差值，不追加 task/subtask 句子。

## 4. 统一输出格式

所有 `angle_delta` 样本统一使用：

```xml
<think>...</think><camera>...</camera>
```

字段语义：

1. `<think>`
   只写一条非常短的结构化说明。
2. `<camera>`
   直接输出 `Rotate(dx, dy)`。

## 5. 统一回答模板

回答模板固定为：

```xml
<think>Frames: 2 total (1 history + current). From frame 1 to frame 2, the rotation difference is ({dx}, {dy}).</think><camera>Rotate({dx}, {dy})</camera>
```

说明：

1. `frame 1` 固定表示历史图。
2. `frame 2` 固定表示当前图。
3. `({dx}, {dy})` 就是两个图片视角的 `rotate` 直接作差。
4. `<camera>` 与 `<think>` 中的数值必须完全一致。

## 6. 示例

### 6.1 左转

```xml
<think>Frames: 2 total (1 history + current). From frame 1 to frame 2, the rotation difference is (30, 0).</think><camera>Rotate(30, 0)</camera>
```

### 6.2 右转

```xml
<think>Frames: 2 total (1 history + current). From frame 1 to frame 2, the rotation difference is (-20, 0).</think><camera>Rotate(-20, 0)</camera>
```

### 6.3 同时存在两个轴的变化

```xml
<think>Frames: 2 total (1 history + current). From frame 1 to frame 2, the rotation difference is (15, -10).</think><camera>Rotate(15, -10)</camera>
```

### 6.4 不转

```xml
<think>Frames: 2 total (1 history + current). From frame 1 to frame 2, the rotation difference is (0, 0).</think><camera>Rotate(0, 0)</camera>
```

## 7. 不应该出现的写法

下面这些写法现在都不应该再出现：

1. 如果在 user prompt 里显式加入 task instruction，却使用子任务指令。

2. 输出 `<answer>` 字段，例如：

```xml
<think>...</think><answer>...</answer>
```

3. 在 `<think>` 里写成长段解释，例如：

```text
These two images belong to the same subtask, so we should compare the whole motion process carefully...
```

4. 把角度差理解成“从 anchor 到 current 的中间规划角累计和”。

5. 只输出单轴 `Rotate(dx, 0)`，但忽略第二轴差值。

6. `<think>` 里的数值和 `<camera>` 里的数值不一致。

## 8. 我之前的理解偏差

我之前把 `angle_delta` 理解成：

1. 同一子任务里选一张 anchor 图和一张 current 图。
2. 回答“从 anchor 到 current，中间累计执行了多少水平旋转”。
3. 实现上按中间每一步的规划角去累加。

这个理解来自当前仓库里已有的代码和文档：

1. [script/rotate_vlm/__init__.py:403](/home/admin1/Desktop/RoboTwin/script/rotate_vlm/__init__.py#L403)
   现在的 `_collect_angle_delta_pairs()` 会固定第一张 slot 作为 anchor。
2. [script/rotate_vlm/__init__.py:417](/home/admin1/Desktop/RoboTwin/script/rotate_vlm/__init__.py#L417)
   这里会把 `previous_slot.planned_delta_deg` 逐步累加到 `cumulative_deg`。
3. [docs/fix_rebuild_plan.md:84](/home/admin1/Desktop/RoboTwin/docs/fix_rebuild_plan.md#L84)
   之前的文字也写成了“从历史帧到当前帧累计水平转了多少度”。
4. [docs/rotate_vqa_generation_logic.md:393](/home/admin1/Desktop/RoboTwin/docs/rotate_vqa_generation_logic.md#L393)
   这里同样写的是把中间每一步 `planned_delta_deg` 累加起来。

所以我之前的理解是：

```text
angle_delta = sum(intermediate planned_delta_deg)
```

而不是你现在明确指定的：

```text
angle_delta = current_view_rotate - history_view_rotate
```

这就是我之前理解错的地方。

## 9. 下一步实现约束

如果后面按这份模板去改真实导出逻辑，需要同时改两件事：

1. 模板格式：
   - user prompt 改成固定两图版本
   - assistant 输出改成 `<think><camera>`
2. 数值语义：
   - 不再累计中间 `planned_delta`
   - 直接用两张图各自的 `rotate` 做差

否则只改模板、不改数值来源，语义还是错的。
