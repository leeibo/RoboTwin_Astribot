这是一个按 `z_new_envs_tips.md` 整理的新任务汇总文档。

## 场景参数配置

- demo_clean_fan_double
- demo_clean
- demo_randomized_fan_double
- demo_randomized
- 测试可以用demo_clean_fan_double_test进行测试

注意如果要调整桌子范围，特别是上层的，要看一看任务里plate等加载位置和调整后的是否适配（默认都是很合适的）

## 1. search_object 系列

### search_object

对应文件：`envs/search_object.py`

任务描述：
- 单层扇形桌面任务。
- 桌上有一个 cabinet，目标物体一开始藏在 drawer 里。
- 机器人先搜索目标物体；如果看不到，就转而搜索 cabinet，拉开 drawer，再重新搜索并抓起目标物体。
- 当前目标物体池包括三类：彩色小方块、`057_toycar`、`073_rubikscube`。

成功判定：
- 目标物体被最终抓起，并且高度相对初始位置提升超过 `SUCCESS_LIFT_Z = 0.03`。
- 最终夹爪需要保持闭合。

补充区域
- 如果发现打开柜子后物品不容易看到，可以调整`OBJECT_OUTER_EDGE_OFFSET`参数，设置物品向外偏移的距离
- 目前开柜子的方法是接近，往后拉动`DRAWER_PULL_TOTAL_DIS = 0.20` `DRAWER_PULL_STEPS = 2`，然后手臂往两边移动防止碰撞。如果要改柜子到机器人距离，可能需要同步调整这两个参数进行适配
- 左右手实际上用的是不一样的调用逻辑，明明设置成完全对称的参数，但是就是做不到对称动作的抓取，总会一边成功率高一边成功率低。因此最后采用了这样分开用逻辑的方法。

### search_object_left

对应文件：`envs/search_object_left.py`

任务描述：
- 继承自 `search_object`。
- 只改 cabinet 的摆放侧和半径：`CABINET_THETA_SIGN_CHOICES = 1.0`，`CABINET_CYL_R = 0.7`。
- 可以理解为把柜子固定在机器人左前方，强调左侧开柜这一支流程。

当前结果：
- `data_show/search_object_left/demo_clean__easy_fan150`：`20 / 44 = 45.45%`


### search_object_right

对应文件：`envs/search_object_right.py`

任务描述：
- 继承自 `search_object`。
- 只改 cabinet 的摆放侧和半径：`CABINET_THETA_SIGN_CHOICES = -1.0`，`CABINET_CYL_R = 0.6`。
- 可以理解为把柜子固定在机器人右前方，强调右侧开柜这一支流程。

当前结果：
- `data_show/search_object_right/demo_clean__easy_fan150`：`20 / 37 = 54.05%`


## 2. put_block_on 系列

### put_block_on

对应文件：`envs/put_block_on.py`

任务描述：
- 双层扇形桌面任务。
- 场景里生成若干给定随机颜色 block 和一个 plate。
- 机器人需要反复执行“搜索 block -> 抓取 block -> 搜索 plate -> 放到 plate 上”的链条，直到全部放完。
- 共享逻辑同时覆盖下层抓取、上层抓取、同层放置和跨层放置。

成功判定：
- 所有 block 都需要进入 plate 的目标区域附近。
- 默认位置阈值为 `SUCCESS_EPS = [0.08, 0.08, 0.08]`。
- 最终左右夹爪都要张开。


补充区域：
- 下面四个子任务仅仅改动了参数，是plate位置和blocks个数，同时为了适配具体任务也稍微调整了其他参数。
- `BLOCK_SIZE_RANGE`,`PLATE_PLACE_SLOT_OFFSETS`如果发现blocks多了在放到plate中容易死掉，大概率是对plate能放置区域的规划算不出来了。可以调整blocks大小和放置预设点尝试。目前默认的是一组算是比较好的参数。
- 上层抓取如果容易碰到上层桌子，可以把`PLATE_PLACE_SLOT_OFFSETS`调整大一点（目前0.025已经比较合适了）
- 改具体任务直接在具体任务里面➕参数就行，不会影响其他任务

### put_block_on_upper_easy

对应文件：`envs/put_block_on_upper_easy.py`

任务描述：
- 继承自 `put_block_on`。
- `PLATE_LAYER = "upper"`。
- `BLOCK_COUNT = 2`，`BLOCK_LAYER_SEQUENCE = ("lower", "lower")`。
- 即：两个 block 都在下层，plate 在上层，属于较简单的“下层抓取 -> 上层放置”设置。

当前结果：
- `data_show/put_block_on_upper_easy/demo_clean_fan_double__medium_fan180`：`15 / 23 = 65.22%`

### put_block_on_lower_easy

对应文件：`envs/put_block_on_lower_easy.py`

任务描述：
- 继承自 `put_block_on`。
- `PLATE_LAYER = "lower"`。
- `BLOCK_COUNT = 2`，`BLOCK_LAYER_SEQUENCE = ("upper", "upper")`。
- 即：两个 block 都在上层，plate 在下层，属于较简单的“上层抓取 -> 下层放置”设置。
- 另外重写了 `PLATE_PLACE_SLOT_OFFSETS`，让两个/三个 block 在 lower plate 上的摆位更规整。

当前结果：
- `data_show/put_block_on_lower_easy/demo_clean_fan_double__medium_fan180`：`15 / 29 = 51.72%`

### put_block_on_upper_hard

对应文件：`envs/put_block_on_upper_hard.py`

任务描述：
- 继承自 `put_block_on`。
- `PLATE_LAYER = "upper"`。
- `BLOCK_COUNT = 3`，`BLOCK_LAYER_SEQUENCE = ("lower", "lower", "upper")`。
- 即：三个 block 里两个在下层、一个在上层，plate 在上层，需要混合处理跨层与同层放置。

当前结果：
- `data_show/put_block_on_upper_hard/demo_clean_fan_double__medium_fan180`：`15 / 45 = 33.33%`

### put_block_on_lower_hard

对应文件：`envs/put_block_on_lower_hard.py`

任务描述：
- 继承自 `put_block_on`。
- `PLATE_LAYER = "lower"`。
- `BLOCK_COUNT = 3`，`BLOCK_LAYER_SEQUENCE = ("upper", "upper", "lower")`。
- 即：三个 block 里两个在上层、一个在下层，plate 在下层。
- 这个变体还单独改了：
  - `PLATE_PLACE_SLOT_OFFSETS`
  - `LOWER_PLACE_DIS = 0.05`
  - `BLOCK_SIZE_RANGE = (0.015, 0.02)`
- 因此它不只是数量更难，盘内摆放参数也更紧。

当前结果：
- `data_show/put_block_on_lower_hard/demo_clean_fan_double__medium_fan180`：`15 / 76 = 19.74%`

补充：
- 下层放置容易找不到合适的位置，所以需要再调整参数

## 3. put_block_target_fan_double 系列

### 共享母逻辑

共享文件：`envs/_put_block_target_fan_double_base.py`

任务描述：
- 这一组任务都可以看成 `put_block_on` 的单 block 目标容器版。
- 场景里只有一个绿色 block 和一个目标容器。
- 机器人执行“搜索 block -> 抓取 block -> 搜索目标容器 -> 把 block 放到容器里/上”的流程。
- 这三个任务都在各自文件里额外加入了目标角度 `±5°` 随机扰动。

共享成功判定：
- block 要靠近目标容器的功能点。
- 同时需要满足接触关系和夹爪张开。


### put_block_plasticbox_fan_double

对应文件：`envs/put_block_plasticbox_fan_double.py`

任务描述：
- 目标容器是 `062_plasticbox`。
- `TARGET_LAYER = "upper"`，默认语义是把 block 放进 plastic box。
- 关键差异参数：
  - `TARGET_PADDING = 0.10`
  - `TARGET_TASK_PREPOSITION = "into"`
  - `SUCCESS_XY_TOL = 0.08`
  - `SUCCESS_Z_TOL = 0.06`

当前结果：
- `data_show/put_block_plasticbox_fan_double/demo_clean_fan_double__medium_fan180`：`15 / 22 = 68.18%`

### put_block_breadbasket_fan_double

对应文件：`envs/put_block_breadbasket_fan_double.py`

任务描述：
- 目标容器是 `076_breadbasket`。
- `TARGET_LAYER = "upper"`，默认语义是把 block 放进 breadbasket。
- 关键差异参数：
  - `TARGET_PADDING = 0.08`
  - `TARGET_TASK_PREPOSITION = "into"`
  - `SUCCESS_XY_TOL = 0.09`
  - `SUCCESS_Z_TOL = 0.08`

当前结果：
- `data_show/put_block_breadbasket_fan_double/demo_clean_fan_double__medium_fan180`：`15 / 22 = 68.18%`

### put_block_skillet_fan_double

对应文件：`envs/put_block_skillet_fan_double.py`

任务描述：
- 目标容器是 `106_skillet`。
- `TARGET_LAYER = "upper"`，默认语义是把 block 放到 skillet 上。
- 关键差异参数：
  - `TARGET_PADDING = 0.06`
  - `TARGET_TASK_PREPOSITION = "on"`
  - `TARGET_LAYER_SPECS[*].qpos` 改成 skillet 朝向
  - `SUCCESS_XY_TOL = 0.06`
  - `SUCCESS_Z_TOL = 0.05`

当前结果：
- `data_show/put_block_skillet_fan_double/demo_clean_fan_double__medium_fan180`：`15 / 24 = 62.50%`

## 4. blocks_ranking 系列

### 共享母逻辑

对应文件：
- `envs/blocks_ranking_rgb_fan_double.py`
- `envs/blocks_ranking_size_fan_double.py`

任务描述：
- 双层扇形桌面上的三 block 排序任务。
- 三个 block 中，`A` 作为左侧锚点，直接放在目标行最左端；`B` 和 `C` 需要被依次搜索、抓取和摆到右侧。
- 目标行固定在下层，沿一条圆弧从左到右排开。

共享成功判定：
- 三个 block 都要落在各自目标位置附近。
- 默认阈值是 `SUCCESS_XY_TOL = 0.09`、`SUCCESS_Z_TOL = 0.08`。
- 最终左右夹爪都要张开。

补充区域：
- 目前颜色，大小是写死的（为了保持anchor，也就是参考block在下层），如果要随机化可以再进行调整
- 这个任务本来成功率挺高的，结果昨天测试又莫名其妙低了？……

### blocks_ranking_rgb_fan_double

对应文件：`envs/blocks_ranking_rgb_fan_double.py`

任务描述：
- 三个 block 的身份固定为红、绿、蓝。
- 任务要求最终从左到右排成：红 -> 绿 -> 蓝。
- 其中红色 block 是左侧锚点，绿色和蓝色需要依次搬运。

当前结果：
- `data_show/blocks_ranking_rgb_fan_double/demo_clean_fan_double__medium_fan180`：`15 / 76 = 19.74%`

### blocks_ranking_size_fan_double

对应文件：`envs/blocks_ranking_size_fan_double.py`

任务描述：
- 三个 block 的身份固定为大、中、小。
- 任务要求最终从左到右排成：大 -> 中 -> 小。
- `BLOCK_DEFS` 里直接给了三档不同的尺寸采样范围。
- 其中大 block 是左侧锚点，中、小两个 block 需要依次搬运。

当前结果：
- `data_show/blocks_ranking_size_fan_double/demo_clean_fan_double__medium_fan180`：`15 / 103 = 14.56%`

## 5. place_object_basket_fan_double

### place_object_basket_fan_double

对应文件：`envs/place_object_basket_fan_double.py`

任务描述：
- 双层扇形桌面任务。
- 当前实现里，目标物体池包括：
  - `081_playingcards`
  - `057_toycar`
  - `071_can`
- 目标容器当前不是普通 basket，而是 `076_breadbasket`；任务名保留了旧名字。
- 默认设置下，目标物体在下层，breadbasket 在上层。
- 任务流程是：搜索物体 -> 抓起物体 -> 搜索上层 breadbasket -> 放入容器。

成功判定：
- 当前实现使用简化判定：物体中心到 breadbasket 中心的距离小于 `SUCCESS_DIST = 0.18` 即算成功。

当前结果：
- 用户提供：`20 / 56 = 35.71%`

补充区域：
- `can`最好抓取，因此成功案例中的比例最大
