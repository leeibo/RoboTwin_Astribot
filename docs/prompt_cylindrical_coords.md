# 🧠 Role（角色设定）
资深具身开发专家、算法工程师，熟悉 SAPIEN 仿真、坐标系变换与 RoboTwin 代码库

---

# 🎯 Task（核心任务）
在 RoboTwin 项目中实现**柱坐标系（圆柱坐标系）**与**世界坐标系**之间的双向变换，并封装便于在柱坐标系下放置物体的工具函数。当前桌面已为扇形，但物体描述仍使用绝对世界坐标，不利于在扇形工作空间内以「距离 + 角度」的方式描述和放置物体。

---

# 📋 Requirements（功能要求）

## 1. 核心变换函数

实现 `world_to_robot` 和 `robot_to_world`（互为逆函数）：

- **world_to_robot(world_pt, robot_root_xy, robot_yaw_rad=0)**  
  - 将世界坐标系下的点或位姿转换为柱坐标系（机器人视角）表示  
  - 支持输入：三维点 `(x, y, z)` 或七维位姿 `(x, y, z, qw, qx, qy, qz)`

- **robot_to_world(robot_pt, robot_root_xy, robot_yaw_rad=0)**  
  - 将柱坐标系下的点或位姿转换为世界坐标系表示  
  - 支持输入：三维柱坐标 `(r, theta, z)` 或七维柱坐标系位姿

## 2. 柱坐标系定义

- **轴心**：以机器人 root 的 xy 为柱坐标系的轴中心点 `(cx, cy)`
- **z 轴**：与世界 z 轴重合，`world z=0` 与柱坐标 `z=0` 对齐
- **角度 theta**：绕 z 轴右手定则，机器人**左转**为正（逆时针俯视为正）
- **零度方向**：`theta=0` 表示机器人初始朝向（正对桌子/工作区方向）

柱坐标表示：`(r, theta, z)`  
- `r`：到轴心的径向距离  
- `theta`：绕 z 轴的角度（弧度）  
- `z`：高度，与世界 z 一致

## 3. 七维位姿的柱坐标形式

定义柱坐标系下的七维位姿：`(r, theta, z, qw, qx, qy, qz)`

- **位置**：`(r, theta, z)` 如上
- **姿态**：四元数 `(qw, qx, qy, qz)` 在**柱坐标系局部基**下表示：
  - 局部 x 轴：径向向外（r 增大方向）
  - 局部 y 轴：切向（theta 增大方向，即左转方向）
  - 局部 z 轴：与世界 z 轴一致

需要实现世界七维位姿与该柱坐标七维位姿之间的双向转换。

## 4. 封装工具函数

提供便于用柱坐标系放置物体的函数，例如：

- `place_pose_cyl(r, theta, z, qw, qx, qy, qz, robot_root_xy, robot_yaw_rad=0)` → 世界坐标七维位姿
- `place_point_cyl(r, theta, z, robot_root_xy, robot_yaw_rad=0)` → 世界坐标三维点
- `rand_pose_cyl(rlim, theta_lim, zlim, robot_root_xy, robot_yaw_rad=0, ...)` → 柱坐标范围内的随机位姿（世界坐标）
- 与现有 `create_actor`、`rand_pose`、`rand_create_*` 等接口可组合使用

---

# 🚫 Constraints（限制条件）

- **不**修改 `create_actor`、`create_fan_table` 等已有函数的签名和默认行为
- **必须**与项目现有约定一致：
  - 七维位姿格式：`[x, y, z, qw, qx, qy, qz]`（SAPIEN `Pose` 的 `p + q`）
  - 使用 `transforms3d`、`sapien.Pose`、`numpy` 等既有依赖
- **必须**支持 `robot_root_xy` 和 `robot_yaw_rad` 作为参数，便于在不同机器人配置/桌面对齐方式下复用
- 对于 `r=0` 等退化情况，需做健壮处理（如 theta 未定义时采用默认值或给出明确约定）

---

# 📦 Output Format（输出格式）

1. **完整实现**：  
   - 新建模块文件，如 `envs/utils/cylindrical_coords.py`  
   - 包含 `world_to_robot`、`robot_to_world`、`place_pose_cyl`、`place_point_cyl`、`rand_pose_cyl` 等函数  
   - 在 `envs/utils/__init__.py` 中导出新函数（若适用）

2. **简要说明**：  
   - 在模块顶部或 docstring 中说明柱坐标系约定（轴心、零度方向、左转为正）  
   - 对 `r=0`、theta 范围（如 `[-π, π]` 或 `[0, 2π)`）的约定做注释

3. **使用示例**：  
   - 在 docstring 或独立示例中展示：如何从 `Base_Task`/`Robot` 获取 `robot_root_xy`，以及如何用柱坐标放置物体

---

# 🧠 Thinking Instructions（思考方式）

1. **先分析再编码**  
   - 明确 world → cylinder 的数学关系：`(x,y,z)` 相对 `(cx,cy)` 的极坐标形式  
   - 七维位姿的转换：世界四元数 → 柱坐标局部基下的四元数（涉及旋转矩阵变换）

2. **列出实现步骤**  
   - 三维点转换公式  
   - 七维位姿中旋转部分的转换（世界旋转矩阵 → 柱坐标局部旋转矩阵 → 四元数）  
   - 逆变换的推导

3. **检查潜在问题（"What could go wrong?"）**  
   - `r=0` 时 theta 无定义  
   - 角度归一化到 `[-π, π]` 或 `[0, 2π)` 的一致性  
   - 批量输入（`np.ndarray` 形状）的支持与广播  
   - 与 `create_fan_table` 的 `theta_start`、`theta_end`、`center_deg` 的对应关系，确保放置逻辑与扇形桌几何一致

---

# 🧾 Final Prompt

```text
你是一位资深具身开发专家。请在 RoboTwin 项目中实现柱坐标系与世界坐标系的双向变换及放置工具。

## 任务概述
桌面为扇形，但物体仍用绝对世界坐标描述。需要实现以机器人 root 为中心的柱坐标系，使放置物体时可以用 (r, theta, z) 或 (r, theta, z, qw, qx, qy, qz) 描述。

## 柱坐标系约定
- 轴心：机器人 root 的 xy 为柱坐标轴心 (cx, cy)
- z 轴：与世界 z 轴重合
- 角度：绕 z 轴右手定则，机器人左转为正，theta=0 为机器人正对桌子方向

## 必须实现的函数
1. world_to_robot(world_pt, robot_root_xy, robot_yaw_rad=0)
   - 输入：3D (x,y,z) 或 7D (x,y,z,qw,qx,qy,qz)
   - 输出：3D (r,theta,z) 或 7D (r,theta,z,qw,qx,qy,qz)

2. robot_to_world(robot_pt, robot_root_xy, robot_yaw_rad=0)
   - 输入：3D (r,theta,z) 或 7D (r,theta,z,qw,qx,qy,qz)
   - 输出：3D (x,y,z) 或 7D (x,y,z,qw,qx,qy,qz)

3. 七维柱坐标中，四元数在柱坐标局部基下：x=径向向外，y=切向（theta 增大），z=世界 z

4. 封装：place_point_cyl、place_pose_cyl、rand_pose_cyl 等，便于在柱坐标下放置物体

## 约束
- 与 envs/utils/transforms.py 中的 _toPose、pose2list 等保持一致
- 七维格式 [x,y,z,qw,qx,qy,qz]
- 处理 r=0 等退化情况
- 新建 envs/utils/cylindrical_coords.py，并在 __init__.py 中导出

请给出完整实现、简要文档和使用示例。
```

---

# 📥 输入任务（原始描述）
> 现在还有个问题就是虽然桌面变成了扇形，但是描述物体的时候仍然为绝对坐标，导致放置物体的时候不方便！现在请写一个转换函数 :world_to_robot和其逆函数，作用是将机器人面前的柱标系和世界坐标系之间的变换。输入是七维坐标或者三维点。规定：按照机器人的root的xy为柱坐标的轴中心点，world z=0和柱的z=0对齐，按照z轴右手定则为旋转方向，也就是机器人左转为正。机器人初始方向角度设定为0（也就是正对桌子为0），柱坐标系中的一点可以表示为（r,theta,z），同时你需要定义出在柱坐标系下的七维坐标和其与世界坐标系的转换。最后你需要封装一些函数方便使用轴坐标系来放置物体。
