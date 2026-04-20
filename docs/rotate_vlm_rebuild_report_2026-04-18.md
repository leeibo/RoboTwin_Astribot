# Rotate VLM Rebuild Report

日期: 2026-04-18

## 1. 执行范围

- 配置: `demo_randomized_easy_ep2`
- 白名单任务: `task_config/rotate_task_whitelist.yml`
- 任务数: 26 个 rotate-view 任务
- 每个任务采集: 2 个 episode

## 2. 本次实现的关键修改

### 2.1 Rotate VLM / VQA 链路

- 重写了 rotate memory compression 规则:
  - 先合并连续 `Rotate(0, 0)` 区间，只保留每段最新帧
  - 再按时间从老到新增量加入帧
  - 新帧加入后，如果旧帧已被其它保留帧覆盖，则删除旧帧
- `stage1` / `stage2` 起止帧进入记忆，`stage3` 按 action chunk 进入记忆
- memory compression VQA 支持从压缩最优帧集合扩增大量子集样本
- angle delta 改为同一子任务、stage<=2、基于规划角度的累计水平旋转量
- object search:
  - `stage2` 视为信息充分
  - 若目标只出现在历史帧，则 `think` 显式写成“在第 k 帧 (x, y) 发现目标”
  - `stage3` 的 `action` 字段改为真实 10 步 chunk，而不是占位符

### 2.2 可见性与标注

- AABB 可见性改为 40% 暴露阈值
- frame annotation 增加:
  - `visible_object_uv_map`
  - `discovered_last_uv_map`
  - `visible_object_ratio_map`
- object-search QA 视频右侧字段收敛为:
  - `Video Frame`
  - `QA Step`
  - `Q`
  - `Q think`
  - `info`
  - `frame`
  - `camera`

### 2.3 Randomized / 任务稳定性

- `demo_randomized.yml` 与 `demo_randomized_easy_ep2.yml` 调整为 fan randomized 且更稳定的配置
- fan clutter 采样改为扇形区域采样，满足 `r >= 0.55`，并偏向后侧
- 过滤掉缺失资产的 objaverse 杂物，避免把不存在的 `model.urdf` 抽进采集流程
- 对失败任务做了任务级稳定性收敛:
  - `click_alarmclock_rotate_view`
  - `place_a2b_left_rotate_view`
  - `place_a2b_right_rotate_view`
  - `place_burger_fries_rotate_view`
  - `place_cans_plasticbox_rotate_view`

## 3. 采集执行过程

### 3.1 第一轮白名单全量采集

- 先清空了 `data/`
- 首轮结果:
  - 成功: 21
  - 失败: 5
- 首轮失败任务:
  - `click_alarmclock_rotate_view`
  - `place_a2b_left_rotate_view`
  - `place_a2b_right_rotate_view`
  - `place_burger_fries_rotate_view`
  - `place_cans_plasticbox_rotate_view`

### 3.2 补救动作

- 修复杂物缺失资产问题
- 对失败任务收紧任务物体的可执行区域，并扩大禁止杂物侵入的 padding
- 对 `click_alarmclock` 增加 press pose fallback:
  - 先尝试基于接触点的 top-down 候选位姿
  - 失败时退回到物体中心上方的 top-down press pose
- 清空失败任务目录后，按修正后的逻辑从头重采失败任务

## 4. 最终产物

### 4.1 采集结果

- 最终任务完成数: 26 / 26
- 最终 episode 数: 52
- 最终 object-search QA 注释视频数: 52

### 4.2 VQA 总量

- `object_search`: 854
- `angle_delta`: 199
- `memory_compression_vqa`: 11685

### 4.3 汇总文件

- 白名单导出汇总 JSON:
  - `data/collection_reports/export_rotate_vlm_whitelist__demo_randomized_easy_ep2__summary.json`

## 5. 验收结果

- 26 个白名单任务目录全部存在
- 每个任务都存在:
  - 2 个 `episode*.hdf5`
  - 2 个 `subtask_metadata/episode*.json`
  - 3 类 `vlm/*.json`
  - 2 个 `episode*_annotated.mp4`
  - 2 个 `episode*_annotated_object_search_qa.mp4`
- `collection_failure.json` 已清零
- 52 个 QA 注释视频全部通过编码检查:
  - codec: `h264`
  - pixel format: `yuv420p`

## 6. 本次跑过的关键验证

- `pytest -q test/test_camera_visibility.py test/test_rotate_memory_compression.py test/test_rotate_randomized_fan_sampling.py test/test_rotate_object_search_render.py`
- `python script/export_rotate_vlm_whitelist.py --task-config demo_randomized_easy_ep2 --summary-path data/collection_reports/export_rotate_vlm_whitelist__demo_randomized_easy_ep2__summary.json`
- `python script/render_object_search_qa_video.py --data-root data --overwrite`
- 额外做了目录完整性检查和全量 `ffprobe` 编码检查

## 7. 备注

- 某些简单任务没有 memory compression 触发事件，因此 `memory_compression_vqa.json` 中样本数为 0，这属于预期结果，不是导出缺失
- 本次最终结果基于修正后的 randomized-easy 配置和任务级稳定性修补重新生成
