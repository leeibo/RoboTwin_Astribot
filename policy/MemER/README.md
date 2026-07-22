# MemER Astribot 评测

本目录是 MemER 在 RoboTwin Astribot 上的完整评测适配。MemER 由两个独立模型服务组成：

1. 高层 Qwen3-VL HTTP server 根据完整任务、近期帧和 episode memory 生成当前 subtask。
2. 低层 RLinf/OpenPI websocket server 根据 subtask 生成 50 步 action chunk。
3. RoboTwin adapter 负责 session、重规划、动作执行、失败状态和日志。

本文是当前唯一使用说明。历史设计问答和阶段性调试指南已移除，避免与实际脚本分叉。

## 目录内容

| 文件 | 用途 |
|---|---|
| `ckpt_mapping.yaml` | 高层/低层 checkpoint、source 和配置映射 |
| `deploy_policy.yml` | RoboTwin adapter 默认参数 |
| `deploy_policy.py` | 高低层 client、session、重规划和 action 转换 |
| `serve_high_policy.py` | Qwen3-VL 高层 HTTP server |
| `serve_low_policy.py` | RLinf/OpenPI 低层 websocket server |
| `install_env.sh` | 创建隔离的高层、低层和 RoboTwin client 环境 |
| `check.sh` | 检查环境、权重、processor、协议和 seed list |
| `eval.sh` | 单环境组 `smoke`、`formal`、`custom` 评测 |
| `eval_seed_whitelist_randomized.sh` | 单个 Clean/Randomized 组的动态调度和续跑 |
| `eval_all.sh` | 九卡同时启动 Clean 与 Randomized 全量评测 |
| `tests/test_memer_eval.py` | 不加载真实权重的高层、低层和 adapter 测试 |

`runs/`、`env/robotwin_client_deps/`、日志和 Python 缓存均为本地产物，由
`.gitignore` 排除。

## 默认运行合同

- 默认 checkpoint key：`memer_astribot_step_4500_18000`
- 可选 checkpoint key：`memer_astribot_step_13500_18000`
- MemER source：`/root/autodl-tmp/MEMER_eval/MemER`
- RoboTwin Python：`/root/autodl-tmp/conda_env/RoboTwin/bin/python`
- 高层环境：`/root/autodl-tmp/MEMER_eval/MemER/env/qwen3vl`
- 低层环境：`/root/autodl-tmp/MEMER_eval/MemER/env/rlinf-pi05`
- 高层使用 `transformers==4.57.0`；低层使用 RLinf 补丁版 `4.53.2`，不能混用环境
- 高层输入图像固定缩放为 320x180
- 低层 action horizon 50，默认每 5 个 env steps 请求一次高层并执行最多 5 个 action
- action/state 有效维度为 18；低层 norm stats 内部为 32 维，后 14 维是 padding
- 动作语义为 absolute
- 默认关闭 SAPIEN OIDN

路径由 `ckpt_mapping.yaml` 和脚本环境变量共同控制。权重、模型 source 和完整 Python
环境不应复制或提交到本目录。

## 安装

安装脚本创建两个隔离模型环境，并在 `policy/MemER/env/robotwin_client_deps` 安装轻量
websocket 依赖。默认使用国内镜像：

```bash
DRY_RUN=1 bash policy/MemER/install_env.sh
bash policy/MemER/install_env.sh
```

已有兼容 PI05 低层环境时，默认会自动复用；也可以显式要求：

```bash
REUSE_PI05_LOW_ENV=1 bash policy/MemER/install_env.sh
```

仅重建 RoboTwin client 依赖：

```bash
INSTALL_HIGH=0 INSTALL_LOW=0 INSTALL_CLIENT=1 \
bash policy/MemER/install_env.sh
```

使用官方源时设置 `USE_CN_MIRRORS=0`。其他安装参数以
`bash policy/MemER/install_env.sh --help` 为准。

## 检查和轻量测试

允许模型环境暂缺的代码检查：

```bash
ALLOW_MISSING_ENV=1 bash policy/MemER/check.sh
```

正式评测前执行严格检查：

```bash
bash policy/MemER/check.sh
```

运行不加载真实模型的单元测试：

```bash
/root/autodl-tmp/conda_env/RoboTwin/bin/python \
  -m unittest policy/MemER/tests/test_memer_eval.py
```

## 单环境组评测

只验证参数、checkpoint 和 seed list，不启动服务：

```bash
DRY_RUN=1 MODE=smoke ENVIRONMENT_TYPE=clean \
TASK_NAME=beat_block_hammer_rotate_view EVAL_TEST_NUM=1 \
HIGH_GPU=0 LOW_GPU=1 bash policy/MemER/eval.sh
```

使用 mock 服务验证完整进程和协议链路：

```bash
MODE=smoke ENVIRONMENT_TYPE=clean \
TASK_NAME=beat_block_hammer_rotate_view EVAL_TEST_NUM=1 \
HIGH_GPU=0 LOW_GPU=1 HIGH_MOCK=1 LOW_MOCK=1 \
CONTINUE_ON_ERROR=0 bash policy/MemER/eval.sh
```

mock action 不具备完成任务的能力，success rate 为 0 是正常结果；此步骤只验证服务生命周期、
协议、action shape 和日志。

真实 Clean smoke：

```bash
MODE=smoke ENVIRONMENT_TYPE=clean \
TASK_NAME=beat_block_hammer_rotate_view EVAL_TEST_NUM=1 \
HIGH_GPU=0 LOW_GPU=1 CONTINUE_ON_ERROR=0 \
bash policy/MemER/eval.sh
```

真实 Randomized smoke：

```bash
MODE=smoke ENVIRONMENT_TYPE=randomized \
TASK_NAME=beat_block_hammer_rotate_view EVAL_TEST_NUM=1 \
HIGH_GPU=5 LOW_GPU=6 CONTINUE_ON_ERROR=0 \
bash policy/MemER/eval.sh
```

Clean 必须搭配 `info_gathering_demo`，Randomized 必须搭配
`info_gathering_randomized`；launcher 会拒绝错配。

## 单组并发

一个高层服务可供多个低层/simulation worker 共享。先从小规模验证：

```bash
ENVIRONMENT_TYPE=clean HIGH_GPU=0 LOW_GPUS=1,2 \
MODE=custom TASK_LIMIT=2 EVAL_TEST_NUM=1 TMUX_MONITOR=0 \
bash policy/MemER/eval_seed_whitelist_randomized.sh
```

Clean 默认使用高层 GPU 0、低层 GPU 1-4；Randomized 默认使用高层 GPU 5、低层
GPU 6-8。高层推理请求排队执行，各 worker 的 environment/episode/session state 独立。

## 九卡全量评测

先验证两个组的启动计划：

```bash
DRY_RUN=1 TASK_LIMIT=1 EVAL_TEST_NUM=1 TMUX_MONITOR=0 \
bash policy/MemER/eval_all.sh
```

同时运行 Clean 和 Randomized 的 35 tasks x 50 episodes：

```bash
EVAL_TEST_NUM=50 TASK_LIMIT=0 TASK_TIMEOUT_SECONDS=0 TMUX_MONITOR=0 \
bash policy/MemER/eval_all.sh
```

资源划分：

```text
GPU 0: Clean high              GPU 1-4: Clean low + simulation
GPU 5: Randomized high         GPU 6-8: Randomized low + simulation
```

## 断点续跑

单个环境组中断后，将原目录传给调度器：

```bash
ENVIRONMENT_TYPE=clean EVAL_TEST_NUM=50 \
RESUME_FROM_RUN=/absolute/path/to/policy/MemER/runs/eval_seed_whitelist_randomized/<run_id> \
bash policy/MemER/eval_seed_whitelist_randomized.sh
```

状态为 0 的任务会跳过。续跑必须保持 environment type、checkpoint、task config、episode
数、task limit、执行 horizon、重规划间隔和 recent-frame 间隔一致。

## 时序和失败口径

三个参数彼此独立：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `LOW_LEVEL_EXECUTION_HORIZON` | 5 | 每个低层 chunk 最多实际执行的 action 数 |
| `HIGH_LEVEL_REPLAN_INTERVAL` | 5 | 高层 subtask 重规划间隔 |
| `RECENT_FRAME_INTERVAL` | 5 | 高层 recent context 的原始 env-frame 采样间隔 |
| `RECENT_FRAMES` | 8 | recent context 最大帧数 |
| `MEMORY_FRAMES` | 8 | episode memory 最大帧数 |
| `VLM_LOG_IMAGES` | 1 | 是否保存实际送入高层的去重图像 |

高层输出严格只接受 `current_subtask` 和 `keyframe_positions`。首次解析失败会立即重试；
连续失败时，已有合法 subtask 最多复用一个 replan 周期，随后标记该 episode 失败。失败
episode 保留在成功率分母中。

## 输出与排障

单组输出位于 `runs/eval/<run_id>/`；动态调度输出位于
`runs/eval_seed_whitelist_randomized/<run_id>/`。每个 run 包含 manifest、summary、report、
服务日志、任务日志、结构化 request debug 和 eval result。高层 VLM trace 位于
`vlm_logs/vlm_requests.jsonl`。

- readiness 超时：检查端口占用、两套 Python、checkpoint 和 server token 对应的日志。
- processor contract 错误：确认高层 checkpoint 目录包含训练时的
  `preprocessor_config.json` 和 `video_preprocessor_config.json`。
- subtask 解析失败：查看 VLM trace 中的完整 prompt、原始输出和解析错误。
- 低层动作异常：查看 `memer_request_debug.jsonl` 中的 state18、action18、prompt 和 norm
  stats 检查结果。
- 中断后残留进程：只终止对应 run manifest/server log 中记录的 PID，不使用模糊的全局
  kill 命令。
