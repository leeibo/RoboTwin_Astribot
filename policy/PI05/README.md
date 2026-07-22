# PI05 Astribot 评测

本目录是 PI05 在 RoboTwin Astribot 上的评测适配。RoboTwin 进程负责仿真和任务调度，
独立 RLinf/OpenPI websocket server 负责模型推理；adapter 将 head RGB、18 维状态和指令
发送给 server，并执行返回的 action chunk。

本文是当前唯一使用说明。历史单任务 A/B seed 配置、某次补跑命令和运行日志不属于可提交
源码，已通过 `.gitignore` 排除。

## 目录内容

| 文件 | 用途 |
|---|---|
| `ckpt_mapping.yaml` | PI05 checkpoint、RLinf source 和 OpenPI config 映射 |
| `deploy_policy.yml` | RoboTwin adapter 默认参数 |
| `deploy_policy.py` | websocket client、状态/action 转换和评测入口 |
| `install_env.sh` | 安装 RoboTwin 侧轻量 websocket/msgpack 依赖 |
| `check.sh` | 检查两侧环境、checkpoint、配置、任务表和 seed list |
| `eval.sh` | 单 GPU `smoke`、`formal`、`custom` 评测 |
| `eval_seed_whitelist_randomized.sh` | 多 GPU 持久 server、动态任务队列和续跑 |
| `run_full_100k_clean_randomized.sh` | 按 0-4 / 5-8 GPU 同时启动 Clean 和 Randomized |
| `analyze_ab_debug.py` | 可选的 run 级 torso/action debug 汇总工具 |

`runs/`、`nohup_logs/`、`run_configs/`、`env/robotwin_client_deps/` 和缓存均为本地产物，
不会进入提交。

## 默认运行合同

- 默认 checkpoint key：`pi05_astribot_global_step_100000`
- 可选 checkpoint key：`pi05_astribot_global_step_20000_new`
- checkpoint：`/root/autodl-tmp/ckpt/baseline_model/REAL/PI05/.../actor`
- PI05 project：`/root/autodl-tmp/PI05_eval/PI05`
- RLinf source：`/root/autodl-tmp/PI05_eval/PI05/source/RLinf`
- PI05 Python：`/root/autodl-tmp/PI05_eval/PI05/env/rlinf-pi05/bin/python`
- RoboTwin Python：`/root/autodl-tmp/conda_env/RoboTwin/bin/python`
- OpenPI client source：仓库内 `policy/pi05/packages/openpi-client/src`
- action/state 有效维度：18；模型 action horizon：50
- 默认每次请求最多执行 16 个 action，模型 denoising steps 为 5
- 动作语义：absolute
- 默认关闭 SAPIEN OIDN

`ROBOTWIN_PYTHON` 和 `PI05_PYTHON` 必须分离：前者加载仿真和轻量 client，后者加载
RLinf/OpenPI server。权重、完整 PI05 环境和运行结果不应复制进本目录。

## 准备 client 依赖

PI05/RLinf server 环境由外部 PI05 project 管理。本目录的安装脚本只准备 RoboTwin 侧
`websockets` 和 `msgpack`：

```bash
DRY_RUN=1 bash policy/PI05/install_env.sh
bash policy/PI05/install_env.sh
```

默认使用国内镜像；使用官方 PyPI 时设置 `USE_CN_MIRRORS=0`。安装目标默认为
`policy/PI05/env/robotwin_client_deps`，该目录可随时重建且不会上传。

## 检查

允许 PI05 server 环境暂缺的代码和数据检查：

```bash
ALLOW_MISSING_ENV=1 EVAL_TEST_NUM=1 bash policy/PI05/check.sh
```

正式评测前执行严格检查：

```bash
bash policy/PI05/check.sh
```

检查覆盖 RoboTwin/PI05 Python、client import、RLinf import、checkpoint required files、
任务白名单和对应 task config 的 seed list 数量。

## 单 GPU 评测

只解析配置和输出计划，不启动模型或仿真：

```bash
DRY_RUN=1 MODE=smoke TASK_CONFIG=info_gathering_demo \
GPU_ID=0 EVAL_TEST_NUM=1 bash policy/PI05/eval.sh
```

真实 Clean smoke：

```bash
MODE=smoke TASK_CONFIG=info_gathering_demo GPU_ID=0 \
EVAL_TEST_NUM=1 CONTINUE_ON_ERROR=0 bash policy/PI05/eval.sh
```

Randomized smoke 使用 `TASK_CONFIG=info_gathering_randomized`。`MODE=formal` 默认完整任务表、
每任务 50 episodes；`MODE=custom` 使用显式 episode、task limit 和 timeout 设置。

已有外部 server 时：

```bash
PI05_SERVER_MANAGED=0 PI05_HOST=127.0.0.1 PI05_PORT=5702 \
MODE=smoke TASK_CONFIG=info_gathering_demo bash policy/PI05/eval.sh
```

## 多 GPU 评测

一任务 dry-run：

```bash
DRY_RUN=1 TMUX_MONITOR=0 GPU_LIST=0 TASK_LIMIT=1 EVAL_TEST_NUM=1 \
TASK_CONFIG=info_gathering_demo \
bash policy/PI05/eval_seed_whitelist_randomized.sh
```

Clean 全量示例：

```bash
GPU_LIST="0 1 2 3 4" TASK_CONFIG=info_gathering_demo \
PI05_CKPT_KEY=pi05_astribot_global_step_100000 \
EVAL_TEST_NUM=50 TASK_LIMIT=0 PI05_MAX_ACTIONS_PER_CALL=16 \
bash policy/PI05/eval_seed_whitelist_randomized.sh
```

Randomized 全量示例：

```bash
GPU_LIST="5 6 7 8" TASK_CONFIG=info_gathering_randomized \
PI05_CKPT_KEY=pi05_astribot_global_step_100000 \
EVAL_TEST_NUM=50 TASK_LIMIT=0 PI05_MAX_ACTIONS_PER_CALL=16 \
bash policy/PI05/eval_seed_whitelist_randomized.sh
```

每张 GPU 启动一个持久 PI05 server 和一个 RoboTwin worker，任务通过加锁队列动态分配。
端口为 `PI05_PORT_BASE + physical_gpu_id`，默认 base 是 5702。

## 同时启动 Clean 和 Randomized

推荐先验证启动计划：

```bash
DRY_RUN=1 TASK_LIMIT=1 EVAL_TEST_NUM=1 \
bash policy/PI05/run_full_100k_clean_randomized.sh
```

正式启动：

```bash
EVAL_TEST_NUM=50 TASK_LIMIT=0 ROBOTWIN_EVAL_VIDEO_LOG=False \
bash policy/PI05/run_full_100k_clean_randomized.sh
```

该脚本后台启动两组任务，默认 GPU 0-4 用于 Clean、5-8 用于 Randomized；入口日志写入
`nohup_logs/`，完整运行结果写入 `runs/eval_seed_whitelist_randomized/`。

## 断点续跑

```bash
GPU_LIST="0 1 2 3 4" TASK_CONFIG=info_gathering_demo EVAL_TEST_NUM=50 \
RESUME_FROM_RUN=/absolute/path/to/policy/PI05/runs/eval_seed_whitelist_randomized/<run_id> \
bash policy/PI05/eval_seed_whitelist_randomized.sh
```

状态为 0 的任务会跳过。续跑必须保持 checkpoint、task config、episode 数、task limit、
action horizon、每次执行 action 数和 action semantics 一致。

## 重要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `PI05_NUM_STEPS` | 5 | 模型 denoising iterations，不是执行 action 数 |
| `PI05_ACTION_HORIZON` | 50 | server 返回的最大 action horizon |
| `PI05_MAX_ACTIONS_PER_CALL` | 16 | 每次推理实际执行的 action 数 |
| `PI05_ACTION_SEMANTICS` | `absolute` | 可选 `absolute` 或 `delta_to_abs` |
| `PI05_REQUEST_TIMEOUT` | 180 | 多 GPU 脚本的单次请求超时秒数 |
| `TASK_TIMEOUT_SECONDS` | 0/按模式 | 多 GPU 中 0 表示禁用单任务超时 |
| `ROBOTWIN_EVAL_VIDEO_LOG` | `False` | 是否保存 episode 视频 |
| `ROBOTWIN_SAPIEN_RT_DENOISER` | `none` | 当前默认关闭 OIDN |
| `TMUX_MONITOR` | 1 | 多 GPU 时创建只读监控窗口 |

完整多 GPU 参数以 `bash policy/PI05/eval_seed_whitelist_randomized.sh --help` 为准。

## 输出和分析

单 GPU 输出位于 `runs/eval/<run_id>/`，多 GPU 输出位于
`runs/eval_seed_whitelist_randomized/<run_id>/`。每个 run 包含 manifest、summary、report、
server/task 日志、request debug 和 eval result。

需要比较多个 run 的 torso/action debug 时：

```bash
python policy/PI05/analyze_ab_debug.py \
  policy/PI05/runs/eval/<run-a> \
  policy/PI05/runs/eval/<run-b>
```

- server readiness 超时：检查 `PI05_PYTHON`、`PI05_RLINF_ROOT`、checkpoint 和端口。
- websocket client import 失败：重新运行 `install_env.sh`，不要提交生成的 `env/`。
- action shape/norm stats 错误：运行 `check.sh` 并检查 checkpoint 的
  `robotwin_astribot_pi05/norm_stats.json`。
- CUDA/OIDN 错误：保持 `ROBOTWIN_SAPIEN_RT_DENOISER=none`，确认 worker GPU 隔离。
