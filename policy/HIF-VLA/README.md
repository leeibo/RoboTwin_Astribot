# HIF-VLA Astribot 评测

本目录是 HIF-VLA 在 RoboTwin Astribot 上的评测适配。RoboTwin 负责仿真和任务调度，
独立 HIF-VLA HTTP server 负责模型推理；adapter 在两侧之间传递 head RGB、18 维状态、
自然语言指令和 18 维 action chunk。

本文是当前唯一使用说明。历史实现约定、单次补跑命令和运行日志不属于可提交源码。

## 目录内容

| 文件 | 用途 |
|---|---|
| `ckpt_mapping.yaml` | 模型、OpenVLA base、HIF-VLA source 和 checkpoint 映射 |
| `deploy_policy.yml` | RoboTwin adapter 默认参数 |
| `deploy_policy.py` | HTTP client、状态/action 转换和评测入口 |
| `serve_policy.py` | HIF-VLA HTTP server，提供 `/healthz`、`/reset`、`/act` |
| `install_env.sh` | 创建独立 HIF-VLA 推理环境 |
| `check.sh` | 检查环境、权重、维度合同、任务表和 seed list |
| `eval.sh` | 单 GPU `smoke`、`formal`、`custom` 评测 |
| `eval_seed_whitelist_randomized.sh` | 多 GPU 动态队列、汇总和断点续跑 |
| `tests/test_hifvla_eval.py` | 不加载真实模型的 adapter/server 单元测试 |

`runs/`、`nohup_logs/`、`env/`、缓存和日志由 `.gitignore` 排除，不应上传。

## 默认运行合同

- checkpoint key：`hifvla_astribot35_150k`
- checkpoint：`/root/autodl-tmp/ckpt/baseline_model/REAL/HIF-VLA/hifvla_astribot35_150k`
- OpenVLA base：`/root/autodl-tmp/ckpt/openvla/openvla-7b`
- HIF-VLA source：`/root/autodl-tmp/HIF-VLA_eval/HIF-VLA/source/HiF-VLA`
- HIF-VLA Python：`/root/autodl-tmp/HIF-VLA_eval/HIF-VLA/env/hifvla/bin/python`
- RoboTwin Python：`/root/autodl-tmp/conda_env/RoboTwin/bin/python`
- state/action：18 维，action horizon 8，每次最多执行 8 个 action
- 动作语义：absolute；adapter 转换为 RoboTwin 19 维 qpos action
- motion history：8 帧
- 默认关闭 SAPIEN OIDN，避免并行评测中的 GPU illegal memory access

路径变化时优先修改 `ckpt_mapping.yaml`，或通过脚本同名环境变量覆盖；不要把权重、模型
环境或绝对运行产物复制进本目录。

## 安装和检查

从仓库根目录执行。安装脚本默认使用国内镜像，可用 `USE_CN_MIRRORS=0` 切换官方源。

```bash
DRY_RUN=1 bash policy/HIF-VLA/install_env.sh
bash policy/HIF-VLA/install_env.sh
```

已有环境时可以只检查：

```bash
ALLOW_MISSING_ENV=1 EVAL_TEST_NUM=1 bash policy/HIF-VLA/check.sh
bash policy/HIF-VLA/check.sh
```

第一条允许模型环境暂缺，适合代码审查；正式评测前必须让严格检查通过。运行轻量测试：

```bash
/root/autodl-tmp/conda_env/RoboTwin/bin/python \
  -m unittest policy/HIF-VLA/tests/test_hifvla_eval.py
```

## 单 GPU 评测

只解析 checkpoint、任务和输出计划，不启动 server 或仿真：

```bash
DRY_RUN=1 MODE=smoke TASK_CONFIG=info_gathering_demo \
TASK_NAME=beat_block_hammer_rotate_view EVAL_TEST_NUM=1 GPU_ID=0 \
bash policy/HIF-VLA/eval.sh
```

真实 Clean smoke：

```bash
MODE=smoke TASK_CONFIG=info_gathering_demo \
TASK_NAME=beat_block_hammer_rotate_view EVAL_TEST_NUM=1 GPU_ID=0 \
CONTINUE_ON_ERROR=0 bash policy/HIF-VLA/eval.sh
```

Randomized smoke 只需将 `TASK_CONFIG` 改为 `info_gathering_randomized`。`MODE=formal`
默认运行完整任务表、每任务 50 episodes；`MODE=custom` 使用显式的 `TASK_NAME`、
`EVAL_TEST_NUM` 和 timeout 设置。

已有外部 server 时可设置：

```bash
HIFVLA_SERVER_MANAGED=0 HIFVLA_HOST=127.0.0.1 HIFVLA_PORT=5802 \
MODE=smoke TASK_CONFIG=info_gathering_demo \
bash policy/HIF-VLA/eval.sh
```

## 多 GPU 评测

先做一任务 dry-run：

```bash
DRY_RUN=1 TMUX_MONITOR=0 GPU_LIST=0 TASK_LIMIT=1 EVAL_TEST_NUM=1 \
TASK_CONFIG=info_gathering_demo \
bash policy/HIF-VLA/eval_seed_whitelist_randomized.sh
```

Clean 全量示例：

```bash
GPU_LIST="0 1 2 3 4" TASK_CONFIG=info_gathering_demo \
EVAL_TEST_NUM=50 TASK_LIMIT=0 ROBOTWIN_EVAL_VIDEO_LOG=False \
bash policy/HIF-VLA/eval_seed_whitelist_randomized.sh
```

Randomized 全量示例：

```bash
GPU_LIST="5 6 7 8" TASK_CONFIG=info_gathering_randomized \
EVAL_TEST_NUM=50 TASK_LIMIT=0 ROBOTWIN_EVAL_VIDEO_LOG=False \
bash policy/HIF-VLA/eval_seed_whitelist_randomized.sh
```

每张 GPU 启动一个持久 server 和一个 worker，任务通过加锁队列动态分配。端口为
`HIFVLA_PORT_BASE + physical_gpu_id`，默认 base 是 5802。

## 断点续跑

续跑时使用原 run 目录，并保持 checkpoint、task config、episode 数和动作参数一致：

```bash
GPU_LIST="0 1 2 3 4" TASK_CONFIG=info_gathering_demo EVAL_TEST_NUM=50 \
RESUME_FROM_RUN=/absolute/path/to/policy/HIF-VLA/runs/eval_seed_whitelist_randomized/<run_id> \
bash policy/HIF-VLA/eval_seed_whitelist_randomized.sh
```

`summary.tsv` 中状态为 0 的任务会跳过；不要用 Clean run 续跑 Randomized。

## 重要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `HIFVLA_ACTION_HORIZON` | 8 | server 返回的 action horizon，必须与 checkpoint 一致 |
| `HIFVLA_MAX_ACTIONS_PER_CALL` | 8 | 每次推理实际执行的 action 数 |
| `HIFVLA_HISTORY_LENGTH` | 8 | motion history 长度 |
| `HIFVLA_REQUEST_TIMEOUT` | 300 | 单次 HTTP 请求超时秒数 |
| `TASK_TIMEOUT_SECONDS` | 0/按模式 | 单任务超时；多 GPU 中 0 表示禁用 |
| `ROBOTWIN_EVAL_VIDEO_LOG` | `False` | 是否保存 episode 视频 |
| `ROBOTWIN_SAPIEN_RT_DENOISER` | `none` | 当前评测默认关闭 OIDN |
| `TMUX_MONITOR` | 1 | 多 GPU 时创建只读监控窗口 |

多 GPU 脚本的完整参数以 `bash policy/HIF-VLA/eval_seed_whitelist_randomized.sh --help`
为准。

## 输出与排障

单 GPU 输出位于 `runs/eval/<run_id>/`，多 GPU 输出位于
`runs/eval_seed_whitelist_randomized/<run_id>/`。每次运行独立保存 `manifest.yaml`、
`summary.tsv`、`report.md`、server/task 日志和 `eval_result/`。

- server readiness 超时：检查 `HIFVLA_PYTHON`、checkpoint 路径、端口和 server 日志。
- 维度或 stats 错误：先运行 `check.sh`，确认 18/18/8 合同和 `astribot_35_mix`。
- CUDA/OIDN 错误：保持 `ROBOTWIN_SAPIEN_RT_DENOISER=none`，并确认每个 worker 的
  `CUDA_VISIBLE_DEVICES` 隔离正常。
- 临时网络失败：调整 `HIFVLA_REQUEST_RETRIES`、`HIFVLA_RETRY_BACKOFF` 和请求超时。
