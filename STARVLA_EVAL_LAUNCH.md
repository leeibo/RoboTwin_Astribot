# StarVLA RoboTwin 评测启动指南

本文说明如何启动 `eval_seed_whitelist_randomized.sh` 支持的各个 StarVLA
模型。命令不依赖固定机器路径；换机器后只需要重新设置仓库、Python 环境和
模型资源路径。

## 1. 目录和环境变量

推荐将两个仓库放在同一父目录下：

```text
workspace/
├── RoboTwin_Astribot/
└── starVLA-A/
```

也可以放在任意位置，此时显式设置以下变量：

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin_Astribot
export STARVLA_ROOT=/path/to/starVLA-A

export ROBOTWIN_PYTHON=/path/to/conda/envs/RoboTwin/bin/python
export STARVLA_PYTHON=/path/to/conda/envs/starVLA/bin/python

export STARGVLA_FAST_TOKENIZER="$STARVLA_ROOT/playground/Pretrained_models/fast"
```

如果两个仓库互为同级目录，`STARVLA_ROOT` 可以省略。两个 Python 变量也可以
省略，启动脚本会尝试查找名为 `RoboTwin` 和 `starVLA` 的 conda 环境。

脚本默认从每个训练结果的 `config.yaml` 解析 base VLM。模型文件建议保持以下
目录结构：

```text
$STARVLA_ROOT/
├── ckpt_mapping.yaml
├── playground/Pretrained_models/
│   ├── Qwen3-VL-2B-Instruct/
│   ├── Qwen3-VL-2B-Instruct-Action/
│   └── fast/
└── results/<MODEL_NAME>/
    ├── config.yaml
    └── checkpoints/steps_<STEP>_pytorch_model.pt
```

如果 base VLM 或 FAST tokenizer 位于其他目录，可覆盖：

```bash
export STARGVLA_BASE_VLM=/path/to/Qwen3-VL-2B-Instruct
export STARGVLA_FAST_TOKENIZER=/path/to/fast
```

`fast_subtask_action_12_ws` 使用
`Qwen3-VL-2B-Instruct-Action`。通常不要全局设置
`STARGVLA_BASE_VLM`，让脚本分别读取各模型的 `config.yaml`；只有模型配置里的
路径在新机器上不可用时才覆盖它。

## 2. 推荐启动方式

先进入 RoboTwin 仓库：

```bash
cd "$ROBOTWIN_ROOT"
```

建议在一个持久化的 tmux 会话里运行总启动脚本：

```bash
tmux new-session -s robotwin-eval
```

进入 tmux 后运行：

```bash
MODEL=oft_subtask_action_12_ws
GPU_LIST="0,1,2,3" \
EVAL_TEST_NUM=50 \
STARVLA_ROOT="$STARVLA_ROOT" \
ROBOTWIN_PYTHON="$ROBOTWIN_PYTHON" \
STARVLA_PYTHON="$STARVLA_PYTHON" \
bash ./eval_seed_whitelist_randomized.sh "$MODEL"
```

总启动脚本会为每张卡启动一个策略服务和一个 RoboTwin worker，并在所有 worker
之间共享 35 个任务的队列。端口规则如下：

- Action server: `19000 + GPU ID`
- Planner server: `20000 + GPU ID`，仅 `planner_oft` 使用

不要只运行 StarVLA 里的 `run_policy_server.sh`。该脚本只会把模型加载到显卡并
监听端口，不会启动 `eval_policy.py`，因此不会真正执行 RoboTwin 任务。

启动后可用下面的命令确认服务端和评测端都存在：

```bash
pgrep -af 'server_policy.py|script/eval_policy.py'
nvidia-smi
```

## 3. 各模型启动命令

下面的 step 来自 `starVLA-A/ckpt_mapping.yaml`。先统一设置 GPU 和 episodes：

```bash
export GPU_LIST="0,1,2,3"
export EVAL_TEST_NUM=50
cd "$ROBOTWIN_ROOT"
```

| 模型 | Step | 启动命令 |
|---|---:|---|
| `fast_subtask_action_12_ws` | 100,000 | `bash ./eval_seed_whitelist_randomized.sh fast_subtask_action_12_ws` |
| `gr00t_subtask_action_12_ws` | 90,000 | `bash ./eval_seed_whitelist_randomized.sh gr00t_subtask_action_12_ws` |
| `oft_instruction_action_12_ws` | 100,000 | `bash ./eval_seed_whitelist_randomized.sh oft_instruction_action_12_ws` |
| `oft_no_action_12_ws` | 100,000 | `bash ./eval_seed_whitelist_randomized.sh oft_no_action_12_ws` |
| `oft_no_no_0_wos` | 20,000 | `bash ./eval_seed_whitelist_randomized.sh oft_no_no_0_wos` |
| `oft_subtask_action_12_wos` | 90,000 | `bash ./eval_seed_whitelist_randomized.sh oft_subtask_action_12_wos` |
| `oft_subtask_action_12_ws` | 100,000 | `bash ./eval_seed_whitelist_randomized.sh oft_subtask_action_12_ws` |
| `oft_subtask_action_6_ws` | 55,000 | `bash ./eval_seed_whitelist_randomized.sh oft_subtask_action_6_ws` |
| `oft_subtask_motion_12_ws` | 100,000 | `bash ./eval_seed_whitelist_randomized.sh oft_subtask_motion_12_ws` |
| `oft_subtask_motion_6_ws` | 100,000 | `bash ./eval_seed_whitelist_randomized.sh oft_subtask_motion_6_ws` |
| `oft_subtask_subtask_12_ws` | 95,000 | `bash ./eval_seed_whitelist_randomized.sh oft_subtask_subtask_12_ws` |
| `oft_subtask_subtask_6_ws` | 100,000 | `bash ./eval_seed_whitelist_randomized.sh oft_subtask_subtask_6_ws` |
| `planner_oft` | Action 55,000 / Planner 90,000 | `bash ./eval_seed_whitelist_randomized.sh planner_oft` |

`gr00tdual_subtask_action_12_wos` 和 `gr00tdual_subtask_action_12_ws` 当前在
`ckpt_mapping.yaml` 的 `unavailable` 区域中，没有可用 checkpoint，启动器会拒绝
启动，而不会静默选择其他 step。

`planner_oft` 会同时读取 `planner_oft` 和 `planner_oft_planner` 两个 checkpoint，
并在每张 GPU 上启动 action server 与 planner server。它比普通模型需要更多显存；
显存不足时应减少 `GPU_LIST` 中的卡数，而不是让两个 worker 共用同一端口。

### 子任务关键帧模型的 Qwen 服务

`oft_subtask_subtask_12_ws` 和 `oft_subtask_subtask_6_ws` 会调用兼容 OpenAI API 的
Qwen3.5-9B 服务。默认地址为：

```text
http://127.0.0.1:8000/v1/chat/completions
```

启动评测前先确认服务可用：

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

若服务使用其他地址，修改对应模型的 `deploy_policy.yml` 中
`subtask_planner_url`。`oft_subtask_subtask_6_ws` 还提供一个等待空闲 GPU 和 Qwen
健康检查的后台启动助手：

```bash
GPU_COUNT=4 \
CANDIDATE_GPUS="0 1 2 3 4 5" \
QWEN_HEALTH_URL=http://127.0.0.1:8000/v1/models \
bash policy/oft_subtask_subtask_6_ws/run_eval_50ep.sh
```

## 4. 启动前检查

启动器会检查 checkpoint、Python、base VLM、FAST tokenizer、随机种子列表和 CUDA
工具链。可以先做一次不启动服务和评测进程的配置检查：

```bash
GPU_LIST="0" \
EVAL_TEST_NUM=50 \
DRY_RUN=1 \
bash ./eval_seed_whitelist_randomized.sh oft_subtask_action_12_ws
```

正式启动前还应确认目标端口没有被旧服务占用：

```bash
ss -ltnp | grep -E ':(1900[0-9]|2000[0-9])\b' || true
```

如果端口已被同一模型的孤立策略服务占用，应先正常停止旧服务，再从总启动脚本
重新启动。不要在同一端口上叠加第二个服务。

## 5. 日志、报告和恢复

每轮输出位于：

```text
$ROBOTWIN_ROOT/logs/eval_seed_whitelist_randomized/<MODEL_NAME>/<RUN_ID>/
├── report.md
├── summary.tsv
├── servers.tsv
├── tasks/
├── servers/
├── gpu_consoles/
└── eval_result/
```

报告只有同时满足下面两项时才算完整：

```text
Progress: 35/35
Failed: 0
```

需要从上一轮继续时，可以导入上一轮中成功的任务：

```bash
MODEL=oft_subtask_action_12_ws
PREVIOUS_RUN="$ROBOTWIN_ROOT/logs/eval_seed_whitelist_randomized/$MODEL/<RUN_ID>"

GPU_LIST="0,1,2,3" \
EVAL_TEST_NUM=50 \
RESUME_FROM_RUN="$PREVIOUS_RUN" \
bash ./eval_seed_whitelist_randomized.sh "$MODEL"
```

只有 checkpoint、任务配置、种子列表和 episodes 数量都相同时才应使用
`RESUME_FROM_RUN`，否则新报告会混入不同评测设置的结果。

调试时可限制任务数量：

```bash
GPU_LIST="0" TASK_LIMIT=1 EVAL_TEST_NUM=1 \
bash ./eval_seed_whitelist_randomized.sh oft_subtask_action_12_ws
```

固定种子版本使用相同的模型名和环境变量，但将入口替换为：

```bash
bash ./eval_seed_whitelist.sh "$MODEL"
```

## 6. FastWAM 独立入口

FastWAM 不使用 StarVLA 策略服务，入口为
`eval_fastwam_seed_whitelist_randomized.sh`。FastWAM 源码、Python 环境和 checkpoint
默认从同级或本机路径读取，也都可以显式覆盖：

```bash
cd "$ROBOTWIN_ROOT"

FASTWAM_ROOT=/path/to/FastWAM \
FASTWAM_PYTHON=/path/to/conda/envs/fastwam/bin/python \
CHECKPOINT_PATH=/path/to/step_008995.pt \
GPU_LIST="0,1" \
EVAL_TEST_NUM=50 \
bash ./eval_fastwam_seed_whitelist_randomized.sh
```

如果共享存储上的依赖加载不稳定，可以先构建节点本地运行副本，再从本地副本
启动。checkpoint 仍保留在原位置，不会被复制：

```bash
SOURCE_FASTWAM_ROOT=/path/to/FastWAM \
SOURCE_FASTWAM_ENV=/path/to/conda/envs/fastwam \
bash ./prepare_fastwam_local_runtime.sh

USE_LOCAL_RUNTIME=1 \
GPU_LIST="0,1" \
EVAL_TEST_NUM=50 \
bash ./eval_fastwam_seed_whitelist_randomized.sh
```

FastWAM 的报告位于
`logs/eval_fastwam_seed_whitelist_randomized/<RUN_ID>/report.md`。本地生成的
`policy/fastwam_policy` 软链接和 `.fastwam_eval_*.pt` checkpoint 别名已被 Git
忽略，不应提交。
