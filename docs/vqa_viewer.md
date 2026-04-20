# VQA Viewer

## 用途

本地网页工具，用来浏览 `data/*/*/vlm/*.json` 里的 VQA 样本，并联动查看:

- 输入图片序列
- 解析后的 `think / info / frame / camera / action / answer`
- 原始 prompt / completion
- metadata
- annotated 视频与 QA overlay 视频

它不依赖 Node.js，也不会改写现有数据。只需要启动一个本地 Python 服务。

## 启动方式

在仓库根目录运行:

```bash
python script/vqa_viewer_server.py
```

默认地址:

```text
http://127.0.0.1:8765
```

可选参数:

```bash
python script/vqa_viewer_server.py --host 0.0.0.0 --port 9000 --data-root data
```

## 主要功能

- 左侧任务列表:
  - 按任务名搜索
  - 按 storage 过滤
  - 快速查看三类 VQA 数量

- 顶部类型切换:
  - `object_search`
  - `angle_delta`
  - `memory_compression_vqa`

- 样本过滤:
  - episode
  - stage
  - subtask
  - 文本搜索
  - `history evidence only`

- 详情预览:
  - 图片缩略图与大图预览
  - 当前帧 / evidence 帧高亮
  - Prompt / Completion 原文
  - 解析标签
  - 原始 JSON
  - Annotated 视频与 QA 视频
  - 一键跳到当前样本帧附近

## 交互说明

- `j` / `k`: 下一条 / 上一条样本
- `/`: 聚焦样本搜索框
- `Esc`: 关闭大图预览

## 备注

- 视频使用 Range 方式返回，浏览器里可以正常拖动和 seek。
- 数据目录默认是仓库根目录下的 `data/`。
- 页面只在当前任务与当前类型下按需加载样本，不会一次性把全部 VQA JSON 全读到浏览器里。
