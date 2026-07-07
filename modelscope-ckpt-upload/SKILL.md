---
name: modelscope-ckpt-upload
description: Robustly upload one or more local checkpoint/output folders to the ModelScope model repository conroy1201/robotwin-ckpt. Use when the user asks to publish, sync, or upload RoboTwin checkpoints or other local folders to ModelScope, especially when there are many weight files or large .pt files that need multithreaded upload, retry handling, .pt chunk splitting, SHA256 manifests, and restore/verify support after download.
---

# Modelscope Ckpt Upload

## Overview

Upload local folders to `conroy1201/robotwin-ckpt` on ModelScope with a stability-first workflow. Each specified local folder becomes a same-level remote directory by default; files are staged locally, ignored files are omitted, large `.pt` files are split into checksum-tracked parts, and a SHA256 manifest is uploaded for download-time verification and restore.

## Workflow

1. Confirm the user provided one or more folder paths. If the folders or ignore rules are ambiguous, ask only for the missing paths/patterns.
2. Run the bundled uploader script; do not rewrite ad hoc upload commands unless the script cannot cover the request.
3. Let the script check ModelScope authentication first with `ms whoami` or `modelscope whoami`. If that fails, stop and tell the user to run `ms login` or set `MODELSCOPE_API_TOKEN`.
4. Let the script stage files before uploading. It uses hardlinks where possible, computes SHA256 metadata, and splits large `.pt` files before upload.
5. Upload each staged folder to `conroy1201/robotwin-ckpt` with `path_in_repo` equal to the remote top-level folder name. Use ModelScope upload cache, retry, and `--max-workers`.
6. Report the uploaded folders, ignored patterns, split counts, manifest path, and any command failures.

## Command

Use this script from the skill directory:

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py FOLDER [FOLDER ...]
```

Default behavior:

- Repository: `conroy1201/robotwin-ckpt`
- Repository type: `model`
- Remote folder name: local folder basename
- Internal structure: preserved except large split `.pt` files, which are represented as `.modelscope_split_parts/...` plus `.pt.split.json` placeholders and can be restored exactly after download
- Authentication: checked before upload
- Upload concurrency: `--max-workers` defaults to a conservative CPU-based value, capped at 8
- Retry: each folder upload is retried before failing
- Integrity: `.modelscope_upload_manifest.json` records SHA256 and size metadata

## Stable Weight Uploads

Use the default stable mode for checkpoint folders containing many weight files:

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  /ckpts/run_a /ckpts/run_b
```

Large `.pt` files are split only when they are at least `512MiB`; default part size is `256MiB`. Override this when ModelScope upload is unstable or the network is weak:

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  /ckpts/run_a \
  --split-threshold 128M \
  --chunk-size 64M \
  --max-workers 8 \
  --upload-retries 5
```

To preserve original `.pt` files in the remote tree without splitting, use:

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  /ckpts/run_a --no-split-pt
```

To prepare and inspect the staged upload tree without uploading, use:

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  /ckpts/run_a --prepare-only
```

## Ignore Rules

Use `--ignore KEY=GLOB` for per-folder ignore patterns. `KEY` can be the local folder path, the absolute folder path, or the remote top-level folder name.

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  /ckpts/run_a /ckpts/run_b \
  --ignore run_a='**/optimizer.pt' \
  --ignore run_a='wandb/**' \
  --ignore run_b='logs/**'
```

Use `--ignore-all GLOB` for a pattern that applies to every folder.

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  /ckpts/run_a /ckpts/run_b \
  --ignore-all='**/.DS_Store' \
  --ignore-all='**/__pycache__/**'
```

For complex mappings, create a temporary JSON spec and pass `--spec`:

```json
[
  {
    "path": "/ckpts/local_run_a",
    "remote": "run_a",
    "ignore": ["**/optimizer.pt", "wandb/**"]
  },
  {
    "path": "/ckpts/local_run_b",
    "ignore": ["logs/**"]
  }
]
```

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py --spec /tmp/modelscope_upload.json
```

## Useful Options

- `--dry-run`: show resolved folders, remote paths, ignored patterns, split candidates, and sizes without staging or uploading.
- `--prepare-only`: create the staged upload tree and manifest without uploading.
- `--remote-name LOCAL=REMOTE`: override the remote top-level folder name for a local folder.
- `--repo-id OWNER/NAME`: override the default target repository only if the user explicitly asks.
- `--commit-message MESSAGE`: set a custom commit message.
- `--max-workers N`: pass parallel upload worker count to ModelScope CLI.
- `--stage-workers N`: parallelize local hardlink/copy/split/SHA preparation.
- `--split-threshold SIZE`: split `.pt` files at or above this size, for example `128M`.
- `--chunk-size SIZE`: size of each uploaded part for split `.pt` files.
- `--split-extension EXT`: also split another extension such as `.pth`.
- `--no-split-pt`: upload `.pt` files whole.
- `--no-sha-manifest`: skip SHA256 entries for regular files; split `.pt` parts still have SHA256.
- `--upload-retries N`: retry failed ModelScope upload attempts.
- `--staging-dir PATH`: choose where temporary staged files are created.
- `--keep-staging`: keep staged files for inspection after upload.
- `--endpoint URL`: use a specific ModelScope endpoint.
- `--cli ms|modelscope|/path/to/cli`: force the CLI executable.

## Download Restore And Verify

After downloading a remote folder that contains split `.pt` files, run the restore helper in the downloaded top-level folder:

```bash
python modelscope-ckpt-upload/scripts/restore_modelscope_ckpt_parts.py \
  /downloaded/run_a
```

The helper verifies all split parts against the manifest, reconstructs each original `.pt`, and checks the final SHA256. To verify without writing restored files:

```bash
python modelscope-ckpt-upload/scripts/restore_modelscope_ckpt_parts.py \
  /downloaded/run_a --verify-only
```

## Examples

Upload two checkpoint folders as sibling remote folders:

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  outputs/ckpt_epoch_10 outputs/ckpt_epoch_20
```

Upload one folder under a chosen remote name:

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  /mnt/ckpts/latest --remote-name /mnt/ckpts/latest=robotwin_latest
```

Preview before uploading:

```bash
python modelscope-ckpt-upload/scripts/upload_modelscope_ckpt.py \
  /mnt/ckpts/latest --ignore latest='tmp/**' --dry-run
```
