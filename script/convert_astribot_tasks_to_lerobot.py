import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_LEROBOT_PYTHON = "/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/conda_envs/lerobot/bin/python"


def discover_tasks(data_root, tasks):
    data_root = Path(data_root)
    if tasks:
        task_dirs = [data_root / task for task in tasks]
    else:
        task_dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir()]
    missing = [str(path) for path in task_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing task directories: {missing}")
    return task_dirs


def repo_id_for_task(repo_prefix, task_dir):
    prefix = repo_prefix.rstrip("/")
    return f"{prefix}/{task_dir.name}" if prefix else task_dir.name


def convert_task(task_dir, args):
    repo_id = repo_id_for_task(args.repo_prefix, task_dir)
    cmd = [
        args.python,
        "script/convert_astribot_hdf5_to_lerobot.py",
        str(task_dir),
        "--repo-id",
        repo_id,
        "--output-root",
        args.output_root,
        "--mode",
        args.mode,
        "--decode-workers",
        str(args.decode_workers),
        "--image-writer-processes",
        str(args.image_writer_processes),
        "--image-writer-threads",
        str(args.image_writer_threads),
        "--fps",
        str(args.fps),
        "--camera",
        args.camera,
        "--robot-type",
        args.robot_type,
        "--image-size",
        str(args.image_size[0]),
        str(args.image_size[1]),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.allow_missing_keyframes:
        cmd.append("--allow-missing-keyframes")
    if args.no_consolidate:
        cmd.append("--no-consolidate")
    if args.skip_compute_stats:
        cmd.append("--skip-compute-stats")
    if args.max_episodes_per_task is not None:
        cmd.extend(["--max-episodes", str(args.max_episodes_per_task)])

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HOME"] = str(Path(args.hf_home).resolve())
    env["HF_DATASETS_CACHE"] = str(Path(args.hf_home).resolve() / "datasets")
    env["HF_LEROBOT_HOME"] = str(Path(args.output_root).resolve())
    env.pop("LEROBOT_HOME", None)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_dir.name}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=args.repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "task": task_dir.name,
        "repo_id": repo_id,
        "returncode": proc.returncode,
        "log_path": str(log_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Convert multiple Astribot tasks to separate LeRobot repos in parallel.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task directory names under --data-root. Defaults to all tasks.")
    parser.add_argument("--task-workers", type=int, default=4, help="Number of task repos to convert concurrently.")
    parser.add_argument("--python", default=DEFAULT_LEROBOT_PYTHON, help="Python executable from the isolated lerobot env.")
    parser.add_argument("--repo-root", default=".", help="Repository root used as subprocess cwd.")
    parser.add_argument("--repo-prefix", default="local", help="Prefix for per-task repo ids.")
    parser.add_argument("--output-root", default="lerobot_data")
    parser.add_argument("--log-dir", default="logs/lerobot_convert")
    parser.add_argument("--hf-home", default=".cache/huggingface")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--image-writer-processes", type=int, default=8)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--camera", default="camera_head")
    parser.add_argument("--robot-type", default="astribot")
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 480), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--allow-missing-keyframes", action="store_true")
    parser.add_argument("--no-consolidate", action="store_true")
    parser.add_argument("--skip-compute-stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    task_dirs = discover_tasks(args.data_root, args.tasks)
    print(f"Found {len(task_dirs)} task(s); task_workers={args.task_workers}")
    failures = []
    with ThreadPoolExecutor(max_workers=max(args.task_workers, 1)) as executor:
        futures = {executor.submit(convert_task, task_dir, args): task_dir for task_dir in task_dirs}
        for future in as_completed(futures):
            result = future.result()
            status = "OK" if result["returncode"] == 0 else "FAIL"
            print(f"[{status}] {result['task']} -> {result['repo_id']} log={result['log_path']}")
            if result["returncode"] != 0:
                failures.append(result)
    if failures:
        print("Failed tasks:")
        for failure in failures:
            print(f"  {failure['task']} log={failure['log_path']}")
        raise SystemExit(1)
    print("Done")


if __name__ == "__main__":
    main()
