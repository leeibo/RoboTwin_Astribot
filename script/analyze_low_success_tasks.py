import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from script.collect_data import apply_task_table_config, get_embodiment_config
from envs._GLOBAL_CONFIGS import CONFIGS_PATH


def _load_args(task_name, task_config, env):
    config_path = Path("task_config") / f"{task_config}.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args["task_name"] = task_name
    args = apply_task_table_config(env, args)

    embodiment_type = args["embodiment"]
    embodiment_types = yaml.safe_load(Path(CONFIGS_PATH, "_embodiment_config.yml").read_text(encoding="utf-8"))

    def embodiment_file(name):
        return embodiment_types[name]["file_path"]

    if len(embodiment_type) != 1:
        raise ValueError("This diagnostic currently expects one dual-arm embodiment entry")

    args["left_robot_file"] = embodiment_file(embodiment_type[0])
    args["right_robot_file"] = embodiment_file(embodiment_type[0])
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = str(embodiment_type[0])
    args["task_config"] = task_config
    args["save_path"] = "debug_data/low_success_diagnostics"
    args["need_plan"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    return args


def _task_class(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)


def run_seed(task_name, task_config, seed, verbose_failure=False):
    cls = _task_class(task_name)
    env = cls()
    args = _load_args(task_name, task_config, env)
    result = {
        "task": task_name,
        "seed": int(seed),
        "ok": False,
        "plan_success": False,
        "check_success": False,
        "exception_type": None,
        "exception_message": None,
        "info": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=int(seed), **args)
        info = env.play_once()
        result["plan_success"] = bool(getattr(env, "plan_success", False))
        result["check_success"] = bool(env.check_success()) if result["plan_success"] else False
        result["ok"] = bool(result["plan_success"] and result["check_success"])
        result["info"] = info.get("info", None) if isinstance(info, dict) else None
    except Exception as exc:  # noqa: BLE001 - diagnostic script must record all failure modes.
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        if verbose_failure:
            traceback.print_exc()
    finally:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            if verbose_failure:
                traceback.print_exc()
    return result


def main():
    parser = argparse.ArgumentParser(description="Sweep seeds for DYJ low-success task diagnostics.")
    parser.add_argument("--config", default="demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=20, help="exclusive upper seed bound")
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--verbose-failure", action="store_true")
    args = parser.parse_args()

    for task_name in args.tasks:
        first_success = None
        attempts = 0
        for seed in range(args.start, args.end):
            result = run_seed(task_name, args.config, seed, verbose_failure=args.verbose_failure)
            attempts += 1
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["ok"]:
                first_success = seed
                if args.stop_on_success:
                    break
        summary = {
            "task": task_name,
            "start": args.start,
            "end": args.end,
            "attempts": attempts,
            "first_success": first_success,
        }
        print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # Keep renderer sanity and post-processing out of diagnostic sweeps unless the caller opts in elsewhere.
    os.environ.setdefault("ROBOTWIN_SKIP_RENDER_TEST", "1")
    os.environ.setdefault("ROBOTWIN_SKIP_ANNOTATED_VIDEO", "1")
    os.environ.setdefault("ROBOTWIN_SKIP_INSTRUCTIONS", "1")
    main()
