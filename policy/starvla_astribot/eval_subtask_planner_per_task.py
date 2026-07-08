from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from subtask_planner import PROMPT_VERSION  # noqa: E402
from test_subtask_planner_prompt import PairCase, _clean, _instruction_items, _paraphrases  # noqa: E402


def _metadata_paths(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("*/meta/astribot_subtask_metadata.json"))


def _build_cases_for_task(meta_path: Path, *, per_task: int, seed: int) -> list[PairCase]:
    rng = random.Random(f"{seed}:{meta_path.parent.parent.name}")
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    positives: list[PairCase] = []
    negatives: list[PairCase] = []
    for episode in metadata.get("episodes", []):
        items = _instruction_items(episode)
        if not items:
            continue
        task_name = _clean(episode.get("task_name")) or meta_path.parent.parent.name
        episode_index = int(episode.get("lerobot_episode_index", episode.get("episode_index", -1)))
        task_instruction = _clean(episode.get("task_instruction"))
        candidates = [text for _, text in items]

        for subtask_id, text in items:
            paras = _paraphrases(text)
            current = rng.choice(paras) if paras else text
            positives.append(
                PairCase(
                    task_name=task_name,
                    episode_index=episode_index,
                    task_instruction=task_instruction,
                    previous_subtask=text,
                    current_subtask=current,
                    label_same=True,
                    case_type="same_paraphrase",
                    candidate_subtasks=candidates,
                    previous_subtask_idx=subtask_id,
                    current_subtask_idx=subtask_id,
                )
            )

        for left_idx in range(len(items)):
            for right_idx in range(left_idx + 1, len(items)):
                prev_id, prev = items[left_idx]
                cur_id, cur = items[right_idx]
                if _clean(prev).lower() == _clean(cur).lower():
                    continue
                negatives.append(
                    PairCase(
                        task_name=task_name,
                        episode_index=episode_index,
                        task_instruction=task_instruction,
                        previous_subtask=prev,
                        current_subtask=cur,
                        label_same=False,
                        case_type="different_training_subtask",
                        candidate_subtasks=candidates,
                        previous_subtask_idx=prev_id,
                        current_subtask_idx=cur_id,
                    )
                )

    rng.shuffle(positives)
    rng.shuffle(negatives)
    target_pos = per_task // 2
    target_neg = per_task - target_pos
    selected = positives[:target_pos] + negatives[:target_neg]
    if len(selected) < per_task:
        used = {(case.episode_index, case.previous_subtask_idx, case.current_subtask_idx, case.case_type) for case in selected}
        extras = [
            case
            for case in positives[target_pos:] + negatives[target_neg:]
            if (case.episode_index, case.previous_subtask_idx, case.current_subtask_idx, case.case_type) not in used
        ]
        selected.extend(extras[: per_task - len(selected)])
    rng.shuffle(selected)
    return selected


def _post(url: str, case: PairCase, timeout: float) -> tuple[float, dict[str, Any]]:
    body = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "task_instruction": case.task_instruction,
            "previous_subtask": case.previous_subtask,
            "current_subtask": case.current_subtask,
            "candidate_subtasks": case.candidate_subtasks,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return time.perf_counter() - start, data


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_sec"]) for row in rows if row.get("latency_sec") is not None]
    return {
        "num_cases": len(rows),
        "num_success": len(latencies),
        "errors": sum(1 for row in rows if row.get("error")),
        "correct": sum(1 for row in rows if row.get("correct")),
        "accuracy": sum(1 for row in rows if row.get("correct")) / max(len(rows), 1),
        "latency_avg_sec": sum(latencies) / max(len(latencies), 1),
        "latency_min_sec": min(latencies) if latencies else None,
        "latency_p50_sec": _pct(latencies, 0.5),
        "latency_p90_sec": _pct(latencies, 0.9),
        "latency_max_sec": max(latencies) if latencies else None,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/data/lmz/code/starVLA-A/playground/dataset/RoboTwin_Astribot_lerobot",
    )
    parser.add_argument("--url", default="http://127.0.0.1:7991/classify")
    parser.add_argument("--per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument(
        "--output",
        default="/data/lmz/code/RoboTwin_Astribot/policy/starvla_astribot/logs/subtask_planner_per_task_64.jsonl",
    )
    parser.add_argument("--summary-output", default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset_root = Path(args.dataset_root)
    paths = _metadata_paths(dataset_root)
    if args.task_limit:
        paths = paths[: int(args.task_limit)]
    if not paths:
        raise RuntimeError(f"No metadata files found under {dataset_root}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    task_summaries = []
    start_all = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as f:
        for task_idx, meta_path in enumerate(paths, start=1):
            task_name = meta_path.parent.parent.name
            cases = _build_cases_for_task(meta_path, per_task=int(args.per_task), seed=int(args.seed))
            task_rows = []
            task_start = time.perf_counter()
            for local_idx, case in enumerate(cases, start=1):
                latency = None
                prediction_same = None
                decision = None
                error = None
                try:
                    latency, data = _post(args.url, case, timeout=float(args.timeout))
                    if data.get("ok"):
                        decision = data.get("decision")
                        prediction_same = bool((decision or {}).get("same"))
                    else:
                        error = data.get("error") or json.dumps(data, ensure_ascii=False)
                except Exception as exc:
                    error = repr(exc)
                correct = prediction_same is not None and prediction_same == bool(case.label_same)
                row = {
                    "task_idx": task_idx,
                    "local_idx": local_idx,
                    "task_dir": task_name,
                    **asdict(case),
                    "prediction_same": prediction_same,
                    "correct": bool(correct),
                    "decision": decision,
                    "latency_sec": latency,
                    "error": error,
                }
                task_rows.append(row)
                all_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            task_summary = {
                "task_idx": task_idx,
                "task_dir": task_name,
                "elapsed_sec": time.perf_counter() - task_start,
                **_summarize(task_rows),
            }
            task_summaries.append(task_summary)
            print(json.dumps(task_summary, ensure_ascii=False), flush=True)

    final_summary = {
        "prompt_version": PROMPT_VERSION,
        "url": args.url,
        "dataset_root": str(dataset_root),
        "per_task": int(args.per_task),
        "num_tasks": len(paths),
        "elapsed_sec": time.perf_counter() - start_all,
        **_summarize(all_rows),
        "tasks": task_summaries,
        "output": str(output_path),
        "summary_output": str(summary_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
