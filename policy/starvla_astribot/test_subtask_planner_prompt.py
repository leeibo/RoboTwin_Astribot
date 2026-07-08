from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from subtask_planner import (  # noqa: E402
    DEFAULT_QWEN3_30B_PATH,
    PROMPT_VERSION,
    SubtaskPlannerError,
    build_subtask_equivalence_prompt,
    build_subtask_planner,
)


@dataclass
class PairCase:
    task_name: str
    episode_index: int
    task_instruction: str
    previous_subtask: str
    current_subtask: str
    label_same: bool
    case_type: str
    candidate_subtasks: list[str]
    previous_subtask_idx: int | None = None
    current_subtask_idx: int | None = None


def _clean(value) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _episode_iter(dataset_root: Path, task_limit: int | None = None) -> Iterable[dict]:
    paths = sorted(dataset_root.glob("*/meta/astribot_subtask_metadata.json"))
    if task_limit:
        paths = paths[: int(task_limit)]
    for meta_path in paths:
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        for episode in metadata.get("episodes", []):
            yield episode


def _instruction_items(episode: dict) -> list[tuple[int, str]]:
    mapping = episode.get("subtask_instruction_map") or {}
    items = []
    for key, text in mapping.items():
        text = _clean(text)
        if text:
            items.append((int(key), text))
    return sorted(items)


def _paraphrases(text: str) -> list[str]:
    variants = set()
    replacements = [
        ("find ", "locate "),
        ("find ", "search for "),
        ("pick it up", "grasp it"),
        ("pick it up", "grab it"),
        ("pick it up", "take it"),
        ("press its top center", "push its top center"),
        ("press its top center", "tap its top center"),
        ("place ", "put "),
        ("drop ", "place "),
        ("carefully ", ""),
    ]
    for old, new in replacements:
        if old in text:
            variants.add(text.replace(old, new))
    match = re.match(r"find (.*?) and pick it up$", text)
    if match:
        obj = match.group(1)
        variants.add(f"grasp {obj}")
        variants.add(f"pick up {obj}")
    match = re.match(r"find (.*?) and place (.*?) to its (left|right)$", text)
    if match:
        ref_obj, moving_obj, side = match.groups()
        variants.add(f"locate {ref_obj} and put {moving_obj} to its {side}")
        variants.add(f"place {moving_obj} on the {side} side of {ref_obj}")
    return sorted(v for v in variants if _clean(v) and _clean(v) != text)


def build_pair_cases(
    dataset_root: Path,
    *,
    max_pairs: int,
    seed: int,
    task_limit: int | None = None,
) -> list[PairCase]:
    rng = random.Random(seed)
    positive: list[PairCase] = []
    negative: list[PairCase] = []

    for episode in _episode_iter(dataset_root, task_limit=task_limit):
        items = _instruction_items(episode)
        if not items:
            continue
        task_name = _clean(episode.get("task_name"))
        episode_index = int(episode.get("lerobot_episode_index", episode.get("episode_index", -1)))
        task_instruction = _clean(episode.get("task_instruction"))
        candidates = [text for _, text in items]

        for _, text in items:
            paras = _paraphrases(text)
            if paras:
                current = rng.choice(paras)
            else:
                current = text
            positive.append(
                PairCase(
                    task_name=task_name,
                    episode_index=episode_index,
                    task_instruction=task_instruction,
                    previous_subtask=text,
                    current_subtask=current,
                    label_same=True,
                    case_type="same_paraphrase",
                    candidate_subtasks=candidates,
                )
            )

        for left_idx in range(len(items)):
            for right_idx in range(left_idx + 1, len(items)):
                _, prev = items[left_idx]
                _, cur = items[right_idx]
                negative.append(
                    PairCase(
                        task_name=task_name,
                        episode_index=episode_index,
                        task_instruction=task_instruction,
                        previous_subtask=prev,
                        current_subtask=cur,
                        label_same=False,
                        case_type="different_training_subtask",
                        candidate_subtasks=candidates,
                    )
                )

    rng.shuffle(positive)
    rng.shuffle(negative)
    half = max_pairs // 2
    selected = positive[:half] + negative[: max_pairs - half]
    rng.shuffle(selected)
    return selected


def build_full_different_cases(
    dataset_root: Path,
    *,
    task_limit: int | None = None,
    max_pairs: int | None = None,
) -> list[PairCase]:
    cases: list[PairCase] = []
    for episode in _episode_iter(dataset_root, task_limit=task_limit):
        items = _instruction_items(episode)
        if len(items) < 2:
            continue
        task_name = _clean(episode.get("task_name"))
        episode_index = int(episode.get("lerobot_episode_index", episode.get("episode_index", -1)))
        task_instruction = _clean(episode.get("task_instruction"))
        candidates = [text for _, text in items]

        for left_idx in range(len(items)):
            for right_idx in range(left_idx + 1, len(items)):
                prev_id, prev = items[left_idx]
                cur_id, cur = items[right_idx]
                if _clean(prev).lower() == _clean(cur).lower():
                    continue
                cases.append(
                    PairCase(
                        task_name=task_name,
                        episode_index=episode_index,
                        task_instruction=task_instruction,
                        previous_subtask=prev,
                        current_subtask=cur,
                        label_same=False,
                        case_type="full_episode_different_subtask",
                        candidate_subtasks=candidates,
                        previous_subtask_idx=prev_id,
                        current_subtask_idx=cur_id,
                    )
                )
                if max_pairs is not None and len(cases) >= int(max_pairs):
                    return cases
    return cases


def _case_key(case: PairCase) -> tuple:
    return (
        case.task_name,
        int(case.episode_index),
        case.previous_subtask_idx,
        case.current_subtask_idx,
        case.previous_subtask,
        case.current_subtask,
        case.case_type,
    )


def _row_key(row: dict) -> tuple:
    return (
        row.get("task_name"),
        int(row.get("episode_index", -1)),
        row.get("previous_subtask_idx"),
        row.get("current_subtask_idx"),
        row.get("previous_subtask"),
        row.get("current_subtask"),
        row.get("case_type"),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/data/lmz/code/starVLA-A/playground/dataset/RoboTwin_Astribot_lerobot",
    )
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm", "http", "prompt_only"])
    parser.add_argument("--case-mode", default="sampled", choices=["sampled", "full_different"])
    parser.add_argument("--model-path", default=DEFAULT_QWEN3_30B_PATH)
    parser.add_argument("--url", default="http://127.0.0.1:7991/classify")
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--task-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--output",
        default="/data/lmz/code/RoboTwin_Astribot/policy/starvla_astribot/logs/subtask_planner_prompt_eval.jsonl",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-output", default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    max_pairs = None if int(args.max_pairs) <= 0 else int(args.max_pairs)
    if args.case_mode == "full_different":
        cases = build_full_different_cases(
            Path(args.dataset_root),
            max_pairs=max_pairs,
            task_limit=args.task_limit,
        )
    else:
        cases = build_pair_cases(
            Path(args.dataset_root),
            max_pairs=max_pairs or 32,
            seed=int(args.seed),
            task_limit=args.task_limit,
        )
    if not cases:
        raise RuntimeError(f"No subtask pairs found under {args.dataset_root}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "prompt_only":
        with output_path.open("w", encoding="utf-8") as f:
            for case in cases:
                prompt = build_subtask_equivalence_prompt(
                    case.previous_subtask,
                    case.current_subtask,
                    task_instruction=case.task_instruction,
                    candidate_subtasks=case.candidate_subtasks,
                )
                f.write(json.dumps({**asdict(case), "prompt": prompt}, ensure_ascii=False) + "\n")
        print(f"prompt_only wrote {len(cases)} prompts to {output_path}")
        return

    done: set[tuple] = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(_row_key(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    pending_cases = [case for case in cases if _case_key(case) not in done]

    planner = build_subtask_planner(
        args.backend,
        url=args.url,
        model_path=args.model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    correct = 0
    total_done = 0
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_done += 1
                correct += int(bool(row.get("correct", False)))
    start = time.perf_counter()
    write_mode = "a" if args.resume and output_path.exists() else "w"
    batch_size = max(1, int(args.batch_size))
    can_batch = batch_size > 1 and hasattr(planner, "compare_many")

    def write_result(f, idx: int, case: PairCase, result) -> None:
        nonlocal correct
        if isinstance(result, Exception):
            prediction_same = None
            decision_dict = None
            error = repr(result)
        else:
            prediction_same = bool(result.same)
            decision_dict = result.to_dict()
            error = None
        ok = prediction_same is not None and prediction_same == bool(case.label_same)
        correct += int(ok)
        row = {
            "idx": idx,
            **asdict(case),
            "prediction_same": prediction_same,
            "correct": bool(ok),
            "decision": decision_dict,
            "error": error,
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        print(
            f"[{idx:06d}/{len(cases):06d}] label={case.label_same} pred={prediction_same} "
            f"ok={ok} type={case.case_type} error={error}",
            flush=True,
        )

    with output_path.open(write_mode, encoding="utf-8") as f:
        if can_batch:
            for batch_start in range(0, len(pending_cases), batch_size):
                batch = pending_cases[batch_start : batch_start + batch_size]
                requests = [
                    {
                        "previous_subtask": case.previous_subtask,
                        "current_subtask": case.current_subtask,
                        "task_instruction": case.task_instruction,
                        "candidate_subtasks": case.candidate_subtasks,
                    }
                    for case in batch
                ]
                try:
                    results = planner.compare_many(requests)
                except Exception as exc:
                    results = [exc] * len(batch)
                for offset, (case, result) in enumerate(zip(batch, results), start=1):
                    idx = total_done + batch_start + offset
                    write_result(f, idx, case, result)
            pending_iter = []
        else:
            pending_iter = enumerate(pending_cases, start=1)
        for pending_idx, case in pending_iter:
            idx = total_done + pending_idx
            try:
                decision = planner.compare(
                    case.previous_subtask,
                    case.current_subtask,
                    task_instruction=case.task_instruction,
                    candidate_subtasks=case.candidate_subtasks,
                )
            except SubtaskPlannerError as exc:
                decision = exc
            write_result(f, idx, case, decision)
    elapsed = time.perf_counter() - start
    completed = total_done + len(pending_cases)
    accuracy = correct / completed if completed else 0.0
    summary = {
        "prompt_version": PROMPT_VERSION,
        "backend": args.backend,
        "case_mode": args.case_mode,
        "batch_size": batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "num_cases": len(cases),
        "num_completed": completed,
        "num_pending_before_run": len(pending_cases),
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_sec": elapsed,
        "output": str(output_path),
        "summary_output": str(summary_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
