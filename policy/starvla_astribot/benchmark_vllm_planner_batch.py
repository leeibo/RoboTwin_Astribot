from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from subtask_planner import (  # noqa: E402
    DEFAULT_QWEN3_30B_PATH,
    PROMPT_VERSION,
    _plain_chat_messages,
    apply_candidate_distinct_guard,
    build_subtask_equivalence_messages,
    parse_subtask_decision,
)
from test_subtask_planner_prompt import (  # noqa: E402
    PairCase,
    build_full_different_cases,
    build_pair_cases,
)


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            sizes.append(max(1, int(item)))
    if not sizes:
        raise ValueError("--batch-sizes cannot be empty")
    return sizes


def _build_cases(args: argparse.Namespace) -> list[PairCase]:
    dataset_root = Path(args.dataset_root)
    max_pairs = None if int(args.max_pairs) <= 0 else int(args.max_pairs)
    if args.case_mode == "full_different":
        return build_full_different_cases(
            dataset_root,
            task_limit=args.task_limit,
            max_pairs=max_pairs,
        )
    return build_pair_cases(
        dataset_root,
        max_pairs=max_pairs or 64,
        seed=int(args.seed),
        task_limit=args.task_limit,
    )


def _build_prompts(cases: list[PairCase], processor: Any) -> list[str]:
    prompts = []
    for case in cases:
        messages = build_subtask_equivalence_messages(
            case.previous_subtask,
            case.current_subtask,
            task_instruction=case.task_instruction,
            candidate_subtasks=case.candidate_subtasks,
        )
        prompts.append(
            processor.apply_chat_template(
                _plain_chat_messages(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def _run_batch_size(llm: Any, sampling_params: Any, cases: list[PairCase], prompts: list[str], batch_size: int) -> dict[str, Any]:
    rows = []
    correct = 0
    parse_errors = 0
    start = time.perf_counter()
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_cases = cases[batch_start : batch_start + batch_size]
        batch_t0 = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        batch_elapsed = time.perf_counter() - batch_t0
        per_item_latency = batch_elapsed / max(len(outputs), 1)
        for case, output in zip(batch_cases, outputs):
            text = output.outputs[0].text if output.outputs else ""
            prediction_same = None
            decision_dict = None
            error = None
            try:
                decision = parse_subtask_decision(text, latency_sec=per_item_latency)
                decision = apply_candidate_distinct_guard(
                    decision,
                    case.previous_subtask,
                    case.current_subtask,
                    case.candidate_subtasks,
                )
                prediction_same = bool(decision.same)
                decision_dict = decision.to_dict()
            except Exception as exc:
                parse_errors += 1
                error = repr(exc)
            ok = prediction_same is not None and prediction_same == bool(case.label_same)
            correct += int(ok)
            rows.append(
                {
                    **asdict(case),
                    "prediction_same": prediction_same,
                    "correct": bool(ok),
                    "decision": decision_dict,
                    "raw_text": text,
                    "error": error,
                }
            )
    elapsed = time.perf_counter() - start
    total = len(prompts)
    return {
        "batch_size": int(batch_size),
        "num_prompts": total,
        "elapsed_sec": elapsed,
        "sec_per_prompt": elapsed / max(total, 1),
        "prompts_per_sec": total / max(elapsed, 1e-9),
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "parse_errors": parse_errors,
        "rows": rows,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/data/lmz/code/starVLA-A/playground/dataset/RoboTwin_Astribot_lerobot",
    )
    parser.add_argument("--model-path", default=DEFAULT_QWEN3_30B_PATH)
    parser.add_argument("--case-mode", default="sampled", choices=["sampled", "full_different"])
    parser.add_argument("--task-limit", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-sizes", default="1,4,8,16,32")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compile", action="store_true", help="Allow torch.compile instead of enforce_eager.")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--output",
        default="/data/lmz/code/RoboTwin_Astribot/policy/starvla_astribot/logs/vllm_batch_benchmark.jsonl",
    )
    parser.add_argument("--summary-output", default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    cases = _build_cases(args)
    if not cases:
        raise RuntimeError(f"No benchmark cases found under {args.dataset_root}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = _build_prompts(cases, processor)
    print(f"prepared_prompts={len(prompts)} batch_sizes={batch_sizes}", flush=True)

    load_start = time.perf_counter()
    llm = LLM(
        model=str(args.model_path),
        dtype=str(args.dtype),
        trust_remote_code=True,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enforce_eager=not bool(args.compile),
        disable_log_stats=True,
    )
    load_sec = time.perf_counter() - load_start
    print(f"load_sec={load_sec:.3f}", flush=True)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=int(args.max_new_tokens))
    if not args.skip_warmup:
        warmup_start = time.perf_counter()
        llm.generate(prompts[:1], sampling_params, use_tqdm=False)
        print(f"warmup_sec={time.perf_counter() - warmup_start:.3f}", flush=True)

    summaries = []
    with output_path.open("w", encoding="utf-8") as f:
        for batch_size in batch_sizes:
            result = _run_batch_size(llm, sampling_params, cases, prompts, batch_size)
            summary = {k: v for k, v in result.items() if k != "rows"}
            summaries.append(summary)
            for row in result["rows"]:
                f.write(json.dumps({"batch_size": batch_size, **row}, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    final_summary = {
        "prompt_version": PROMPT_VERSION,
        "model_path": str(args.model_path),
        "case_mode": args.case_mode,
        "num_cases": len(cases),
        "load_sec": load_sec,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "enforce_eager": not bool(args.compile),
        "batch_results": summaries,
        "output": str(output_path),
        "summary_output": str(summary_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
