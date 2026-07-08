from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from subtask_planner import (
    DEFAULT_QWEN3_30B_PATH,
    PROMPT_VERSION,
    SubtaskPlannerError,
    build_subtask_equivalence_prompt,
    build_subtask_planner,
)


class PlannerHandler(BaseHTTPRequestHandler):
    planner = None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"ok": True, "prompt_version": PROMPT_VERSION})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in {"/classify", "/"}:
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            previous_subtask = str(payload["previous_subtask"])
            current_subtask = str(payload["current_subtask"])
            task_instruction = payload.get("task_instruction")
            candidate_subtasks = payload.get("candidate_subtasks") or []
            decision = self.planner.compare(
                previous_subtask,
                current_subtask,
                task_instruction=task_instruction,
                candidate_subtasks=candidate_subtasks,
            )
            self._send_json(
                200,
                {
                    "ok": True,
                    "prompt_version": PROMPT_VERSION,
                    "decision": decision.to_dict(),
                },
            )
        except (KeyError, json.JSONDecodeError, SubtaskPlannerError, ValueError) as exc:
            self._send_json(400, {"ok": False, "error": repr(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": repr(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[subtask_planner_server] {self.address_string()} {fmt % args}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7991)
    parser.add_argument("--model-path", default=DEFAULT_QWEN3_30B_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--print-prompt", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.print_prompt:
        print(
            build_subtask_equivalence_prompt(
                "find apple and pick it up",
                "locate apple and grasp it",
                task_instruction="Pick up the apple",
                candidate_subtasks=["find apple and pick it up"],
            ),
            flush=True,
        )
    PlannerHandler.planner = build_subtask_planner(
        args.backend,
        model_path=args.model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    server = ThreadingHTTPServer((args.host, args.port), PlannerHandler)
    print(
        f"[subtask_planner_server] listening on http://{args.host}:{args.port}/classify "
        f"model={args.model_path}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
