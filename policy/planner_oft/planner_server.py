from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import _apply_runtime_model_path_overrides, build_framework


def load_planner_framework(checkpoint: str):
    checkpoint_path = Path(checkpoint)
    run_dir = checkpoint_path.parents[1]
    config_path = run_dir / "config.full.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Planner config does not exist under {run_dir}")

    config = OmegaConf.load(config_path)
    config.trainer.pretrained_checkpoint = None
    _apply_runtime_model_path_overrides(config)
    framework = build_framework(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    framework.load_state_dict(state_dict, strict=True)
    return framework


class PlannerOFTServerWrapper:
    def __init__(self, checkpoint: str, device: str = "cuda", use_bf16: bool = True) -> None:
        framework = load_planner_framework(checkpoint)
        if not hasattr(framework, "qwen_vl_interface"):
            raise TypeError(f"Planner checkpoint has no Qwen-VL interface: {type(framework).__name__}")
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        self.framework = framework.to(device).eval()
        self.interface = self.framework.qwen_vl_interface
        self.processor = self.interface.processor
        self.device = device

        datasets = getattr(self.framework.config, "datasets", None)
        vlm_data = getattr(datasets, "vlm_data", None) if datasets is not None else None
        self.prompt_template = getattr(vlm_data, "CoT_prompt", "{instruction}")
        history = getattr(vlm_data, "history", None) if vlm_data is not None else None
        self.max_history_frames = int(getattr(history, "max_frames", 12) or 12)
        self.metadata = {
            "env": "planner_oft_server",
            "ckpt_path": str(checkpoint),
            "max_history_frames": self.max_history_frames,
        }

    @staticmethod
    def _images(example: dict[str, Any]) -> list[Any]:
        images = example.get("image")
        if images is None:
            raise KeyError("Planner example is missing 'image'")
        if not isinstance(images, list):
            images = [images]
        if not images:
            raise ValueError("Planner example contains no images")
        return [to_pil_preserve(np.asarray(image)) for image in images]

    @torch.inference_mode()
    def predict_action(
        self,
        examples: list[dict[str, Any]] | dict[str, Any],
        max_new_tokens: int = 192,
        **_: Any,
    ) -> dict[str, Any]:
        if not isinstance(examples, list):
            examples = [examples]
        if len(examples) != 1:
            raise ValueError(f"Planner server currently expects one example, got {len(examples)}")

        example = examples[0]
        images = self._images(example)
        instruction = str(example.get("lang", example.get("task_lang", ""))).strip()
        if not instruction:
            raise ValueError("Planner example has no task instruction")
        prompt = str(self.prompt_template).replace("{instruction}", instruction)
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [[{"role": "user", "content": content}]]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.interface.model.device)
        prompt_length = int(inputs["input_ids"].shape[1])
        generated = self.interface.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max(int(max_new_tokens), 1),
        )
        texts = self.processor.tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {"planner_text": texts, "num_input_frames": len(images)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--idle-timeout", type=int, default=-1)
    parser.add_argument("--no-bf16", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    wrapper = PlannerOFTServerWrapper(
        checkpoint=args.ckpt_path,
        use_bf16=not args.no_bf16,
    )
    hostname = socket.gethostname()
    logging.info("Starting planner server host=%s port=%s", hostname, args.port)
    server = WebsocketPolicyServer(
        policy=wrapper,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
