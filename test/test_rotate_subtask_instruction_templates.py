import importlib.util
import json
import random
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "description" / "utils" / "instruction_template_utils.py"
UTILS_SPEC = importlib.util.spec_from_file_location("instruction_template_utils_module", UTILS_PATH)
instruction_utils = importlib.util.module_from_spec(UTILS_SPEC)
assert UTILS_SPEC.loader is not None
UTILS_SPEC.loader.exec_module(instruction_utils)


def _load_task_payload(task_name):
    json_path = REPO_ROOT / "description" / "task_instruction" / f"{task_name}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_whitelist_rotate_tasks_define_subtask_instruction_template_map():
    whitelist_path = REPO_ROOT / "task_config" / "rotate_task_whitelist.yml"
    with open(whitelist_path, "r", encoding="utf-8") as f:
        whitelist = yaml.safe_load(f)

    for task_name in whitelist:
        payload = _load_task_payload(task_name)
        assert "subtask_instruction_template_map" in payload, task_name
        assert isinstance(payload["subtask_instruction_template_map"], dict), task_name
        assert len(payload["subtask_instruction_template_map"]) > 0, task_name


def test_resolve_task_instruction_bank_returns_concrete_episode_instruction():
    payload = _load_task_payload("place_burger_fries_rotate_view")
    instruction = instruction_utils.resolve_instruction_bank(
        {
            "seen": payload["seen"],
            "unseen": payload["unseen"],
        },
        {
            "{A}": "006_hamburg/base3",
            "{B}": "008_tray/base3",
            "{C}": "005_french-fries/base0",
        },
        max_descriptions=1,
        rng=random.Random(0),
    )[0]

    assert "{" not in instruction
    assert "}" not in instruction
    assert "the " in instruction.lower()


def test_resolve_subtask_instruction_templates_uses_same_placeholder_pipeline():
    payload = _load_task_payload("place_burger_fries_rotate_view")
    template_map = payload["subtask_instruction_template_map"]
    episode_info = {
        "{A}": "006_hamburg/base3",
        "{B}": "008_tray/base3",
        "{C}": "005_french-fries/base0",
    }

    resolved = {}
    for instruction_idx, template_bank in template_map.items():
        resolved[instruction_idx] = instruction_utils.resolve_instruction_bank(
            template_bank,
            episode_info,
            max_descriptions=1,
            rng=random.Random(f"burger:{instruction_idx}"),
        )[0]

    assert resolved["1"] != template_map["1"]["seen"][0]
    assert resolved["2"] != template_map["2"]["seen"][0]
    assert all("{" not in text and "}" not in text for text in resolved.values())


def test_color_placeholder_subtask_instruction_resolves_without_object_json():
    payload = _load_task_payload("place_fan_rotate_view")
    instruction = instruction_utils.resolve_instruction_bank(
        payload["subtask_instruction_template_map"]["2"],
        {
            "{A}": "099_fan/base4",
            "{B}": "Blue",
            "{a}": "left",
        },
        max_descriptions=1,
        rng=random.Random(0),
    )[0]

    assert "{" not in instruction
    assert "Blue" in instruction
    assert "mat" in instruction.lower()
