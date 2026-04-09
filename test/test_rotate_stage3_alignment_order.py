import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WHITELIST_PATH = REPO_ROOT / "task_config" / "rotate_task_whitelist.yml"

ENTER_THEN_FACE_PATTERN = re.compile(
    r"enter_rotate_action_stage\([^\n]*\)\n\s*self\.face_(?:object|world_point)_with_torso\(",
    re.MULTILINE,
)
FACE_WORLD_THEN_ENTER_PATTERN = re.compile(
    r"self\.face_world_point_with_torso\([^\n]*\)\n\s*self\.enter_rotate_action_stage\(",
    re.MULTILINE,
)


def _load_whitelist_tasks():
    tasks = []
    for raw_line in WHITELIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            tasks.append(line[2:].strip())
    return tasks


def test_whitelist_rotate_tasks_do_not_align_torso_immediately_after_entering_stage3():
    tasks = _load_whitelist_tasks()
    assert tasks, "rotate task whitelist is empty"

    offenders = []
    for task_name in tasks:
        task_path = REPO_ROOT / "envs" / f"{task_name}.py"
        source = task_path.read_text(encoding="utf-8")
        if ENTER_THEN_FACE_PATTERN.search(source):
            offenders.append(task_name)

    assert not offenders, (
        "These whitelist rotate-view tasks still align torso immediately after "
        f"`enter_rotate_action_stage(...)`: {offenders}"
    )


def test_whitelist_rotate_tasks_do_not_align_torso_immediately_before_entering_stage3():
    tasks = _load_whitelist_tasks()
    assert tasks, "rotate task whitelist is empty"

    offenders = []
    for task_name in tasks:
        task_path = REPO_ROOT / "envs" / f"{task_name}.py"
        source = task_path.read_text(encoding="utf-8")
        if FACE_WORLD_THEN_ENTER_PATTERN.search(source):
            offenders.append(task_name)

    assert not offenders, (
        "These whitelist rotate-view tasks still align torso immediately before "
        f"`enter_rotate_action_stage(...)`: {offenders}"
    )
