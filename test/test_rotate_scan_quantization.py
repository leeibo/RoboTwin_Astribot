from pathlib import Path
import ast
import importlib.util
import numpy as np


def _load_rotate_theta_module():
    path = Path("envs/utils/rotate_theta.py")
    spec = importlib.util.spec_from_file_location("rotate_theta_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quantize_theta_to_unit_outward():
    rotate_theta = _load_rotate_theta_module()
    unit = np.deg2rad(15.0)

    snapped = rotate_theta.quantize_theta_to_unit(
        0.95,
        unit_rad=unit,
        mode="outward",
        min_steps=1,
        max_abs_rad=np.pi,
    )
    assert np.isclose(snapped, np.deg2rad(60.0)), "0.95 rad should snap outward to 60 degrees"

    snapped_small = rotate_theta.quantize_theta_to_unit(
        -0.1,
        unit_rad=unit,
        mode="outward",
        min_steps=1,
        max_abs_rad=np.pi,
    )
    assert np.isclose(snapped_small, -unit), "small non-zero scan angles should snap to one unit step"


def test_quantize_scan_thetas_for_task_orders_large_swings_first():
    rotate_theta = _load_rotate_theta_module()

    class DummyTask:
        rotate_scan_theta_unit_rad = np.deg2rad(15.0)
        rotate_scan_quantize_mode = "outward"
        rotate_scan_min_steps = 1
        rotate_scan_large_swing_first = True
        rotate_object_theta_half_rad = np.pi

    snapped = rotate_theta.quantize_scan_thetas_for_task(DummyTask(), [0.26, -0.78, 0.52])
    expected = [np.deg2rad(-45.0), np.deg2rad(30.0), np.deg2rad(15.0)]
    assert len(snapped) == len(expected)
    assert np.allclose(snapped, expected), "scan order should prefer larger discrete swings first"


def test_build_scan_theta_search_sequence_for_task():
    rotate_theta = _load_rotate_theta_module()

    class DummyTask:
        rotate_scan_theta_unit_rad = np.deg2rad(15.0)
        rotate_object_theta_half_rad = np.deg2rad(70.0)
        rotate_scan_sequence_steps = (4, -4, 2, -2, 0)

    sequence = rotate_theta.build_scan_theta_search_sequence_for_task(DummyTask())
    expected = [
        np.deg2rad(60.0),
        np.deg2rad(-60.0),
        np.deg2rad(30.0),
        np.deg2rad(-30.0),
        0.0,
    ]
    assert np.allclose(sequence, expected), "default coarse search should follow the configured large-swing sequence"


def test_sort_scan_thetas_left_to_right():
    rotate_theta = _load_rotate_theta_module()
    ordered = rotate_theta.sort_scan_thetas([np.deg2rad(-30.0), np.deg2rad(45.0), 0.0], order="left_to_right")
    expected = [np.deg2rad(45.0), 0.0, np.deg2rad(-30.0)]
    assert np.allclose(ordered, expected), "left_to_right ordering should scan positive theta to negative theta"


def test_default_scan_strategy_is_object_coverage():
    rotate_theta = _load_rotate_theta_module()
    assert rotate_theta.DEFAULT_SCAN_STRATEGY == "object_coverage"


def test_base_task_scan_theta_entry_uses_coarse_search_helper():
    path = Path("envs/_base_task.py")
    module_ast = ast.parse(path.read_text(encoding="utf-8"))

    target = None
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == "Base_Task":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_get_scan_thetas_from_object_list":
                    target = item
                    break
    assert target is not None, "Base_Task._get_scan_thetas_from_object_list must exist"

    calls = [n for n in ast.walk(target) if isinstance(n, ast.Call)]
    called_names = []
    for call in calls:
        if isinstance(call.func, ast.Name):
            called_names.append(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            called_names.append(call.func.attr)
    assert "build_scan_theta_search_sequence_for_task" in called_names, (
        "scan theta selection must support the shared coarse-search sequence"
    )
