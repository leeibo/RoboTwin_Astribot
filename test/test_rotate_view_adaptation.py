from pathlib import Path
import ast


def _load_ast(path: Path):
    source = path.read_text(encoding="utf-8")
    return ast.parse(source), source


def _find_class(module_ast: ast.Module, class_name: str):
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def test_adjust_bottle_rotate_view_structure():
    path = Path("envs/adjust_bottle_rotate_view.py")
    assert path.exists(), "rotate-view module file is missing"

    module_ast, source = _load_ast(path)
    class_node = _find_class(module_ast, "adjust_bottle_rotate_view")
    assert class_node is not None, "rotate-view class is missing"

    base_names = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)
    assert "adjust_bottle" in base_names, "rotate-view class must inherit original task class"

    method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
    assert "setup_demo" in method_names, "rotate-view class must override setup_demo"
    assert "play_once" in method_names, "rotate-view class must override play_once"

    assert '"table_shape", "fan"' in source
    assert '"astribot_torso_joint_2"' in source
