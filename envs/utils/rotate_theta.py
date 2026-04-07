import numpy as np


DEFAULT_ROTATE_THETA_SHARED_RATIO = 1.0
DEFAULT_SIDE_INNER_RATIO = 0.55
DEFAULT_FIXED_RATIO = 0.75
DEFAULT_MIXED_NEAR_RATIO = 0.45
DEFAULT_SCAN_THETA_UNIT_DEG = 15.0
DEFAULT_SCAN_QUANTIZE_MODE = "outward"
DEFAULT_SCAN_MIN_STEPS = 1
DEFAULT_SCAN_STRATEGY = "coarse_search"
DEFAULT_SCAN_SEQUENCE_STEPS = (4, -4, 2, -2, 0)


def _normalize_theta_range(theta_lim):
    arr = np.array(theta_lim, dtype=np.float64).reshape(-1)
    if arr.shape[0] == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    if arr.shape[0] == 1:
        return np.array([arr[0], arr[0]], dtype=np.float64)
    lo, hi = float(arr[0]), float(arr[1])
    if hi < lo:
        lo, hi = hi, lo
    return np.array([lo, hi], dtype=np.float64)


def init_rotate_theta_bounds(
    task,
    kwargs,
    default_fan_angle_deg=220.0,
    default_object_margin_deg=10.0,
    default_reference_fan_angle_deg=220.0,
    min_object_half_deg=5.0,
    default_scan_theta_unit_deg=DEFAULT_SCAN_THETA_UNIT_DEG,
):
    """
    Initialize per-task theta adaptation states for rotate-view tasks.

    Runtime config keys:
      - fan_angle_deg
      - rotate_object_margin_deg
      - rotate_theta_reference_fan_angle_deg
      - rotate_min_object_half_deg
      - rotate_theta_shared_ratio
      - rotate_scan_theta_unit_deg
      - rotate_scan_quantize_mode
      - rotate_scan_min_steps
      - rotate_scan_large_swing_first
    """
    fan_angle_deg = float(kwargs.get("fan_angle_deg", default_fan_angle_deg))
    object_margin_deg = float(kwargs.get("rotate_object_margin_deg", default_object_margin_deg))
    reference_fan_angle_deg = float(
        kwargs.get("rotate_theta_reference_fan_angle_deg", default_reference_fan_angle_deg)
    )
    min_half_deg = float(kwargs.get("rotate_min_object_half_deg", min_object_half_deg))

    table_half_rad = np.deg2rad(max(fan_angle_deg * 0.5, 0.0))
    object_half_rad = max(table_half_rad - np.deg2rad(max(object_margin_deg, 0.0)), np.deg2rad(max(min_half_deg, 0.0)))
    ref_object_half_rad = max(
        np.deg2rad(max(reference_fan_angle_deg * 0.5 - object_margin_deg, min_half_deg)),
        1e-6,
    )

    task.rotate_table_theta_half_rad = float(table_half_rad)
    task.rotate_object_theta_half_rad = float(object_half_rad)
    task.rotate_theta_scale = float(object_half_rad / ref_object_half_rad)
    task.rotate_fan_angle_deg = float(fan_angle_deg)
    task.rotate_object_margin_deg = float(object_margin_deg)
    task.rotate_theta_reference_fan_angle_deg = float(reference_fan_angle_deg)
    shared_ratio = float(kwargs.get("rotate_theta_shared_ratio", DEFAULT_ROTATE_THETA_SHARED_RATIO))
    shared_ratio = float(np.clip(shared_ratio, 0.0, 1.0))
    task.rotate_theta_shared_ratio = shared_ratio
    task.rotate_theta_shared_half_rad = float(task.rotate_object_theta_half_rad * shared_ratio)
    scan_theta_unit_deg = max(float(kwargs.get("rotate_scan_theta_unit_deg", default_scan_theta_unit_deg)), 0.0)
    task.rotate_scan_theta_unit_deg = float(scan_theta_unit_deg)
    task.rotate_scan_theta_unit_rad = float(np.deg2rad(scan_theta_unit_deg))
    task.rotate_scan_quantize_mode = str(kwargs.get("rotate_scan_quantize_mode", DEFAULT_SCAN_QUANTIZE_MODE))
    task.rotate_scan_min_steps = max(int(kwargs.get("rotate_scan_min_steps", DEFAULT_SCAN_MIN_STEPS)), 0)
    task.rotate_scan_large_swing_first = bool(kwargs.get("rotate_scan_large_swing_first", True))
    task.rotate_scan_strategy = str(kwargs.get("rotate_scan_strategy", DEFAULT_SCAN_STRATEGY)).lower()
    scan_sequence_steps = kwargs.get("rotate_scan_sequence_steps", DEFAULT_SCAN_SEQUENCE_STEPS)
    if isinstance(scan_sequence_steps, str):
        scan_sequence_steps = [item.strip() for item in scan_sequence_steps.split(",") if item.strip()]
    parsed_steps = []
    for step in np.array(scan_sequence_steps, dtype=np.float64).reshape(-1).tolist():
        parsed_steps.append(int(np.round(step)))
    if len(parsed_steps) == 0:
        parsed_steps = list(DEFAULT_SCAN_SEQUENCE_STEPS)
    task.rotate_scan_sequence_steps = tuple(parsed_steps)

    return kwargs


def quantize_theta_to_unit(
    theta_rad,
    unit_rad,
    mode: str = DEFAULT_SCAN_QUANTIZE_MODE,
    min_steps: int = 0,
    max_abs_rad=None,
):
    theta = float(theta_rad)
    unit = max(float(unit_rad), 0.0)
    if unit <= 1e-9:
        if max_abs_rad is None:
            return theta
        return float(np.clip(theta, -float(max_abs_rad), float(max_abs_rad)))

    abs_theta = abs(theta)
    raw_steps = abs_theta / unit
    mode = str(mode).lower()
    if mode == "nearest":
        steps = int(np.round(raw_steps))
    elif mode == "inward":
        steps = int(np.floor(raw_steps + 1e-9))
    else:
        steps = int(np.ceil(raw_steps - 1e-9))

    if abs_theta > 1e-9:
        steps = max(steps, int(min_steps))

    if max_abs_rad is not None:
        max_steps = int(np.floor(max(float(max_abs_rad), 0.0) / unit + 1e-9))
        steps = min(steps, max_steps)

    if steps <= 0:
        return 0.0
    return float(np.copysign(steps * unit, theta))


def quantize_scan_thetas_for_task(task, theta_list):
    thetas = np.array(theta_list, dtype=np.float64).reshape(-1)
    if thetas.shape[0] == 0:
        return []

    unit_rad = float(getattr(task, "rotate_scan_theta_unit_rad", 0.0))
    mode = str(getattr(task, "rotate_scan_quantize_mode", DEFAULT_SCAN_QUANTIZE_MODE))
    min_steps = int(getattr(task, "rotate_scan_min_steps", DEFAULT_SCAN_MIN_STEPS))
    max_abs_rad = float(getattr(task, "rotate_object_theta_half_rad", np.pi))

    quantized = []
    for theta in thetas.tolist():
        snapped = quantize_theta_to_unit(
            theta,
            unit_rad=unit_rad,
            mode=mode,
            min_steps=min_steps,
            max_abs_rad=max_abs_rad,
        )
        if not any(abs(snapped - existing) < 1e-6 for existing in quantized):
            quantized.append(snapped)

    if bool(getattr(task, "rotate_scan_large_swing_first", True)):
        quantized.sort(key=lambda val: (-abs(val), -val))
    return quantized


def build_scan_theta_search_sequence_for_task(task):
    unit_rad = float(getattr(task, "rotate_scan_theta_unit_rad", 0.0))
    max_abs_rad = float(getattr(task, "rotate_object_theta_half_rad", np.pi))
    if unit_rad <= 1e-9:
        return [0.0]

    max_step = int(np.floor(max(max_abs_rad, 0.0) / unit_rad + 1e-9))
    if max_step <= 0:
        return [0.0]

    sequence = []
    for raw_step in getattr(task, "rotate_scan_sequence_steps", DEFAULT_SCAN_SEQUENCE_STEPS):
        step = int(raw_step)
        if step > 0:
            step = min(step, max_step)
        elif step < 0:
            step = -min(abs(step), max_step)
        snapped = float(step * unit_rad)
        if not any(abs(snapped - existing) < 1e-6 for existing in sequence):
            sequence.append(snapped)

    if len(sequence) == 0:
        return [float(max_step * unit_rad), float(-max_step * unit_rad), 0.0]
    return sequence


def rotate_theta_half(task):
    half = float(getattr(task, "rotate_theta_shared_half_rad", np.pi))
    return max(0.0, half)


def rotate_theta_center(task):
    half = rotate_theta_half(task)
    return [-half, half]


def rotate_theta_side(task, side=1.0):
    side_sign = 1.0 if float(side) >= 0.0 else -1.0
    outer = rotate_theta_half(task)
    inner = outer * DEFAULT_SIDE_INNER_RATIO
    if side_sign > 0:
        return [inner, outer]
    return [-outer, -inner]


def rotate_theta_fixed(task, side=1.0):
    sign = 1.0 if float(side) >= 0.0 else -1.0
    val = rotate_theta_half(task) * DEFAULT_FIXED_RATIO * sign
    return [val, val]


def rotate_theta_mixed(task, side=1.0):
    far = rotate_theta_half(task)
    near = far * DEFAULT_MIXED_NEAR_RATIO
    if float(side) >= 0.0:
        return [-far, near]
    return [-near, far]


def adapt_rotate_theta_range(task, theta_lim):
    """
    Adapt an old hard-coded theta range (designed around reference fan angle)
    to current fan-angle workspace.
    """
    lo, hi = _normalize_theta_range(theta_lim)
    scale = float(getattr(task, "rotate_theta_scale", 1.0))
    half = float(getattr(task, "rotate_object_theta_half_rad", np.pi))

    lo = float(np.clip(lo * scale, -half, half))
    hi = float(np.clip(hi * scale, -half, half))

    if hi < lo:
        mid = float(np.clip(0.5 * (lo + hi), -half, half))
        lo, hi = mid, mid

    return [lo, hi]
