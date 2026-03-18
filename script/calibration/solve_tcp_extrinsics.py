#!/usr/bin/env python3
"""
Solve delta_matrix and global_trans_matrix from calibration samples.

Input sample JSON fields (per sample):
  arm, source, target_tcp_q, actual_endlink_q
Optional:
  target_tcp_p, actual_endlink_p
"""

import argparse
import itertools
import json
import math
import os

import numpy as np
import transforms3d as t3d


def nearest_rotation(mats: list[np.ndarray]) -> np.ndarray:
    mean_m = np.mean(np.stack(mats, axis=0), axis=0)
    u, _, vt = np.linalg.svd(mean_m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def rot_err_deg(r_pred: np.ndarray, r_gt: np.ndarray) -> float:
    d = r_pred.T @ r_gt
    tr = np.clip((np.trace(d) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(tr))


def axis_rotations() -> list[np.ndarray]:
    cands = []
    eye = np.eye(3)
    for perm in itertools.permutations([0, 1, 2]):
        p = eye[:, perm]
        for sx, sy, sz in itertools.product([-1.0, 1.0], repeat=3):
            s = np.diag([sx, sy, sz])
            r = p @ s
            if np.linalg.det(r) > 0.5:
                cands.append(r)
    return cands


def nearest_axis_rotation(r: np.ndarray) -> np.ndarray:
    cands = axis_rotations()
    best = min(cands, key=lambda c: float(np.linalg.norm(r - c)))
    return best


def to_mat(q_wxyz: list[float]) -> np.ndarray:
    return t3d.quaternions.quat2mat(np.array(q_wxyz, dtype=np.float64))


def fmt_mat(r: np.ndarray, nd=6) -> str:
    return np.array2string(np.round(r, nd), separator=", ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="Path to samples JSON")
    parser.add_argument("--arm", default="", choices=["", "left", "right"])
    parser.add_argument("--source", default="")
    parser.add_argument("--out", default="", help="Optional output JSON")
    args = parser.parse_args()

    path = os.path.abspath(args.samples)
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    filtered = []
    for s in samples:
        if args.arm and s.get("arm") != args.arm:
            continue
        if args.source and s.get("source") != args.source:
            continue
        if "target_tcp_q" not in s or "actual_endlink_q" not in s:
            continue
        filtered.append(s)

    if len(filtered) < 3:
        raise SystemExit(f"Need at least 3 samples, got {len(filtered)}")

    r_ee_list = [to_mat(s["actual_endlink_q"]) for s in filtered]
    r_tcp_list = [to_mat(s["target_tcp_q"]) for s in filtered]

    # Step 1: solve delta assuming global=I.
    delta_candidates = [r_ee.T @ r_tcp for r_ee, r_tcp in zip(r_ee_list, r_tcp_list)]
    r_delta = nearest_rotation(delta_candidates)

    # Step 2: solve global with solved delta.
    global_candidates = [r_ee.T @ r_tcp @ r_delta.T for r_ee, r_tcp in zip(r_ee_list, r_tcp_list)]
    r_global = nearest_rotation(global_candidates)

    # Evaluate.
    errs = []
    for r_ee, r_tcp in zip(r_ee_list, r_tcp_list):
        r_pred = r_ee @ r_global @ r_delta
        errs.append(rot_err_deg(r_pred, r_tcp))
    err_mean = float(np.mean(errs))
    err_max = float(np.max(errs))

    # Optional gripper_bias estimation if positions exist.
    bias_vals = []
    yz_res = []
    for s, r_ee in zip(filtered, r_ee_list):
        if "target_tcp_p" not in s or "actual_endlink_p" not in s:
            continue
        p_tcp = np.array(s["target_tcp_p"], dtype=np.float64)
        p_ee = np.array(s["actual_endlink_p"], dtype=np.float64)
        r_tool = r_ee @ r_global @ r_delta
        v = r_tool.T @ (p_tcp - p_ee)
        bias_vals.append(float(v[0]))
        yz_res.append(float(np.linalg.norm(v[1:])))
    bias_est = float(np.mean(bias_vals)) if bias_vals else None
    yz_res_mean = float(np.mean(yz_res)) if yz_res else None

    r_delta_snap = nearest_axis_rotation(r_delta)
    r_global_snap = nearest_axis_rotation(r_global)

    print(f"Samples used: {len(filtered)}")
    print(f"Rotation error after fit: mean={err_mean:.3f} deg, max={err_max:.3f} deg")
    if bias_est is not None:
        print(f"Estimated gripper_bias (x-axis): {bias_est:.6f} m")
        print(f"Mean yz residual in tool frame: {yz_res_mean:.6f} m")

    print("\nRaw delta_matrix:")
    print(fmt_mat(r_delta))
    print("Snapped delta_matrix (nearest axis-aligned rotation):")
    print(fmt_mat(r_delta_snap, nd=0))

    print("\nRaw global_trans_matrix:")
    print(fmt_mat(r_global))
    print("Snapped global_trans_matrix (nearest axis-aligned rotation):")
    print(fmt_mat(r_global_snap, nd=0))

    print("\nSuggested config snippet (use snapped first, then validate):")
    print(
        "delta_matrix: "
        + str(np.round(r_delta_snap, 0).astype(int).tolist())
    )
    print(
        "global_trans_matrix: "
        + str(np.round(r_global_snap, 0).astype(int).tolist())
    )
    if bias_est is not None:
        print(f"gripper_bias: {bias_est:.6f}")

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_samples": len(filtered),
                    "rotation_error_mean_deg": err_mean,
                    "rotation_error_max_deg": err_max,
                    "delta_matrix_raw": r_delta.tolist(),
                    "delta_matrix_snapped": np.round(r_delta_snap, 0).astype(int).tolist(),
                    "global_trans_matrix_raw": r_global.tolist(),
                    "global_trans_matrix_snapped": np.round(r_global_snap, 0).astype(int).tolist(),
                    "gripper_bias_est": bias_est,
                    "yz_residual_mean": yz_res_mean,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
