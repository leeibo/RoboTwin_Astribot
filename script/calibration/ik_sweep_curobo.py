#!/usr/bin/env python3
"""
IK reachable-pose sweep for a specific CuRobo arm config.

Example:
  /home/admin1/yibo/conda/envs/robotwin/bin/python script/calibration/ik_sweep_curobo.py \
    --curobo-yml assets/embodiments/astribot_descriptions/curobo_left.yml \
    --quat 1,0,0,0 \
    --out script/calibration/ik_left_identity.json
"""

import argparse
import json
import os
from itertools import product

import numpy as np
import torch
from tqdm import tqdm
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


def parse_segments(expr: str) -> list[float]:
    """
    Parse "a:b:n,c:d:m" to concatenated linspace values.
    """
    values: list[float] = []
    for seg in expr.split(","):
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid segment: {seg}. Use start:end:num.")
        start, end, num = float(parts[0]), float(parts[1]), int(parts[2])
        values.extend(np.linspace(start, end, num).tolist())
    return values


def parse_quat(expr: str) -> list[float]:
    vals = [float(x.strip()) for x in expr.split(",")]
    if len(vals) != 4:
        raise ValueError("Quaternion must have 4 values: w,x,y,z")
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curobo-yml", required=True, help="Path to curobo_left.yml or curobo_right.yml")
    parser.add_argument("--quat", default="1,0,0,0", help="Target quaternion w,x,y,z")
    parser.add_argument("--x", default="0.35:-0.2:15", help="X segments start:end:num")
    parser.add_argument("--y", default="0.25:-0.2:15", help="Y segments start:end:num")
    parser.add_argument("--z", default="0.25:-0.2:15", help="Z segments start:end:num")
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="", help="Output JSON path")
    args = parser.parse_args()

    yml_path = os.path.abspath(args.curobo_yml)
    cfg = load_yaml(yml_path)
    urdf_file = cfg["robot_cfg"]["kinematics"]["urdf_path"]
    base_link = cfg["robot_cfg"]["kinematics"]["base_link"]
    ee_link = cfg["robot_cfg"]["kinematics"]["ee_link"]

    tensor_args = TensorDeviceType()
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        num_seeds=args.num_seeds,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_cfg)

    x_values = parse_segments(args.x)
    y_values = parse_segments(args.y)
    z_values = parse_segments(args.z)
    quat = parse_quat(args.quat)
    quat_t = torch.tensor([quat], device=args.device, dtype=torch.float32)

    print("Testing IK solutions:")
    print(f"  yml: {yml_path}")
    print(f"  base_link: {base_link}, ee_link: {ee_link}")
    print(f"  quat(wxyz): {quat}")
    print(f"  grid size: {len(x_values)} x {len(y_values)} x {len(z_values)}")
    print("x, y, z, success")

    success_points = []
    total = 0
    for x, y, z in tqdm(product(x_values, y_values, z_values), total=len(x_values) * len(y_values) * len(z_values)):
        total += 1
        goal = Pose(
            position=torch.tensor([[float(x), float(y), float(z)]], device=args.device, dtype=torch.float32),
            quaternion=quat_t,
        )
        result = ik_solver.solve_single(goal)
        ok = bool(result.success.item())
        if ok:
            success_points.append([float(x), float(y), float(z)])
            # print(f"{x:.4f}, {y:.4f}, {z:.4f}, {ok}")

    print(f"\nSummary: {len(success_points)} / {total} reachable")
    if success_points:
        print(f"First reachable point: {success_points[0]}")

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "curobo_yml": yml_path,
                    "base_link": base_link,
                    "ee_link": ee_link,
                    "quat_wxyz": quat,
                    "success_points": success_points,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
