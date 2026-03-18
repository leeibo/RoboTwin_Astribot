#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import deque


def format_row(row):
    return "[" + ", ".join(f"{float(v): .5f}" for v in row) + "]"


def print_record(record):
    if record.get("event") != "left_live_frame_snapshot":
        return
    seq = record.get("seq")
    source = record.get("source")
    ts = record.get("timestamp")
    live_p = record.get("live_world_p", [])
    live_q = record.get("live_world_q", [])
    r_ref_live = record.get("R_ref_live", [])
    print(f"\n[LiveFrame] seq={seq} source={source} timestamp={ts}")
    print(f"  live_world_p: {live_p}")
    print(f"  live_world_q: {live_q}")
    if isinstance(r_ref_live, list) and len(r_ref_live) == 3:
        print("  R_ref_live:")
        print(f"    {format_row(r_ref_live[0])}")
        print(f"    {format_row(r_ref_live[1])}")
        print(f"    {format_row(r_ref_live[2])}")


def read_tail_lines(path, tail_count):
    if tail_count <= 0:
        return []
    buf = deque(maxlen=tail_count)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                buf.append(line)
    return list(buf)


def follow_file(path, interval, start_at_end):
    with open(path, "r", encoding="utf-8") as f:
        if start_at_end:
            f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(interval)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            print_record(record)


def main():
    parser = argparse.ArgumentParser(description="Watch live frame calibration jsonl log.")
    parser.add_argument(
        "--log",
        default="script/calibration/live_frame_records.jsonl",
        help="Path to jsonl log file.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=5,
        help="Show last N existing snapshot records before follow.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Polling interval in seconds when following.",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Only print current tail records and exit.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"log file not found: {args.log}")

    tail_lines = read_tail_lines(args.log, args.tail)
    for line in tail_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        print_record(record)

    if args.no_follow:
        return

    print(f"\nWatching: {args.log}")
    follow_file(args.log, args.interval, start_at_end=True)


if __name__ == "__main__":
    main()
