#!/usr/bin/env python3
"""
Extract calibration samples from RoboTwin logs.

The script pairs:
  - target tcp pose from [TCP_DEBUG:*]
  - actual endlink pose from [ENDLINK_DEBUG:*]
for the same arm/source sequence.
"""

import argparse
import ast
import json
import os
import re


TCP_HDR = re.compile(r"^\[TCP_DEBUG:(?P<src>[^\]]+)\]\s+arm=(?P<arm>\w+)\s*$")
TCP_TGT = re.compile(r"^\s*target tcp:\s*p=\[(?P<p>[^\]]+)\],\s*q=\[(?P<q>[^\]]+)\]\s*$")
END_HDR = re.compile(r"^\[ENDLINK_DEBUG:(?P<src>[^\]]+)\]\s+arm=(?P<arm>\w+)\s*$")
END_ACT = re.compile(r"^\s*actual endlink:\s*p=\[(?P<p>[^\]]+)\],\s*q=\[(?P<q>[^\]]+)\]\s*$")


def parse_vec(text: str) -> list[float]:
    return [float(x) for x in ast.literal_eval("[" + text + "]")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to log file")
    parser.add_argument("--out", required=True, help="Output JSON")
    parser.add_argument("--arm", default="", choices=["", "left", "right"], help="Optional arm filter")
    parser.add_argument("--source", default="", help="Optional source filter, e.g. take_dense_action")
    args = parser.parse_args()

    log_path = os.path.abspath(args.log)
    out_path = os.path.abspath(args.out)

    pending: dict[tuple[str, str], dict] = {}
    samples: list[dict] = []
    tcp_key = None
    end_key = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            m = TCP_HDR.match(line)
            if m:
                tcp_key = (m.group("arm"), m.group("src"))
                continue

            m = END_HDR.match(line)
            if m:
                end_key = (m.group("arm"), m.group("src"))
                continue

            m = TCP_TGT.match(line)
            if m and tcp_key is not None:
                arm, src = tcp_key
                item = pending.setdefault((arm, src), {"arm": arm, "source": src})
                item["target_tcp_p"] = parse_vec(m.group("p"))
                item["target_tcp_q"] = parse_vec(m.group("q"))
                continue

            m = END_ACT.match(line)
            if m and end_key is not None:
                arm, src = end_key
                item = pending.setdefault((arm, src), {"arm": arm, "source": src})
                item["actual_endlink_p"] = parse_vec(m.group("p"))
                item["actual_endlink_q"] = parse_vec(m.group("q"))

                # Emit once we have both target TCP and actual endlink.
                if "target_tcp_q" in item and "actual_endlink_q" in item:
                    if (not args.arm or item["arm"] == args.arm) and (not args.source or item["source"] == args.source):
                        samples.append(dict(item))
                    pending.pop((arm, src), None)
                continue

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(samples)} samples")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
