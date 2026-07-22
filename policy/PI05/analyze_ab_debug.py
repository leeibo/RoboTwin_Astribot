#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

TORSO_LIMITS = (-1.2, 1.2)


def _summary(run_root: Path) -> list[dict[str, str]]:
    path = run_root / "summary.tsv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _debug_paths(run_root: Path) -> list[Path]:
    return sorted(run_root.rglob("pi05_request_debug.jsonl"))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json in {path}:{line_no}: {exc}") from exc
    return rows


def _ratio(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(bool(v) for v in values) / len(values)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def collect(run_root: Path) -> dict:
    rows = []
    for path in _debug_paths(run_root):
        rows.extend(_load_jsonl(path))

    summary_rows = _summary(run_root)
    target_flags = []
    at_limit_push = []
    state_at_limit = []
    torso_targets = []
    raw_torso_first = []
    action_torso_first = []
    semantics = set()
    episodes = set()
    for row in rows:
        semantics.add(row.get("action_semantics", "unknown"))
        episodes.add(row.get("episode"))
        state = row.get("torso_state")
        if isinstance(state, (int, float)):
            state_at_limit.append(state <= TORSO_LIMITS[0] + 0.01 or state >= TORSO_LIMITS[1] - 0.01)
        flags = row.get("torso_target_outside_limit") or []
        target_flags.extend(bool(x) for x in flags)
        at_limit_push.append(bool(row.get("torso_at_limit_push_outside")))
        targets = row.get("torso_targets") or []
        torso_targets.extend(float(x) for x in targets if isinstance(x, (int, float)))
        raw = row.get("first_raw_action18") or []
        act = row.get("first_action18") or []
        if len(raw) > 16 and isinstance(raw[16], (int, float)):
            raw_torso_first.append(float(raw[16]))
        if len(act) > 16 and isinstance(act[16], (int, float)):
            action_torso_first.append(float(act[16]))

    statuses = [r.get("status", "") for r in summary_rows]
    success_rates = [r.get("success_rate", "-") for r in summary_rows]
    return {
        "run": str(run_root),
        "tasks": len(summary_rows),
        "statuses": ",".join(statuses) if statuses else "-",
        "success_rate": ",".join(success_rates) if success_rates else "-",
        "debug_rows": len(rows),
        "episodes_seen": len({e for e in episodes if e is not None}),
        "semantics": ",".join(sorted(semantics)) if semantics else "-",
        "torso_target_outside_ratio": _ratio(target_flags),
        "torso_state_at_limit_ratio": _ratio(state_at_limit),
        "torso_at_limit_push_outside_ratio": _ratio(at_limit_push),
        "torso_target_min": min(torso_targets) if torso_targets else None,
        "torso_target_max": max(torso_targets) if torso_targets else None,
        "first_raw_torso_mean": mean(raw_torso_first) if raw_torso_first else None,
        "first_action_torso_mean": mean(action_torso_first) if action_torso_first else None,
    }


def print_table(items: list[dict]) -> None:
    keys = [
        "run", "statuses", "success_rate", "debug_rows", "episodes_seen", "semantics",
        "torso_target_outside_ratio", "torso_state_at_limit_ratio", "torso_at_limit_push_outside_ratio",
        "torso_target_min", "torso_target_max", "first_raw_torso_mean", "first_action_torso_mean",
    ]
    print("\t".join(keys))
    for item in items:
        print("\t".join(_fmt(item.get(key)) for key in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PI05 A/B debug metrics from eval run roots.")
    parser.add_argument("runs", nargs="+", type=Path, help="PI05 eval run root(s), e.g. policy/PI05/runs/eval/pi05_ab_A_...")
    args = parser.parse_args()
    print_table([collect(path.expanduser().resolve()) for path in args.runs])


if __name__ == "__main__":
    main()
