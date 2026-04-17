import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.rotate_vlm import (  # noqa: E402
    DEFAULT_ACTION_CHUNK_SIZE,
    DEFAULT_MAX_CONTEXT_FRAMES,
    export_task_vlm_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Export rotate-view VLM samples from collected data.")
    parser.add_argument(
        "--save-dir",
        required=True,
        type=str,
        help="Collected task directory, e.g. ./data/place_a2b_left_rotate_view/demo_clean__easy_fan150",
    )
    parser.add_argument(
        "--max-context-frames",
        type=int,
        default=DEFAULT_MAX_CONTEXT_FRAMES,
        help="Maximum number of ordered input frames per VLM sample.",
    )
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=DEFAULT_ACTION_CHUNK_SIZE,
        help="Number of future arm-action steps stored for each stage3 action chunk sample.",
    )
    parser.add_argument(
        "--task-types",
        nargs="*",
        default=None,
        help="Optional rotate VLM task types to export. Defaults to all registered types.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing JSON file instead of overwriting it.",
    )
    args = parser.parse_args()

    summary = export_task_vlm_dataset(
        save_dir=args.save_dir,
        overwrite=not args.append,
        max_context_frames=args.max_context_frames,
        action_chunk_size=args.action_chunk_size,
        task_types=args.task_types,
    )
    paths_text = ", ".join(f"{task_type}={path}" for task_type, path in summary.get("samples_paths", {}).items())
    print(f"[Rotate VLM] exported {summary['sample_count']} samples to {paths_text}")


if __name__ == "__main__":
    main()
