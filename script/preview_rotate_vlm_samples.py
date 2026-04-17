import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preview exported rotate VLM JSON samples.")
    parser.add_argument("--samples-path", required=True, type=str)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    samples_path = Path(args.samples_path)
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    for idx, sample in enumerate(samples[: max(int(args.limit), 0)]):
        print(f"=== sample {idx} ===")
        print("images:", sample.get("images", []))
        print("metadata:", sample.get("metadata", {}))
        messages = sample.get("messages", [])
        if messages:
            print("assistant:", messages[-1].get("content", ""))
        print()


if __name__ == "__main__":
    main()
