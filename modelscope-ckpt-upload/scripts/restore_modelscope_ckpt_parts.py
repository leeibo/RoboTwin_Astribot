#!/usr/bin/env python3
"""Restore and verify .pt files split by upload_modelscope_ckpt.py."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys


MANIFEST_NAME = ".modelscope_upload_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a ModelScope upload manifest and restore split checkpoint files."
    )
    parser.add_argument("folder", help="Downloaded remote folder containing the manifest.")
    parser.add_argument("--manifest", help="Manifest path. Defaults to FOLDER/.modelscope_upload_manifest.json.")
    parser.add_argument("--verify-only", action="store_true", help="Verify checksums without reconstructing .pt files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing restored files.")
    parser.add_argument("--remove-parts", action="store_true", help="Remove split part files after successful restore.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel checksum workers. Default: 4.")
    return parser.parse_args()


def fail(message: str, code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(folder: Path, manifest_arg: str | None) -> dict[str, object]:
    manifest_path = Path(manifest_arg).expanduser() if manifest_arg else folder / MANIFEST_NAME
    if not manifest_path.exists():
        fail(f"manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid manifest JSON {manifest_path}: {exc}")
    if manifest.get("format") != "modelscope-ckpt-upload-manifest-v1":
        fail(f"unsupported manifest format: {manifest.get('format')!r}")
    return manifest


def verify_file(path: Path, expected_size: int, expected_sha256: str | None) -> None:
    if not path.exists():
        fail(f"missing file: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        fail(f"size mismatch for {path}: {actual_size} != {expected_size}")
    if expected_sha256:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            fail(f"sha256 mismatch for {path}: {actual_sha256} != {expected_sha256}")


def verify_regular_files(folder: Path, manifest: dict[str, object], workers: int) -> None:
    entries = [entry for entry in manifest.get("regular_files", []) if entry.get("sha256")]
    if not entries:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                verify_file,
                folder / str(entry["path"]),
                int(entry["size"]),
                str(entry["sha256"]),
            )
            for entry in entries
        ]
        for future in as_completed(futures):
            future.result()
    print(f"verified {len(entries)} regular file(s)")


def verify_parts(folder: Path, split_entry: dict[str, object], workers: int) -> None:
    parts = split_entry.get("parts", [])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                verify_file,
                folder / str(part["path"]),
                int(part["size"]),
                str(part["sha256"]),
            )
            for part in parts
        ]
        for future in as_completed(futures):
            future.result()


def restore_split_file(folder: Path, split_entry: dict[str, object], args: argparse.Namespace) -> None:
    target = folder / str(split_entry["path"])
    expected_sha256 = str(split_entry["sha256"])
    expected_size = int(split_entry["size"])
    verify_parts(folder, split_entry, args.workers)

    if args.verify_only:
        if target.exists():
            verify_file(target, expected_size, expected_sha256)
        return

    if target.exists() and not args.overwrite:
        verify_file(target, expected_size, expected_sha256)
        print(f"already restored and verified: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f".{target.name}.restore_tmp")
    digest = hashlib.sha256()
    written = 0
    with tmp_target.open("wb") as output:
        for part in split_entry.get("parts", []):
            part_path = folder / str(part["path"])
            with part_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
            written += int(part["size"])
    actual_sha256 = digest.hexdigest()
    if written != expected_size:
        tmp_target.unlink(missing_ok=True)
        fail(f"restored size mismatch for {target}: {written} != {expected_size}")
    if actual_sha256 != expected_sha256:
        tmp_target.unlink(missing_ok=True)
        fail(f"restored sha256 mismatch for {target}: {actual_sha256} != {expected_sha256}")
    tmp_target.replace(target)
    print(f"restored: {target}")

    if args.remove_parts:
        for part in split_entry.get("parts", []):
            (folder / str(part["path"])).unlink(missing_ok=True)
        placeholder = split_entry.get("placeholder")
        if placeholder:
            (folder / str(placeholder)).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        fail("--workers must be positive")
    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        fail(f"folder does not exist or is not a directory: {folder}")
    manifest = load_manifest(folder, args.manifest)
    verify_regular_files(folder, manifest, args.workers)
    split_files = manifest.get("split_files", [])
    for split_entry in split_files:
        restore_split_file(folder, split_entry, args)
    print(f"verified split file(s): {len(split_files)}")
    print("restore/verify complete.")


if __name__ == "__main__":
    main()
