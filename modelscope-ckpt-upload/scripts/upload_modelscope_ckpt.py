#!/usr/bin/env python3
"""Upload local folders to the ModelScope robotwin checkpoint repository."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


DEFAULT_REPO_ID = "conroy1201/robotwin-ckpt"
DEFAULT_REPO_TYPE = "model"
DEFAULT_MAX_WORKERS = min(8, max(1, os.cpu_count() or 4))
DEFAULT_STAGE_WORKERS = min(4, DEFAULT_MAX_WORKERS)
DEFAULT_SPLIT_THRESHOLD = 512 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024
MANIFEST_NAME = ".modelscope_upload_manifest.json"
SPLIT_DIR_NAME = ".modelscope_split_parts"


@dataclass
class FolderSpec:
    path: Path
    remote: str
    ignore: list[str] = field(default_factory=list)


@dataclass
class FilePlan:
    source: Path
    rel: str
    size: int
    ignored: bool
    split: bool


@dataclass
class StagedFolder:
    original_path: Path
    remote: str
    staged_path: Path
    total_files: int
    ignored_files: int
    uploaded_files: int
    split_files: int
    manifest_path: Path


def parse_size(value: str) -> int:
    text = str(value).strip()
    if not text:
        fail("size value must not be empty")
    units = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    lower = text.lower()
    number = lower
    multiplier = 1
    for suffix, value_multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if lower.endswith(suffix):
            number = lower[: -len(suffix)]
            multiplier = value_multiplier
            break
    try:
        return int(float(number.strip()) * multiplier)
    except ValueError:
        fail(f"invalid size value: {value}")
    raise AssertionError("unreachable")


def format_size(size: int) -> str:
    value = float(size)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload one or more local folders to ModelScope as sibling remote "
            "directories while preserving each folder's internal structure. Large "
            ".pt files are split into checksum-verified parts by default."
        )
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="Local folders to upload. Each folder is uploaded under its basename by default.",
    )
    parser.add_argument(
        "--spec",
        help=(
            "JSON file containing a list of objects with path, optional remote, "
            "and optional ignore fields."
        ),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target ModelScope repo id.")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, help="Target repo type.")
    parser.add_argument("--revision", help="Target branch or revision.")
    parser.add_argument("--commit-message", help="Commit message for each upload.")
    parser.add_argument("--commit-description", help="Extended commit description.")
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="KEY=GLOB",
        help=(
            "Per-folder ignore pattern. KEY may be the local path, absolute path, "
            "or remote folder name. Repeat for multiple patterns."
        ),
    )
    parser.add_argument(
        "--ignore-all",
        action="append",
        default=[],
        metavar="GLOB",
        help="Ignore pattern applied to every uploaded folder. Repeatable.",
    )
    parser.add_argument(
        "--remote-name",
        action="append",
        default=[],
        metavar="LOCAL=REMOTE",
        help=(
            "Override remote folder name. LOCAL may be the local path, absolute path, "
            "or basename. Repeatable."
        ),
    )
    parser.add_argument("--endpoint", help="ModelScope endpoint, for example https://modelscope.cn.")
    parser.add_argument("--cli", help="CLI executable to use; defaults to ms, then modelscope.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Parallel upload workers passed to the CLI. Default: {DEFAULT_MAX_WORKERS}.",
    )
    parser.add_argument(
        "--stage-workers",
        type=int,
        default=DEFAULT_STAGE_WORKERS,
        help=f"Local staging/checksum workers. Default: {DEFAULT_STAGE_WORKERS}.",
    )
    parser.add_argument(
        "--split-pt",
        dest="split_pt",
        action="store_true",
        help="Split large .pt files before upload. Enabled by default.",
    )
    parser.add_argument(
        "--no-split-pt",
        dest="split_pt",
        action="store_false",
        help="Upload .pt files as original files without splitting.",
    )
    parser.set_defaults(split_pt=True)
    parser.add_argument(
        "--split-extension",
        action="append",
        default=[],
        metavar="EXT",
        help="Additional extension to split, for example .pth. Repeatable.",
    )
    parser.add_argument(
        "--split-threshold",
        default=str(DEFAULT_SPLIT_THRESHOLD),
        type=parse_size,
        help=f"Split files only when size is at least this value. Default: {format_size(DEFAULT_SPLIT_THRESHOLD)}.",
    )
    parser.add_argument(
        "--chunk-size",
        default=str(DEFAULT_CHUNK_SIZE),
        type=parse_size,
        help=f"Part size for split .pt files. Default: {format_size(DEFAULT_CHUNK_SIZE)}.",
    )
    parser.add_argument(
        "--no-sha-manifest",
        dest="sha_manifest",
        action="store_false",
        help="Do not compute SHA256 manifest entries for uploaded files.",
    )
    parser.set_defaults(sha_manifest=True)
    parser.add_argument(
        "--upload-retries",
        type=int,
        default=3,
        help="Retry each folder upload this many times before failing. Default: 3.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=10.0,
        help="Initial seconds to sleep between retries; doubles after each failure.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Pass --no-cache to ModelScope upload.")
    parser.add_argument("--staging-dir", help="Directory where temporary upload trees are prepared.")
    parser.add_argument("--keep-staging", action="store_true", help="Keep staging files after upload.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build the staging tree and SHA manifest, then stop before uploading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print the upload plan without staging or uploading.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable ModelScope progress bars when the CLI supports it.",
    )
    return parser.parse_args()


def fail(message: str, code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_spec(path: str) -> list[FolderSpec]:
    spec_path = Path(path).expanduser()
    if not spec_path.exists():
        fail(f"spec file does not exist: {spec_path}")
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON spec {spec_path}: {exc}")
    if not isinstance(raw, list):
        fail("spec JSON must be a list")

    specs: list[FolderSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            fail(f"spec item {index} must be an object")
        if "path" not in item:
            fail(f"spec item {index} is missing required field: path")
        folder = Path(str(item["path"])).expanduser()
        remote = item.get("remote") or item.get("remote_name") or folder.name
        ignore = item.get("ignore", [])
        if isinstance(ignore, str):
            ignore = [ignore]
        if not isinstance(ignore, list) or not all(isinstance(value, str) for value in ignore):
            fail(f"spec item {index} field 'ignore' must be a string or list of strings")
        specs.append(FolderSpec(path=folder, remote=str(remote).strip("/"), ignore=list(ignore)))
    return specs


def split_mapping(value: str, option_name: str) -> tuple[str, str]:
    if "=" not in value:
        fail(f"{option_name} must use KEY=VALUE syntax: {value}")
    key, mapped = value.split("=", 1)
    key = key.strip()
    mapped = mapped.strip()
    if not key or not mapped:
        fail(f"{option_name} must have non-empty KEY and VALUE: {value}")
    return key, mapped


def build_folder_specs(args: argparse.Namespace) -> list[FolderSpec]:
    specs = load_spec(args.spec) if args.spec else []
    specs.extend(FolderSpec(path=Path(folder).expanduser(), remote=Path(folder).expanduser().name) for folder in args.folders)
    if not specs:
        fail("provide at least one folder path or --spec")

    remote_overrides: dict[str, str] = {}
    for item in args.remote_name:
        key, remote = split_mapping(item, "--remote-name")
        remote_overrides[key] = remote.strip("/")

    for spec in specs:
        path_abs = spec.path.resolve()
        candidates = {str(spec.path), str(path_abs), spec.path.name, spec.remote}
        for key, remote in remote_overrides.items():
            if key in candidates:
                spec.remote = remote

    per_folder_ignores: list[tuple[str, str]] = [split_mapping(item, "--ignore") for item in args.ignore]
    for spec in specs:
        path_abs = spec.path.resolve()
        candidates = {str(spec.path), str(path_abs), spec.path.name, spec.remote}
        spec.ignore.extend(args.ignore_all)
        for key, pattern in per_folder_ignores:
            if key in candidates:
                spec.ignore.append(pattern)

    for spec in specs:
        spec.remote = spec.remote.strip("/")
        if not spec.remote or spec.remote in {".", ".."}:
            fail(f"invalid remote folder name for {spec.path}: {spec.remote!r}")
        if spec.remote.startswith("../") or "/../" in spec.remote:
            fail(f"remote folder name must not contain '..': {spec.remote}")

    remotes: dict[str, Path] = {}
    for spec in specs:
        previous = remotes.get(spec.remote)
        if previous is not None and previous.resolve() != spec.path.resolve():
            fail(
                "two folders resolve to the same remote folder "
                f"{spec.remote!r}: {previous} and {spec.path}. "
                "Use --remote-name LOCAL=REMOTE to disambiguate."
            )
        remotes[spec.remote] = spec.path

    return specs


def validate_args(args: argparse.Namespace) -> None:
    if args.max_workers <= 0:
        fail("--max-workers must be positive")
    if args.stage_workers <= 0:
        fail("--stage-workers must be positive")
    if args.chunk_size <= 0:
        fail("--chunk-size must be positive")
    if args.split_threshold < 0:
        fail("--split-threshold must be non-negative")
    if args.upload_retries <= 0:
        fail("--upload-retries must be positive")
    if args.retry_sleep < 0:
        fail("--retry-sleep must be non-negative")


def validate_folders(specs: list[FolderSpec]) -> None:
    for spec in specs:
        if not spec.path.exists():
            fail(f"folder does not exist: {spec.path}")
        if not spec.path.is_dir():
            fail(f"path is not a folder: {spec.path}")


def choose_cli(cli_arg: str | None) -> str:
    candidates = [cli_arg] if cli_arg else ["ms", "modelscope"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return candidate
    fail(
        "ModelScope CLI not found. Install it with "
        "`pip install -U modelscope-hub`, then run `ms login`."
    )


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def check_login(cli: str, endpoint: str | None) -> None:
    cmd = [cli]
    if endpoint:
        cmd.extend(["--endpoint", endpoint])
    cmd.append("whoami")
    proc = run_command(cmd, check=False)
    if proc.returncode == 0:
        output = (proc.stdout or proc.stderr).strip()
        if output:
            print(f"ModelScope login verified: {output.splitlines()[0]}")
        else:
            print("ModelScope login verified.")
        return

    print((proc.stdout or "").strip(), file=sys.stderr)
    print((proc.stderr or "").strip(), file=sys.stderr)
    fail(
        "ModelScope is not logged in or the token is invalid. "
        "Run `ms login` or set MODELSCOPE_API_TOKEN before uploading.",
        code=1,
    )


def split_extensions(args: argparse.Namespace) -> set[str]:
    extensions = {".pt"}
    for item in args.split_extension:
        normalized = item if item.startswith(".") else f".{item}"
        extensions.add(normalized.lower())
    return extensions


def matches_any(path: str, patterns: list[str], is_dir: bool = False) -> bool:
    candidates = [path, f"./{path}"]
    if is_dir:
        candidates.extend([f"{path}/", f"./{path}/", f"{path}/__dir_probe__", f"./{path}/__dir_probe__"])
    return any(fnmatch.fnmatch(candidate, pattern) for pattern in patterns for candidate in candidates)


def should_split(path: Path, size: int, args: argparse.Namespace) -> bool:
    if not args.split_pt:
        return False
    return path.suffix.lower() in split_extensions(args) and size >= args.split_threshold


def plan_files(spec: FolderSpec, args: argparse.Namespace) -> list[FilePlan]:
    plans: list[FilePlan] = []
    for root, dirs, files in os.walk(spec.path):
        root_path = Path(root)
        kept_dirs = []
        for dirname in dirs:
            rel_dir = (root_path / dirname).relative_to(spec.path).as_posix()
            if not matches_any(rel_dir, spec.ignore, is_dir=True):
                kept_dirs.append(dirname)
        dirs[:] = kept_dirs
        for filename in files:
            source = root_path / filename
            rel = source.relative_to(spec.path).as_posix()
            ignored = matches_any(rel, spec.ignore)
            size = source.stat().st_size
            plans.append(FilePlan(source=source, rel=rel, size=size, ignored=ignored, split=should_split(source, size, args)))
    return sorted(plans, key=lambda item: item.rel)


def summarize_plan(specs: list[FolderSpec], args: argparse.Namespace) -> None:
    print(f"repo: {args.repo_id} ({args.repo_type})")
    print(f"upload workers: {args.max_workers}; staging workers: {args.stage_workers}")
    if args.split_pt:
        extensions = ", ".join(sorted(split_extensions(args)))
        print(
            "split policy: "
            f"{extensions} >= {format_size(args.split_threshold)} into {format_size(args.chunk_size)} parts"
        )
    else:
        print("split policy: disabled")
    for spec in specs:
        plans = plan_files(spec, args)
        ignored = sum(1 for item in plans if item.ignored)
        split_count = sum(1 for item in plans if item.split and not item.ignored)
        upload_count = len(plans) - ignored
        upload_size = sum(item.size for item in plans if not item.ignored)
        print(f"- {spec.path} -> {spec.remote}/")
        print(f"  files: {upload_count} upload, {ignored} file-level ignored, {len(plans)} enumerated")
        print(f"  upload bytes: {format_size(upload_size)}; split candidates: {split_count}")
        if spec.ignore:
            print("  note: ignored directories are pruned before file counting")
            print("  ignore:")
            for pattern in spec.ignore:
                print(f"    - {pattern}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, target: Path) -> str:
    ensure_parent(target)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def split_file(source: Path, rel: str, staged_root: Path, args: argparse.Namespace) -> dict[str, object]:
    rel_path = Path(rel)
    part_dir = Path(SPLIT_DIR_NAME) / rel_path.parent / f"{rel_path.name}.parts"
    staged_part_dir = staged_root / part_dir
    staged_part_dir.mkdir(parents=True, exist_ok=True)
    original_digest = hashlib.sha256()
    parts: list[dict[str, object]] = []
    index = 0
    buffer_size = 8 * 1024 * 1024
    with source.open("rb") as handle:
        while True:
            part_name = f"{rel_path.name}.part{index:05d}"
            part_path = staged_part_dir / part_name
            part_digest = hashlib.sha256()
            part_size = 0
            with part_path.open("wb") as output:
                while part_size < args.chunk_size:
                    read_size = min(buffer_size, args.chunk_size - part_size)
                    data = handle.read(read_size)
                    if not data:
                        break
                    output.write(data)
                    original_digest.update(data)
                    part_digest.update(data)
                    part_size += len(data)
            if part_size == 0:
                part_path.unlink(missing_ok=True)
                break
            parts.append(
                {
                    "path": (part_dir / part_name).as_posix(),
                    "size": part_size,
                    "sha256": part_digest.hexdigest(),
                }
            )
            index += 1
            if part_size < args.chunk_size:
                break

    placeholder_rel = f"{rel}.split.json"
    placeholder = staged_root / placeholder_rel
    ensure_parent(placeholder)
    entry: dict[str, object] = {
        "path": rel,
        "size": source.stat().st_size,
        "sha256": original_digest.hexdigest(),
        "chunk_size": args.chunk_size,
        "parts": parts,
        "placeholder": placeholder_rel,
    }
    placeholder.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entry


def stage_regular_file(plan: FilePlan, staged_root: Path, args: argparse.Namespace) -> dict[str, object]:
    target = staged_root / plan.rel
    mode = link_or_copy(plan.source, target)
    entry: dict[str, object] = {
        "path": plan.rel,
        "size": plan.size,
        "staged_as": mode,
    }
    if args.sha_manifest:
        entry["sha256"] = sha256_file(plan.source)
    return entry


def stage_one_file(plan: FilePlan, staged_root: Path, args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    if plan.split:
        return "split", split_file(plan.source, plan.rel, staged_root, args)
    return "regular", stage_regular_file(plan, staged_root, args)


def create_staging_root(specs: list[FolderSpec], args: argparse.Namespace) -> Path:
    if args.staging_dir:
        base = Path(args.staging_dir).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="modelscope_upload_", dir=str(base)))
    first_parent = specs[0].path.resolve().parent
    try:
        return Path(tempfile.mkdtemp(prefix=".modelscope_upload_", dir=str(first_parent)))
    except OSError:
        return Path(tempfile.mkdtemp(prefix="modelscope_upload_"))


def stage_folder(spec: FolderSpec, stage_root: Path, args: argparse.Namespace) -> StagedFolder:
    plans = plan_files(spec, args)
    staged_path = stage_root / spec.remote
    staged_path.mkdir(parents=True, exist_ok=True)
    upload_plans = [plan for plan in plans if not plan.ignored]
    regular_files: list[dict[str, object]] = []
    split_files: list[dict[str, object]] = []
    errors: list[str] = []

    print(f"Staging {spec.path} -> {staged_path}")
    with ThreadPoolExecutor(max_workers=args.stage_workers) as pool:
        futures = [pool.submit(stage_one_file, plan, staged_path, args) for plan in upload_plans]
        for future in as_completed(futures):
            try:
                kind, entry = future.result()
            except Exception as exc:  # noqa: BLE001 - keep staging failures actionable.
                errors.append(str(exc))
                continue
            if kind == "split":
                split_files.append(entry)
            else:
                regular_files.append(entry)

    if errors:
        for error in errors[:20]:
            print(f"staging error: {error}", file=sys.stderr)
        fail(f"failed to stage {len(errors)} file(s) from {spec.path}")

    manifest = {
        "format": "modelscope-ckpt-upload-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(spec.path.resolve()),
        "remote_folder": spec.remote,
        "split_dir": SPLIT_DIR_NAME,
        "split_extensions": sorted(split_extensions(args)) if args.split_pt else [],
        "split_threshold": args.split_threshold,
        "chunk_size": args.chunk_size,
        "sha_manifest": bool(args.sha_manifest),
        "regular_files": sorted(regular_files, key=lambda item: str(item["path"])),
        "split_files": sorted(split_files, key=lambda item: str(item["path"])),
    }
    manifest_path = staged_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ignored = sum(1 for item in plans if item.ignored)
    split_count = len(split_files)
    print(
        f"  staged {len(upload_plans)} file(s), file-level ignored {ignored}, "
        f"split {split_count}; manifest: {manifest_path}"
    )
    return StagedFolder(
        original_path=spec.path,
        remote=spec.remote,
        staged_path=staged_path,
        total_files=len(plans),
        ignored_files=ignored,
        uploaded_files=len(upload_plans),
        split_files=split_count,
        manifest_path=manifest_path,
    )


def stage_all(specs: list[FolderSpec], args: argparse.Namespace) -> tuple[Path, list[StagedFolder]]:
    stage_root = create_staging_root(specs, args)
    staged: list[StagedFolder] = []
    print(f"staging root: {stage_root}")
    for spec in specs:
        staged.append(stage_folder(spec, stage_root, args))
    return stage_root, staged


def upload_folder(cli: str, staged: StagedFolder, args: argparse.Namespace) -> None:
    cmd = [cli]
    if args.endpoint:
        cmd.extend(["--endpoint", args.endpoint])
    cmd.extend(
        [
            "upload",
            args.repo_id,
            str(staged.staged_path),
            staged.remote,
            "--repo-type",
            args.repo_type,
        ]
    )
    if args.revision:
        cmd.extend(["--revision", args.revision])
    if args.commit_message:
        cmd.extend(["--commit-message", args.commit_message])
    else:
        cmd.extend(["--commit-message", f"upload {staged.remote}"])
    if args.commit_description:
        cmd.extend(["--commit-description", args.commit_description])
    if args.max_workers:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.no_cache:
        cmd.append("--no-cache")
    if args.disable_tqdm:
        cmd.append("--disable-tqdm")

    print(f"Uploading {staged.original_path} -> {args.repo_id}/{staged.remote}/")
    print("+ " + " ".join(shlex.quote(part) for part in cmd))
    delay = args.retry_sleep
    for attempt in range(1, args.upload_retries + 1):
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            return
        if attempt == args.upload_retries:
            fail(
                f"upload failed for {staged.original_path} after {attempt} attempt(s) "
                f"with exit code {proc.returncode}",
                code=proc.returncode,
            )
        print(f"upload attempt {attempt} failed; retrying in {delay:.1f}s", file=sys.stderr)
        time.sleep(delay)
        delay *= 2


def main() -> None:
    args = parse_args()
    validate_args(args)
    specs = build_folder_specs(args)
    validate_folders(specs)

    if args.dry_run:
        summarize_plan(specs, args)
        print("dry run complete; no staging or upload performed.")
        return

    if args.prepare_only:
        summarize_plan(specs, args)
        stage_root, _ = stage_all(specs, args)
        print(f"prepare-only complete; staging kept at {stage_root}")
        return

    cli = choose_cli(args.cli)
    check_login(cli, args.endpoint)
    summarize_plan(specs, args)
    stage_root, staged_folders = stage_all(specs, args)
    try:
        for staged in staged_folders:
            upload_folder(cli, staged, args)
    finally:
        if args.keep_staging:
            print(f"staging kept at {stage_root}")
        else:
            shutil.rmtree(stage_root, ignore_errors=True)
    print("upload complete.")


if __name__ == "__main__":
    main()
