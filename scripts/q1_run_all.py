"""Run and record the complete reproducible pipeline for E-problem question 1."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/q1_features.yaml")
    parser.add_argument("--output-dir", default="results/q1")
    parser.add_argument("--figure-dir", default="figures/q1")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--manifest", default="results/复现清单.json")
    parser.add_argument("--status", choices=("draft", "validated"), default="draft")
    args = parser.parse_args()

    started = time.time()
    output_dir = REPO / args.output_dir
    figure_dir = REPO / args.figure_dir
    qa_dir = figure_dir.parent / f"{figure_dir.name}_qa"
    manifest_path = REPO / args.manifest
    config_path = REPO / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    legacy_manifest = output_dir / "复现清单.json"
    if legacy_manifest.exists():
        legacy_manifest.unlink()

    commands = [
        [sys.executable, "-X", "utf8", "scripts/q1_extract_features.py",
         "--config", args.config, "--output-dir", args.output_dir],
        [sys.executable, "-X", "utf8", "scripts/q1_overlap_audit.py", "--q1-dir", args.output_dir]
        + (["--reference", args.reference] if args.reference else []),
        [sys.executable, "-X", "utf8", "scripts/q1_make_figures.py",
         "--q1-dir", args.output_dir, "--figure-dir", args.figure_dir],
    ]
    for command in commands:
        run(command)

    extraction = json.loads((output_dir / "q1_extraction_reproduction.json").read_text(encoding="utf-8"))
    overlap = json.loads((output_dir / "q1_overlap_reproduction.json").read_text(encoding="utf-8"))
    logical_outputs = {
        **{f"output/{path.name}": path for path in output_dir.iterdir()
           if path.is_file() and path.name != "q1_checksums.sha256"},
        **{f"figures/{path.name}": path for path in figure_dir.iterdir() if path.is_file()},
        **{f"figure_qa/{path.name}": path for path in qa_dir.iterdir() if path.is_file()},
    }
    manifest = {
        "schema_version": "q1-full-reproduction@1.0",
        "status": args.status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": config["seed"],
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "commands_run": [subprocess.list2cmdline(command) for command in commands],
        "elapsed_sec": time.time() - started,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "config_path": args.config,
        "config_sha256": sha256(config_path),
        "input_sha256": {**extraction["input_sha256"],
                          "attachment2_aligned_50": overlap["reference_sha256"]},
        "invariants": extraction["invariants"],
        "overlap_count": overlap["overlap_count"],
        "outputs": {name: sha256(logical_outputs[name]) for name in sorted(logical_outputs)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    checksum_files = [logical_outputs[name] for name in sorted(logical_outputs)] + [manifest_path]
    checksum_path = output_dir / "q1_checksums.sha256"
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {Path(os.path.relpath(path, output_dir)).as_posix()}"
                  for path in checksum_files) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "manifest": str(manifest_path),
                      "checksummed_files": len(checksum_files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
