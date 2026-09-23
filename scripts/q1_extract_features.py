"""Run E-problem question-1 extraction on a smoke subset or all 100 samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import scipy
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data.q1_dataset import discover_samples  # noqa: E402
from features.q1_extract import extract_sample  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/q1_features.yaml")
    parser.add_argument("--limit", type=int, default=None, help="deterministic prefix; omit for all samples")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    config_path = REPO / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = REPO / (args.output_dir or config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(REPO / config["input"]["search_root"],
                               config["input"]["label_glob"], config["input"]["video_glob"])
    if len(samples) != 100:
        raise RuntimeError(f"attachment-1 invariant failed: expected 100 samples, got {len(samples)}")
    chosen = samples[: args.limit] if args.limit else samples
    started = time.time()
    extracted = []
    for position, sample in enumerate(chosen, 1):
        print(f"[{position}/{len(chosen)}] {sample.sample_id}", flush=True)
        extracted.append((sample, extract_sample(sample, config)))

    sample_ids = np.asarray([sample.sample_id for sample, _ in extracted])
    tensor_keys = ["text", "audio", "vision", "valid_text", "valid_audio", "valid_vision",
                   "injected_missing_audio", "injected_missing_vision", "padding_mask"]
    arrays = {key: np.stack([result[key] for _, result in extracted]) for key in tensor_keys}
    np.savez_compressed(output / "q1_features.npz", sample_id=sample_ids, **arrays)
    sample_rows, manifest_rows, alignment_rows = [], [], []
    for sample, result in extracted:
        sequence_length = len(result["anchors"])
        sample_rows.append({"sample_id": sample.sample_id, "video_id": sample.video_id,
                            "clip_id": sample.clip_id, "text": sample.text,
                            "label_class": sample.label_class, "label_int": sample.label_int,
                            "label_intensity": sample.label_intensity,
                            "sequence_length": sequence_length,
                            "source_path": sample.source_path.relative_to(REPO).as_posix()})
        alignment_rows.extend(result["alignments"])
        for modality, dimension, valid_key in [("text", 128, "valid_text"),
                                                ("audio", 74, "valid_audio"),
                                                ("vision", 35, "valid_vision")]:
            warnings = "" if result[valid_key].sum() == sequence_length else "invalid_nonpadding_positions"
            manifest_rows.append({"sample_id": sample.sample_id, "modality": modality,
                                  "source_path": sample.source_path.relative_to(REPO).as_posix(),
                                  "source_duration_sec": result["source_duration"],
                                  "stream_duration_sec": (result["audio_duration"] if modality == "audio" else
                                                          result["video_duration"] if modality == "vision" else
                                                          result["duration"]),
                                  "sample_rate_or_fps": result["audio_rate"] if modality == "audio" else
                                  (result["source_fps"] if modality == "vision" else "n/a"),
                                  "raw_shape": "n/a", "feature_dim": dimension,
                                  "alignment_granularity": "text_anchor", "sequence_length": sequence_length,
                                  "valid_position_count": int(result[valid_key].sum()),
                                  "padding_position_count": int(result["padding_mask"].sum()),
                                  "alignment_method": "proportional", "feature_path": "q1_features.npz",
                                  "feature_sha256": "computed_after_write", "status": "ok" if not warnings else "warning",
                                  "warnings": warnings})
    pd.DataFrame(sample_rows).to_csv(output / "q1_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(alignment_rows).to_csv(output / "q1_alignment.csv", index=False, encoding="utf-8-sig")
    feature_hash = sha256(output / "q1_features.npz")
    for row in manifest_rows:
        row["feature_sha256"] = feature_hash
    pd.DataFrame(manifest_rows).to_csv(output / "q1_feature_manifest.csv", index=False, encoding="utf-8-sig")
    (output / "q1_normalization.json").write_text(json.dumps({"method": "none", "fitted_on": None,
        "reason": "Question 1 emits raw descriptors; train-only fitting belongs to later I01 production."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "q1_README.md").write_text(
        "# Q1 feature bundle\n\n"
        "This directory is a question-1 result, not a frozen I01/I02 exchange bundle.\n\n"
        "- `q1_samples.csv`: labels, transcripts, IDs and sequence lengths.\n"
        "- `q1_features.npz`: float32 `text/audio/vision` tensors and explicit boolean masks.\n"
        "- `q1_alignment.csv`: zero-based text anchors mapped to half-open time/sample/frame ranges.\n"
        "- `q1_feature_manifest.csv`: per-sample and per-modality quality rows.\n"
        "- `q1_normalization.json`: confirms that no train-fitted normalization was applied.\n"
        "- `q1_extraction_reproduction.json`: extraction command, versions, hashes and invariant checks.\n"
        "- `../复现清单.json`: the single full-chain reproduction record.\n\n"
        "Load with `numpy.load('q1_features.npz')` and `pandas.read_csv(...)`. Missingness must be "
        "read from masks; never infer it from zero-valued features.\n",
        encoding="utf-8",
    )
    mask_keys = ["valid_text", "valid_audio", "valid_vision", "injected_missing_audio",
                 "injected_missing_vision", "padding_mask"]
    interval_coverage = all(
        rows[0]["start_sec"] == 0.0
        and abs(rows[-1]["end_sec"] - result["duration"]) < 1e-7
        and all(abs(left["end_sec"] - right["start_sec"]) < 1e-7 for left, right in zip(rows, rows[1:]))
        and all(left["audio_end_idx"] == right["audio_start_idx"] for left, right in zip(rows, rows[1:]))
        and all(0 <= row["audio_start_idx"] <= row["audio_end_idx"] <= result["audio_sample_count"] for row in rows)
        and all(0 <= row["video_start_idx"] <= row["video_end_idx"] <= result["video_frame_count"] for row in rows)
        for (_, result), rows in zip(extracted, ([r for r in alignment_rows if r["sample_id"] == sample.sample_id]
                                                 for sample, _ in extracted))
    )
    invariants = {
        "sample_count": len(extracted), "all_finite": all(np.isfinite(arrays[k]).all() for k in ("text", "audio", "vision")),
        "shape_text": list(arrays["text"].shape), "shape_audio": list(arrays["audio"].shape),
        "shape_vision": list(arrays["vision"].shape),
        "mask_sequence_consistent": bool(np.all(arrays["valid_text"].sum(axis=1) == (~arrays["padding_mask"]).sum(axis=1))),
        "mask_dtype_bool": all(arrays[key].dtype == np.bool_ for key in mask_keys),
        "valid_excludes_padding": all(not np.any(arrays[key] & arrays["padding_mask"])
                                      for key in ("valid_text", "valid_audio", "valid_vision")),
        "injected_missing_excludes_padding": all(not np.any(arrays[key] & arrays["padding_mask"])
                                                  for key in ("injected_missing_audio", "injected_missing_vision")),
        "alignment_interval_legal": all(r["end_sec"] > r["start_sec"] >= 0 for r in alignment_rows),
        "alignment_contiguous_and_bounded": interval_coverage,
        "alignment_end_within_streams": all(
            result["duration"] <= duration + 1e-7
            for _, result in extracted
            for duration in (result["source_duration"], result["audio_duration"], result["video_duration"])
            if duration > 0
        ),
    }
    if not all(value for key, value in invariants.items() if key not in {"sample_count", "shape_text", "shape_audio", "shape_vision"}):
        raise RuntimeError(f"output invariant failed: {invariants}")
    data_output_names = ["q1_features.npz", "q1_samples.csv", "q1_alignment.csv",
                         "q1_feature_manifest.csv", "q1_normalization.json", "q1_README.md"]
    outputs = [output / name for name in data_output_names]
    label_path = next((REPO / config["input"]["search_root"]).glob(config["input"]["label_glob"]))
    reproduction = {"schema_version": "q1-reproduction@1.0", "seed": config["seed"],
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "elapsed_sec": time.time() - started,
        "python": platform.python_version(), "versions": {"numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "av": av.__version__, "opencv": cv2.__version__},
        "config_path": args.config, "config_sha256": sha256(config_path),
        "input_sha256": {"label_table": sha256(label_path),
                         "videos": {sample.source_path.relative_to(REPO).as_posix(): sha256(sample.source_path)
                                    for sample, _ in extracted}},
        "invariants": invariants,
        "outputs": {p.name: sha256(p) for p in outputs}}
    (output / "q1_extraction_reproduction.json").write_text(
        json.dumps(reproduction, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(invariants, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
