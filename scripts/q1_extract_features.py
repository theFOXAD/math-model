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


def write_utf8_lf(path: Path, text: str, *, bom: bool = False) -> None:
    """Write deterministic UTF-8 text without platform newline translation."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    payload = normalized.encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + payload)


def write_csv_lf(frame: pd.DataFrame, path: Path) -> None:
    """Write a spreadsheet-friendly UTF-8 CSV with stable LF bytes."""
    write_utf8_lf(path, frame.to_csv(index=False, lineterminator="\n"), bom=True)


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
                   "injected_missing_audio", "injected_missing_vision", "padding_mask",
                   "observed_text", "observed_audio", "observed_vision"]
    arrays = {key: np.stack([result[key] for _, result in extracted]) for key in tensor_keys}
    np.savez_compressed(output / "q1_features.npz", sample_id=sample_ids, **arrays)
    unaligned_keys = ["audio_unaligned", "vision_unaligned", "valid_audio_unaligned",
                      "valid_vision_unaligned", "audio_time_sec_unaligned",
                      "vision_time_sec_unaligned", "vision_frame_index_src_unaligned"]
    unaligned = {key: np.stack([result[key] for _, result in extracted]) for key in unaligned_keys}
    np.savez_compressed(output / "q1_features_unaligned.npz", sample_id=sample_ids, **unaligned)
    sample_rows, manifest_rows, alignment_rows, quality_rows = [], [], [], []
    for sample, result in extracted:
        sequence_length = len(result["anchors"])
        sample_rows.append({"sample_id": sample.sample_id, "video_id": sample.video_id,
                            "clip_id": sample.clip_id, "text": sample.text,
                            "label_class": sample.label_class, "label_int": sample.label_int,
                            "label_intensity": sample.label_intensity,
                            "sequence_length": sequence_length,
                            "text_missing": result["text_missing"],
                            "face_detection_rate": result["face_detection_rate"],
                            "decoded_audio_duration_sec": result["audio_duration"],
                            "decoded_video_duration_sec": result["video_duration"],
                            "source_path": sample.source_path.relative_to(REPO).as_posix()})
        quality_rows.append({"sample_id": sample.sample_id,
                             "container_declared_duration_sec": result["source_duration"],
                             "container_effective_duration_sec": result["container_effective_duration"],
                             "video_stream_declared_duration_sec": result["video_declared_duration"],
                             "audio_decoded_duration_sec": result["audio_duration"],
                             "video_decoded_duration_sec": result["video_duration"],
                             "alignment_timeline_duration_sec": result["duration"],
                             "declared_minus_decoded_timeline_sec": result["source_duration"] - result["duration"],
                             # Ignore normal AAC/container tail discrepancies below 0.2 s.
                             "container_truncation_suspected": result["source_duration"] - result["duration"] > 0.2,
                             "face_detection_rate": result["face_detection_rate"],
                             "sampled_video_frame_count": result["video_frame_count"]})
        alignment_rows.extend(result["alignments"])
        for modality, dimension, valid_key in [("text", int(config["text"]["dimension"]), "valid_text"),
                                                ("audio", 74, "valid_audio"),
                                                ("vision", 35, "valid_vision")]:
            warnings = "" if result[valid_key].sum() == sequence_length else "invalid_nonpadding_positions"
            manifest_rows.append({"sample_id": sample.sample_id, "modality": modality,
                                  "source_path": sample.source_path.relative_to(REPO).as_posix(),
                                  "container_declared_duration_sec": result["source_duration"],
                                  "container_effective_duration_sec": result["container_effective_duration"],
                                  "decoded_timeline_duration_sec": result["duration"],
                                  "stream_duration_sec": (result["audio_duration"] if modality == "audio" else
                                                          result["video_duration"] if modality == "vision" else
                                                          result["duration"]),
                                  "sample_rate_hz": result["audio_rate"] if modality == "audio" else "n/a",
                                  "source_fps": result["source_fps"] if modality == "vision" else "n/a",
                                  "sampled_fps": float(config["vision"]["sample_fps"]) if modality == "vision" else "n/a",
                                  "raw_shape": "n/a", "feature_dim": dimension,
                                  "native_granularity": ("grouped_transcript_anchor" if modality == "text" else
                                                         f"{float(config['audio']['hop_sec'])*1000:.0f}_ms_frame" if modality == "audio" else
                                                         f"{float(config['vision']['sample_fps']):g}_fps_sample"),
                                  "aligned_granularity": "variable_text_anchor_mean_pool",
                                  "sequence_length": sequence_length,
                                  "valid_position_count": int(result[valid_key].sum()),
                                  "padding_position_count": int(result["padding_mask"].sum()),
                                  "alignment_method": "proportional", "feature_path": "q1_features.npz",
                                  "unaligned_feature_path": ("n/a" if modality == "text" else "q1_features_unaligned.npz"),
                                  "face_detection_rate": result["face_detection_rate"] if modality == "vision" else "n/a",
                                  "feature_sha256": "computed_after_write", "status": "ok" if not warnings else "warning",
                                  "warnings": warnings})
    write_csv_lf(pd.DataFrame(sample_rows), output / "q1_samples.csv")
    write_csv_lf(pd.DataFrame(alignment_rows), output / "q1_alignment.csv")
    write_csv_lf(pd.DataFrame(quality_rows), output / "q1_data_quality.csv")
    feature_hash = sha256(output / "q1_features.npz")
    unaligned_hash = sha256(output / "q1_features_unaligned.npz")
    for row in manifest_rows:
        row["feature_sha256"] = feature_hash
        row["unaligned_feature_sha256"] = ("n/a" if row["modality"] == "text" else unaligned_hash)
    write_csv_lf(pd.DataFrame(manifest_rows), output / "q1_feature_manifest.csv")
    statistic_rows = []
    zero_variance = {}
    for modality in ("text", "audio", "vision"):
        values = arrays[modality][arrays[f"observed_{modality}"]]
        if not len(values):
            zero_variance[modality] = int(arrays[modality].shape[-1])
            for dimension in range(arrays[modality].shape[-1]):
                statistic_rows.append({"modality": modality, "dimension": dimension,
                                       "valid_position_count": 0, "mean": np.nan, "std": np.nan,
                                       "min": np.nan, "max": np.nan, "zero_variance": True})
            continue
        means, stds = values.mean(axis=0), values.std(axis=0)
        zero_variance[modality] = int(np.sum(stds <= 1e-12))
        for dimension, (mean, std, minimum, maximum) in enumerate(
                zip(means, stds, values.min(axis=0), values.max(axis=0))):
            statistic_rows.append({"modality": modality, "dimension": dimension,
                                   "valid_position_count": len(values), "mean": mean, "std": std,
                                   "min": minimum, "max": maximum, "zero_variance": std <= 1e-12})
    write_csv_lf(pd.DataFrame(statistic_rows), output / "q1_feature_statistics.csv")
    write_utf8_lf(output / "q1_normalization.json", json.dumps({"method": "none", "fitted_on": None,
        "reason": "Q1 reports raw descriptors and per-dimension statistics. Train-only normalization is deferred to I01 to avoid leakage.",
        "statistics_path": "q1_feature_statistics.csv", "zero_variance_dimensions": zero_variance},
        ensure_ascii=False, indent=2) + "\n")
    write_utf8_lf(output / "q1_README.md",
        "# Q1 feature bundle\n\n"
        "This directory is a question-1 result, not a frozen I01/I02 exchange bundle.\n\n"
        "- `q1_samples.csv`: labels, transcripts, IDs and sequence lengths.\n"
        "- `q1_features.npz`: aligned-50 float32 `text/audio/vision` tensors, valid/injected/padding masks, and derived `observed_*` masks.\n"
        "- `q1_features_unaligned.npz`: independent audio/vision sequences padded or deterministically resampled to 500 positions, with seconds and source-frame maps.\n"
        "- `q1_alignment.csv`: zero-based anchors mapped to half-open time ranges, 16 kHz sample indexes, 10 fps sampled-frame indexes, source-frame indexes and milliseconds.\n"
        "- `q1_feature_manifest.csv`: per-sample and per-modality quality rows.\n"
        "- `q1_data_quality.csv`: declared versus decodable durations, truncation flags and face-detection rates.\n"
        "- `q1_feature_statistics.csv`: per-dimension valid-position statistics and zero-variance flags.\n"
        "- `q1_normalization.json`: explains why train-fitted normalization is deferred to I01.\n"
        "- `q1_extraction_reproduction.json`: extraction command, versions, hashes and invariant checks.\n"
        "- `../复现清单.json`: the single full-chain reproduction record.\n\n"
        "Load with `numpy.load('q1_features.npz')` and `pandas.read_csv(...)`. All index intervals are left-closed/right-open. "
        "`audio_*_idx` uses 16 kHz samples; `video_*_idx` uses the 10 fps sampled list; `video_*_frame_src` uses decoded source-frame numbers. "
        "Missingness must be read from masks; never infer it from zero-valued features. Empty transcripts use 50 uniform time bins with "
        "`valid_text=False`, `padding_mask=False`, and `alignment_source=uniform_missing_text`.\n",
    )
    mask_keys = ["valid_text", "valid_audio", "valid_vision", "injected_missing_audio",
                 "injected_missing_vision", "padding_mask", "observed_text", "observed_audio", "observed_vision"]
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
        "timeline_sequence_consistent": bool(np.all(
            (~arrays["padding_mask"]).sum(axis=1) == np.asarray([len(result["anchors"]) for _, result in extracted]))),
        "mask_dtype_bool": all(arrays[key].dtype == np.bool_ for key in mask_keys),
        "valid_excludes_padding": all(not np.any(arrays[key] & arrays["padding_mask"])
                                      for key in ("valid_text", "valid_audio", "valid_vision")),
        "injected_missing_excludes_padding": all(not np.any(arrays[key] & arrays["padding_mask"])
                                                  for key in ("injected_missing_audio", "injected_missing_vision")),
        "observed_mask_definition": all(np.array_equal(
            arrays[f"observed_{modality}"], arrays[f"valid_{modality}"] & ~arrays["padding_mask"]
            & (~arrays[f"injected_missing_{modality}"] if modality in ("audio", "vision") else True))
            for modality in ("text", "audio", "vision")),
        "alignment_interval_legal": all(r["end_sec"] > r["start_sec"] >= 0 for r in alignment_rows),
        "alignment_contiguous_and_bounded": interval_coverage,
        "alignment_covers_decoded_timeline": all(abs(
            result["duration"] - max(result["audio_duration"], result["video_duration"])) <= 1e-7
            for _, result in extracted),
        "video_frames_respect_half_open_intervals": all(
            pd.isna(row["video_start_ms"])
            or (row["video_start_ms"] + 1e-7 >= 1000.0 * row["start_sec"]
                and row["video_last_pts_ms"] < 1000.0 * row["end_sec"] + 1e-7)
            for row in alignment_rows),
        "unaligned_mask_dtype_bool": all(unaligned[key].dtype == np.bool_
                                           for key in ("valid_audio_unaligned", "valid_vision_unaligned")),
    }
    if not all(value for key, value in invariants.items() if key not in {"sample_count", "shape_text", "shape_audio", "shape_vision"}):
        raise RuntimeError(f"output invariant failed: {invariants}")
    data_output_names = ["q1_features.npz", "q1_features_unaligned.npz", "q1_samples.csv",
                         "q1_alignment.csv", "q1_feature_manifest.csv", "q1_data_quality.csv",
                         "q1_feature_statistics.csv", "q1_normalization.json", "q1_README.md"]
    outputs = [output / name for name in data_output_names]
    label_path = next((REPO / config["input"]["search_root"]).glob(config["input"]["label_glob"]))
    portable_command = ["python", "-X", "utf8", "scripts/q1_extract_features.py", *sys.argv[1:]]
    reproduction = {"schema_version": "q1-reproduction@1.1", "seed": config["seed"],
        "command": subprocess.list2cmdline(portable_command), "command_argv": portable_command,
        "elapsed_sec": time.time() - started,
        "python": platform.python_version(), "versions": {"numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "av": av.__version__, "opencv": cv2.__version__},
        "config_path": args.config, "config_sha256": sha256(config_path),
        "input_sha256": {"label_table": sha256(label_path),
                         "videos": {sample.source_path.relative_to(REPO).as_posix(): sha256(sample.source_path)
                                    for sample, _ in extracted}},
        "invariants": invariants,
        "outputs": {p.name: sha256(p) for p in outputs}}
    write_utf8_lf(output / "q1_extraction_reproduction.json",
                  json.dumps(reproduction, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(invariants, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
