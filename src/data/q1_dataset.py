"""Attachment-1 inventory and label loading for question 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Q1Sample:
    sample_id: str
    video_id: str
    clip_id: int
    text: str
    label_intensity: float
    label_class: str
    label_int: int
    source_path: Path


def discover_samples(search_root: Path, label_glob: str, video_glob: str) -> list[Q1Sample]:
    label_paths = sorted(search_root.glob(label_glob))
    if len(label_paths) != 1:
        raise RuntimeError(f"expected exactly one label-100.xlsx, found {len(label_paths)}")
    label_path = label_paths[0]
    table = pd.read_excel(label_path)
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"missing label columns: {sorted(missing)}")

    video_paths = sorted(label_path.parent.glob(video_glob))
    video_map: dict[str, Path] = {}
    for path in video_paths:
        key = f"{path.parent.name}$_${path.stem}"
        if key in video_map:
            raise ValueError(f"duplicate video key: {key}")
        video_map[key] = path

    samples: list[Q1Sample] = []
    seen: set[str] = set()
    for row in table.itertuples(index=False):
        clip_id = int(row.clip_id)
        sample_id = f"{row.video_id}$_${clip_id}"
        if sample_id in seen:
            raise ValueError(f"duplicate label key: {sample_id}")
        seen.add(sample_id)
        if sample_id not in video_map:
            raise FileNotFoundError(f"video missing for {sample_id}")
        label_class = str(row.annotation)
        label_encoding = {"Negative": 0, "Neutral": 1, "Positive": 2}
        if label_class not in label_encoding:
            raise ValueError(f"unknown label class for {sample_id}: {label_class}")
        samples.append(
            Q1Sample(
                sample_id=sample_id,
                video_id=str(row.video_id),
                clip_id=clip_id,
                text="" if pd.isna(row.text) else str(row.text).strip(),
                label_intensity=float(row.label),
                label_class=label_class,
                label_int=label_encoding[label_class],
                source_path=video_map[sample_id],
            )
        )
    orphans = sorted(set(video_map).difference(seen))
    if orphans:
        raise ValueError(f"orphan videos: {orphans[:5]}")
    return samples
