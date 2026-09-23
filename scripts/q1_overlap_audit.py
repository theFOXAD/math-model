"""Compare Q1 descriptors with Attachment-2 aligned features on the 18 shared IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centered_cka(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k, ell = x @ x.T, y @ y.T
    denominator = np.linalg.norm(k) * np.linalg.norm(ell)
    return None if denominator <= EPS else float(np.sum(k * ell) / denominator)


def procrustes_residual(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    if len(x) < 2:
        return None, 0
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    ux, sx, _ = np.linalg.svd(x, full_matrices=False)
    uy, sy, _ = np.linalg.svd(y, full_matrices=False)
    rank = min(len(x) - 1, x.shape[1], y.shape[1], int(np.sum(sx > EPS)), int(np.sum(sy > EPS)))
    if rank < 1:
        return None, 0
    zx, zy = ux[:, :rank] * sx[:rank], uy[:, :rank] * sy[:rank]
    u, _, vt = np.linalg.svd(zx.T @ zy, full_matrices=False)
    rotation = u @ vt
    return float(np.linalg.norm(zx @ rotation - zy) / (np.linalg.norm(zy) + EPS)), rank


def metric_row(sample_id: str, split: str, modality: str, x: np.ndarray, y: np.ndarray,
               valid: np.ndarray) -> dict:
    xv, yv = x[valid].astype(np.float64), y[valid].astype(np.float64)
    cka = centered_cka(xv, yv)
    residual, rank = procrustes_residual(xv, yv)
    status = "ok" if cka is not None and residual is not None else "insufficient_variation"
    if int(valid.sum()) < 2:
        reason = "fewer_than_two_mask-valid_positions"
    elif np.linalg.norm(xv - xv.mean(axis=0, keepdims=True)) <= EPS:
        reason = "q1_representation_constant"
    elif np.linalg.norm(yv - yv.mean(axis=0, keepdims=True)) <= EPS:
        reason = "reference_representation_constant"
    else:
        reason = "none"
    return {"sample_id": sample_id, "reference_split": split, "modality": modality,
            "common_position_count": int(valid.sum()), "common_rank": rank,
            "centered_linear_cka": cka, "procrustes_residual": residual,
            "status": status, "status_reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-dir", default="results/q1")
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    q1_dir = REPO / args.q1_dir
    reference_path = Path(args.reference) if args.reference else next((REPO / "data/raw/extracted").rglob("aligned_50.pkl"))
    ours = np.load(q1_dir / "q1_features.npz")
    our_index = {str(sample_id): index for index, sample_id in enumerate(ours["sample_id"])}
    with reference_path.open("rb") as stream:
        reference = pickle.load(stream)
    reference_index = {sample_id: (split, index) for split, values in reference.items()
                       for index, sample_id in enumerate(values["id"])}
    overlap = sorted(set(our_index).intersection(reference_index))
    if len(overlap) != 18:
        raise RuntimeError(f"expected 18 overlapping IDs, found {len(overlap)}")
    rows: list[dict] = []
    aggregate: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        modality: ([], []) for modality in ("text", "audio", "vision")}
    for sample_id in overlap:
        oi = our_index[sample_id]
        split, ri = reference_index[sample_id]
        # Attachment-2 stores its authoritative alignment mask in text_bert[:, 1].
        # Q1 availability comes only from explicit masks; zero-valued features are legal data.
        reference_mask = reference[split]["text_bert"][ri, 1].astype(bool)
        for modality in ("text", "audio", "vision"):
            x = ours[modality][oi]
            y = reference[split][modality][ri]
            valid = (ours[f"valid_{modality}"][oi].astype(bool)
                     & ~ours["padding_mask"][oi].astype(bool) & reference_mask)
            rows.append(metric_row(sample_id, split, modality, x, y, valid))
            aggregate[modality][0].append(x[valid])
            aggregate[modality][1].append(y[valid])
    for modality, (xs, ys) in aggregate.items():
        x, y = np.concatenate(xs), np.concatenate(ys)
        rows.append(metric_row("__aggregate__", "mixed", modality, x, y,
                               np.ones(len(x), dtype=bool)))
    output_path = q1_dir / "q1_overlap_similarity.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    reproduction = {"schema_version": "q1-overlap-reproduction@1.0",
                    "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
                    "python": platform.python_version(), "numpy": np.__version__,
                    "q1_features_sha256": sha256(q1_dir / "q1_features.npz"),
                    "reference_path": reference_path.relative_to(REPO).as_posix(),
                    "reference_sha256": sha256(reference_path), "overlap_count": len(overlap),
                    "output_sha256": sha256(output_path)}
    (q1_dir / "q1_overlap_reproduction.json").write_text(
        json.dumps(reproduction, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overlap_count": len(overlap), "row_count": len(rows),
                      "status_counts": pd.Series([row["status"] for row in rows]).value_counts().to_dict()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
