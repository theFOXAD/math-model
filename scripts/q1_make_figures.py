"""Generate the nine Q1 candidate figures plus the reproducible method flowchart."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".matplotlib-cache"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

FIGURE_TOOLS = Path(r"C:\Users\FOXAD\.codex\skills\math-modeling\tools\figure\scripts")
sys.path.insert(0, str(FIGURE_TOOLS))
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout  # noqa: E402

COLORS = {"text": "#0072B2", "audio": "#E69F00", "vision": "#CC79A7",
          "Negative": "#D55E00", "Neutral": "#999999", "Positive": "#0072B2"}


def export(fig, base: Path) -> list[dict]:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    issues = audit_layout(fig)
    plt.close(fig)
    return [{"severity": getattr(issue, "severity", "unknown"), "message": str(issue)} for issue in issues]


def style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.65)


def main() -> None:
    setup_style(journal="general", lang="en")
    plt.rcParams.update({"figure.figsize": (7.2, 4.5), "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8, "svg.fonttype": "none"})
    result_dir = REPO / "results/q1"
    figure_dir = REPO / "figures/q1"
    figure_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(result_dir / "q1_samples.csv")
    manifest = pd.read_csv(result_dir / "q1_feature_manifest.csv")
    alignment = pd.read_csv(result_dir / "q1_alignment.csv")
    overlap = pd.read_csv(result_dir / "q1_overlap_similarity.csv")
    features = np.load(result_dir / "q1_features.npz")
    durations = manifest[manifest.modality == "text"][["sample_id", "source_duration_sec"]]
    profile = samples.merge(durations, on="sample_id")
    profile["word_count"] = profile.text.str.split().str.len()
    audit = []

    fig, ax = plt.subplots()
    counts = samples.label_class.value_counts().reindex(["Negative", "Neutral", "Positive"])
    ax.barh(counts.index, counts.values, color=[COLORS[x] for x in counts.index])
    for y, value in enumerate(counts): ax.text(value + 1, y, f"n={value}", va="center")
    style_ax(ax, "Samples (count)", "Sentiment class", "Attachment-1 class composition")
    ax.set_xlim(0, counts.max() * 1.18)
    audit += export(fig, figure_dir / "raw_q1_class_distribution")

    fig, ax = plt.subplots()
    ax.hist(profile.source_duration_sec, bins=12, color=COLORS["text"], edgecolor="white")
    ax.axvline(profile.source_duration_sec.median(), color="#D55E00", linestyle="--", label="Median")
    ax.legend(frameon=False)
    style_ax(ax, "Video duration (s)", "Samples (count)", "Video-duration distribution (n=100)")
    audit += export(fig, figure_dir / "raw_q1_duration_histogram")

    fig, ax = plt.subplots()
    for label, group in profile.groupby("label_class"):
        ax.scatter(group.word_count, group.source_duration_sec, s=20, alpha=0.72,
                   color=COLORS[label], label=label)
    ax.legend(frameon=False)
    style_ax(ax, "Transcript length (words)", "Video duration (s)", "Transcript length versus duration")
    audit += export(fig, figure_dir / "raw_q1_words_duration_scatter")

    fig, ax = plt.subplots()
    ax.hist(samples.sequence_length, bins=np.arange(4.5, 51.5, 3), color=COLORS["audio"], edgecolor="white")
    style_ax(ax, "Aligned anchors (count)", "Samples (count)", "Text-anchor sequence lengths after grouping")
    audit += export(fig, figure_dir / "process_q1_anchor_lengths")

    fig, ax = plt.subplots()
    data = [manifest[manifest.modality == modality].valid_position_count / samples.set_index("sample_id").loc[
        manifest[manifest.modality == modality].sample_id, "sequence_length"].to_numpy()
            for modality in ("text", "audio", "vision")]
    box = ax.boxplot(data, tick_labels=["Text", "Audio", "Vision"], patch_artist=True, showfliers=False)
    for patch, modality in zip(box["boxes"], ("text", "audio", "vision")): patch.set_facecolor(COLORS[modality])
    rng = np.random.default_rng(20260923)
    for i, values in enumerate(data, 1): ax.scatter(rng.normal(i, 0.035, len(values)), values, s=8, alpha=0.28, color="#333333")
    style_ax(ax, "Modality", "Valid anchors / sequence length", "Per-sample modality validity (box=IQR; n=100)")
    ax.set_ylim(-0.02, 1.05)
    audit += export(fig, figure_dir / "process_q1_validity_boxstrip")

    typical = samples.iloc[(samples.sequence_length - samples.sequence_length.median()).abs().argmin()].sample_id
    rows = alignment[alignment.sample_id == typical]
    fig, ax = plt.subplots()
    for _, row in rows.iterrows():
        ax.broken_barh([(row.start_sec, row.end_sec - row.start_sec)], (0.2, 0.6),
                       facecolors=COLORS["text"], edgecolors="white", linewidth=0.6)
    ax.set_yticks([0.5], ["Text anchors"])
    safe_typical = typical.replace("$_$", " / ")
    style_ax(ax, "Time (s)", "Aligned stream", f"Typical proportional alignment: {safe_typical}")
    audit += export(fig, figure_dir / "process_q1_typical_timeline")

    fig, ax = plt.subplots()
    coverage = manifest.groupby("modality").apply(lambda x: x.valid_position_count.sum() /
                                                   x.sequence_length.sum(), include_groups=False).reindex(["text", "audio", "vision"])
    ax.bar(["Text", "Audio", "Vision"], coverage, color=[COLORS[x] for x in coverage.index])
    for x, value in enumerate(coverage): ax.text(x, value + 0.015, f"{value:.1%}", ha="center")
    style_ax(ax, "Modality", "Valid-anchor coverage", "Q1 full-sample feature coverage")
    ax.set_ylim(0, 1.1)
    audit += export(fig, figure_dir / "result_q1_coverage")

    individual = overlap[overlap.sample_id != "__aggregate__"]
    fig, ax = plt.subplots()
    rng = np.random.default_rng(20260923)
    for i, modality in enumerate(("text", "audio", "vision")):
        values = individual[individual.modality == modality].centered_linear_cka.dropna()
        ax.scatter(rng.normal(i, 0.05, len(values)), values, s=22, alpha=0.72, color=COLORS[modality])
        ax.scatter(i, values.mean(), marker="D", s=42, color="black", zorder=3)
    ax.set_xticks(range(3), ["Text", "Audio", "Vision"])
    style_ax(ax, "Modality", "Centered linear CKA", "Representation agreement on 18 shared samples (diamond=mean)")
    ax.set_ylim(0, 1.02)
    audit += export(fig, figure_dir / "result_q1_overlap_cka")

    fig, ax = plt.subplots()
    data = [individual[individual.modality == modality].procrustes_residual.dropna()
            for modality in ("text", "audio", "vision")]
    box = ax.boxplot(data, tick_labels=["Text", "Audio", "Vision"], patch_artist=True, showfliers=False)
    for patch, modality in zip(box["boxes"], ("text", "audio", "vision")): patch.set_facecolor(COLORS[modality])
    for i, values in enumerate(data, 1): ax.scatter(rng.normal(i, 0.04, len(values)), values, s=16, alpha=0.55, color="#333333")
    style_ax(ax, "Modality", "Procrustes residual (lower is closer)", "Cross-representation residuals (box=IQR; points=samples)")
    audit += export(fig, figure_dir / "result_q1_overlap_procrustes")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_axis_off()
    nodes = [(0.5, .92, "Attachment-1\n100 videos + labels"), (0.5, .76, "ID and stream validation"),
             (0.2, .57, "Text hashing\n128-D"), (0.5, .57, "Audio log-Mel/MFCC\n74-D"),
             (0.8, .57, "Vision appearance/motion\n35-D"), (0.5, .38, "Proportional text anchors\nT <= 50"),
             (0.5, .20, "Mask and interval invariants"), (0.5, .05, "Q1 CSV/NPZ + hashes")]
    for x, y, label in nodes:
        ax.add_patch(FancyBboxPatch((x-.13, y-.045), .26, .09, boxstyle="round,pad=0.012",
                                    facecolor="#EAF2F8", edgecolor="#34495E", linewidth=1))
        ax.text(x, y, label, ha="center", va="center", fontsize=7)
    edges = [(0,1),(1,2),(1,3),(1,4),(2,5),(3,5),(4,5),(5,6),(6,7)]
    for source, target in edges:
        x1,y1,_=nodes[source]; x2,y2,_=nodes[target]
        ax.annotate("", xy=(x2,y2+.055), xytext=(x1,y1-.055), arrowprops={"arrowstyle":"->","color":"#34495E"})
    ax.set_xlim(0,1); ax.set_ylim(-.02,1)
    audit += export(fig, figure_dir / "flow_overall_model")

    summary = {"sample_count": int(len(samples)), "label_counts": counts.to_dict(),
               "duration_sec": profile.source_duration_sec.describe().to_dict(),
               "sequence_length": samples.sequence_length.describe().to_dict(),
               "coverage": coverage.to_dict(), "overlap_count": 18,
               "aggregate_similarity": overlap[overlap.sample_id == "__aggregate__"].to_dict("records"),
               "figure_count": 10, "layout_issue_count": len(audit), "layout_issues": audit}
    (result_dir / "q1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure_count": 10, "layout_issue_count": len(audit), "coverage": coverage.to_dict()}))


if __name__ == "__main__":
    main()
