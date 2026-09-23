"""Generate the nine Q1 candidate figures plus the reproducible method flowchart."""

from __future__ import annotations

import argparse
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
from PIL import Image, ImageOps
import yaml

FIGURE_TOOLS = Path(r"C:\Users\FOXAD\.codex\skills\math-modeling\tools\figure\scripts")
sys.path.insert(0, str(FIGURE_TOOLS))
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout  # noqa: E402

COLORS = {"text": "#0072B2", "audio": "#E69F00", "vision": "#CC79A7",
          "Negative": "#D55E00", "Neutral": "#999999", "Positive": "#0072B2"}


def export(fig, base: Path, qa_dir: Path) -> list[dict]:
    svg_path = base.with_suffix(".svg")
    fig.savefig(svg_path, metadata={"Date": None})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(base.with_suffix(".png"), dpi=300, metadata={"Date": None})
    with Image.open(base.with_suffix(".png")) as rendered:
        ImageOps.grayscale(rendered).save(qa_dir / f"{base.name}_grayscale.png")
    issues = audit_layout(fig)
    plt.close(fig)
    return [{"severity": getattr(issue, "severity", "unknown"), "message": str(issue)} for issue in issues]


def style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.65)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-dir", default="results/q1")
    parser.add_argument("--figure-dir", default="figures/q1")
    args = parser.parse_args()
    setup_style(journal="general", lang="en")
    plt.rcParams.update({"figure.figsize": (7.2, 4.5), "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "svg.fonttype": "none", "svg.hashsalt": "q1-20260923"})
    result_dir = REPO / args.q1_dir
    figure_dir = REPO / args.figure_dir
    qa_dir = figure_dir.parent / f"{figure_dir.name}_qa"
    figure_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(result_dir / "q1_samples.csv")
    manifest = pd.read_csv(result_dir / "q1_feature_manifest.csv")
    alignment = pd.read_csv(result_dir / "q1_alignment.csv")
    overlap = pd.read_csv(result_dir / "q1_overlap_similarity.csv")
    features = np.load(result_dir / "q1_features.npz")
    durations = manifest[manifest.modality == "text"][["sample_id", "source_duration_sec"]]
    profile = samples.merge(durations, on="sample_id")
    profile["word_count"] = profile.text.str.split().str.len()
    audit = []
    def save(fig, name: str) -> None:
        nonlocal audit
        audit += export(fig, figure_dir / name, qa_dir)

    fig, ax = plt.subplots()
    counts = samples.label_class.value_counts().reindex(["Negative", "Neutral", "Positive"])
    ax.barh(counts.index, counts.values, color=[COLORS[x] for x in counts.index])
    for y, value in enumerate(counts): ax.text(value + 1, y, f"n={value}", va="center")
    style_ax(ax, "Samples (count)", "Sentiment class", "Attachment-1 class composition")
    ax.set_xlim(0, counts.max() * 1.18)
    save(fig, "raw_q1_class_distribution")

    fig, ax = plt.subplots()
    ax.hist(profile.source_duration_sec, bins=12, color=COLORS["text"], edgecolor="white")
    ax.axvline(profile.source_duration_sec.median(), color="#D55E00", linestyle="--", label="Median")
    ax.legend(frameon=False)
    style_ax(ax, "Video duration (s)", "Samples (count)", "Video-duration distribution (n=100)")
    save(fig, "raw_q1_duration_histogram")

    fig, ax = plt.subplots()
    for label, group in profile.groupby("label_class"):
        ax.scatter(group.word_count, group.source_duration_sec, s=20, alpha=0.72,
                   color=COLORS[label], label=label)
    ax.legend(frameon=False)
    style_ax(ax, "Transcript length (words)", "Video duration (s)", "Transcript length versus duration")
    save(fig, "raw_q1_words_duration_scatter")

    fig, ax = plt.subplots()
    ax.hist(samples.sequence_length, bins=np.arange(4.5, 51.5, 3), color=COLORS["audio"], edgecolor="white")
    style_ax(ax, "Aligned anchors (count)", "Samples (count)", "Text-anchor sequence lengths after grouping")
    save(fig, "process_q1_anchor_lengths")

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
    save(fig, "process_q1_validity_boxstrip")

    typical = samples.iloc[(samples.sequence_length - samples.sequence_length.median()).abs().argmin()].sample_id
    rows = alignment[alignment.sample_id == typical]
    typical_index = {str(value): i for i, value in enumerate(features["sample_id"])}[typical]
    fig, (ax, feature_ax) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                                         gridspec_kw={"height_ratios": [1.15, 1.0]})
    lane_specs = [("Text anchor", "text", 2.0), ("Audio segment", "audio", 1.0),
                  ("Video frames", "vision", 0.0)]
    for label, modality, y in lane_specs:
        valid = features[f"valid_{modality}"][typical_index]
        for position, (_, row) in enumerate(rows.iterrows()):
            ax.broken_barh([(row.start_sec, row.end_sec - row.start_sec)], (y, 0.72),
                           facecolors=COLORS[modality] if valid[position] else "#BDBDBD",
                           edgecolors="white", linewidth=0.5)
    ax.set_yticks([2.36, 1.36, .36], [item[0] for item in lane_specs])
    ax.set_ylim(-.08, 2.85)
    safe_typical = typical.replace("$_$", " / ")
    style_ax(ax, "", "Aligned streams", f"Typical text–audio–video correspondence: {safe_typical}")
    centers = ((rows.start_sec + rows.end_sec) / 2).to_numpy()
    for modality, marker, line in [("text", "o", "-"), ("audio", "s", "--"), ("vision", "^", ":")]:
        norms = np.linalg.norm(features[modality][typical_index], axis=1)[:len(rows)]
        valid = features[f"valid_{modality}"][typical_index][:len(rows)]
        scale = max(float(norms[valid].max()) if np.any(valid) else 0.0, 1e-12)
        feature_ax.plot(centers, np.where(valid, norms / scale, np.nan), marker=marker,
                        linestyle=line, linewidth=1, markersize=3, color=COLORS[modality],
                        label=modality.title())
    feature_ax.legend(frameon=False, ncols=3)
    style_ax(feature_ax, "Time (s)", "Normalized feature norm", "Three modality features at shared anchors")
    save(fig, "process_q1_typical_timeline")

    fig, ax = plt.subplots()
    coverage = manifest.groupby("modality").apply(lambda x: x.valid_position_count.sum() /
                                                   x.sequence_length.sum(), include_groups=False).reindex(["text", "audio", "vision"])
    ax.bar(["Text", "Audio", "Vision"], coverage, color=[COLORS[x] for x in coverage.index])
    for x, value in enumerate(coverage): ax.text(x, value + 0.015, f"{value:.1%}", ha="center")
    style_ax(ax, "Modality", "Valid-anchor coverage", "Q1 full-sample feature coverage")
    ax.set_ylim(0, 1.1)
    save(fig, "result_q1_coverage")

    individual = overlap[overlap.sample_id != "__aggregate__"]
    fig, ax = plt.subplots()
    rng = np.random.default_rng(20260923)
    cka_counts = []
    for i, modality in enumerate(("text", "audio", "vision")):
        values = individual[individual.modality == modality].centered_linear_cka.dropna()
        cka_counts.append(len(values))
        ax.scatter(rng.normal(i, 0.05, len(values)), values, s=22, alpha=0.72, color=COLORS[modality])
        ax.scatter(i, values.mean(), marker="D", s=42, color="black", zorder=3)
    ax.set_xticks(range(3), [f"{name}\nn={count}" for name, count in zip(("Text", "Audio", "Vision"), cka_counts)])
    style_ax(ax, "Modality", "Centered linear CKA", "Valid estimates among 18 shared samples (diamond=mean)")
    ax.set_ylim(0, 1.02)
    save(fig, "result_q1_overlap_cka")

    fig, ax = plt.subplots()
    data = [individual[individual.modality == modality].procrustes_residual.dropna()
            for modality in ("text", "audio", "vision")]
    box = ax.boxplot(data, tick_labels=[f"{name}\nn={len(values)}" for name, values in
                                       zip(("Text", "Audio", "Vision"), data)],
                     patch_artist=True, showfliers=False)
    for patch, modality in zip(box["boxes"], ("text", "audio", "vision")): patch.set_facecolor(COLORS[modality])
    for i, values in enumerate(data, 1): ax.scatter(rng.normal(i, 0.04, len(values)), values, s=16, alpha=0.55, color="#333333")
    style_ax(ax, "Modality", "Procrustes residual (lower is closer)", "Cross-representation residuals (box=IQR; points=samples)")
    save(fig, "result_q1_overlap_procrustes")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_axis_off()
    nodes = [(0.36, .94, "Attachment-1\n100 videos + labels"), (0.36, .81, "ID and stream validation"),
             (0.36, .68, "Text anchors + proportional\ntime boundaries (T <= 50)"),
             (0.13, .50, "Text hashing\n128-D"), (0.36, .50, "Audio log-Mel/MFCC\n74-D"),
             (0.59, .50, "Vision appearance/motion\n35-D"),
             (0.36, .34, "Explicit masks + interval invariants"),
             (0.36, .19, "Q1 CSV/NPZ"), (0.82, .68, "Attachment-2\n18 shared IDs"),
             (0.78, .36, "Mask-based CKA +\nProcrustes audit"),
             (0.56, .05, "Reports + figures +\nfull-chain hashes")]
    edges = [(0,1),(1,2),(2,3),(2,4),(2,5),(3,6),(4,6),(5,6),(6,7),(7,9),(8,9),(7,10),(9,10)]
    for source, target in edges:
        x1,y1,_=nodes[source]; x2,y2,_=nodes[target]
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), zorder=1,
                    arrowprops={"arrowstyle":"->","color":"#34495E", "shrinkA":28, "shrinkB":28})
    for x, y, label in nodes:
        ax.add_patch(FancyBboxPatch((x-.11, y-.045), .22, .09, boxstyle="round,pad=0.012",
                                    facecolor="#EAF2F8", edgecolor="#34495E", linewidth=1, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=7, zorder=3)
    ax.set_xlim(0,1); ax.set_ylim(-.02,1)
    save(fig, "flow_overall_model")

    figure_contracts = {
        "schema_version": "q1-figure-contracts@1.0",
        "shared_style": {"language": "English", "png_dpi": 300,
                         "formats": ["svg", "png"], "grayscale_qa_dir": f"../{qa_dir.name}"},
        "figures": [
            {"name": "raw_q1_class_distribution", "category": "raw", "claim": "Attachment-1 class composition",
             "source": "q1_samples.csv", "evidence": "class counts and n labels"},
            {"name": "raw_q1_duration_histogram", "category": "raw", "claim": "clip-duration distribution",
             "source": "q1_feature_manifest.csv", "evidence": "histogram and median"},
            {"name": "raw_q1_words_duration_scatter", "category": "raw", "claim": "text length versus duration",
             "source": "q1_samples.csv + q1_feature_manifest.csv", "evidence": "sample-level scatter"},
            {"name": "process_q1_anchor_lengths", "category": "process", "claim": "post-grouping anchor lengths",
             "source": "q1_samples.csv", "evidence": "sequence-length histogram"},
            {"name": "process_q1_validity_boxstrip", "category": "process", "claim": "per-modality valid-anchor ratios",
             "source": "q1_feature_manifest.csv", "evidence": "box and all sample points"},
            {"name": "process_q1_typical_timeline", "category": "process", "claim": "text, audio and video share proportional anchors",
             "source": "q1_alignment.csv + q1_features.npz", "evidence": "three aligned lanes and feature-norm curves"},
            {"name": "result_q1_coverage", "category": "result", "claim": "full-sample valid-anchor coverage",
             "source": "q1_feature_manifest.csv", "evidence": "three labeled coverage bars"},
            {"name": "result_q1_overlap_cka", "category": "result", "claim": "representation agreement on shared samples",
             "source": "q1_overlap_similarity.csv", "evidence": "valid CKA estimates with per-modality n"},
            {"name": "result_q1_overlap_procrustes", "category": "result", "claim": "cross-representation residuals",
             "source": "q1_overlap_similarity.csv", "evidence": "boxplots, points and per-modality n"},
            {"name": "flow_overall_model", "category": "flow", "claim": "implemented extraction and audit order",
             "source": "scripts/q1_*.py", "evidence": "anchors precede pooling; 18-ID audit branch shown"},
        ],
    }
    (figure_dir / "figure_contracts.yaml").write_text(
        yaml.safe_dump(figure_contracts, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = {"sample_count": int(len(samples)), "label_counts": counts.to_dict(),
               "duration_sec": profile.source_duration_sec.describe().to_dict(),
               "sequence_length": samples.sequence_length.describe().to_dict(),
               "coverage": coverage.to_dict(), "overlap_count": 18,
               "aggregate_similarity": overlap[overlap.sample_id == "__aggregate__"].to_dict("records"),
               "figure_count": 10, "grayscale_qa_count": 10,
               "layout_issue_count": len(audit), "layout_issues": audit}
    (result_dir / "q1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate = overlap[overlap.sample_id == "__aggregate__"].set_index("modality")
    invalid = individual[individual.status != "ok"]
    report = f"""# E题第一问：多模态特征提取与时序对齐结果

## 结论

附件1的 100 条标签与 100 个 MP4 严格一一对应，全部进入结果集。三模态统一为最长 50 个文本锚点：文本、音频、视觉张量形状分别为 `100×50×128`、`100×50×74`、`100×50×35`，均为 `float32`；有效、注入缺失与填充状态由独立布尔掩码表达。

容器平均时长 {profile.source_duration_sec.mean():.3f} 秒（范围 {profile.source_duration_sec.min():.3f}--{profile.source_duration_sec.max():.3f} 秒），文本锚点平均 {samples.sequence_length.mean():.2f} 个（范围 {samples.sequence_length.min()}--{samples.sequence_length.max()}）。按非填充锚点统计，有效覆盖率为：文本 {coverage.text:.1%}、音频 {coverage.audio:.1%}、视觉 {coverage.vision:.2%}。未覆盖位置由显式 `valid_*` 掩码标记，不从零值特征推断缺失。

## 方法与验证

1. 文本使用确定性 SHA-256 signed hashing 生成 128 维描述符；音频使用 log-Mel、MFCC、差分与统计生成 74 维描述符；视觉使用 HSV、脸部几何、运动、亮度与边缘生成 35 维描述符。
2. 以文本锚点字符数为权重生成连续半开比例区间，明确标记 `method=proportional,is_fallback=true`，不宣称 forced alignment。
3. 100 条样本全部完成；特征均有限；mask 为 bool 且不与 padding 冲突；时间区间覆盖音频、视频流与容器时长的共同有界区间，索引连续且有界。

## 与附件2的 18 条重叠样本

以本结果显式 mask 和附件2对齐 mask 选择位置。总体中心化线性 CKA 为文本 {aggregate.loc['text'].centered_linear_cka:.4f}、音频 {aggregate.loc['audio'].centered_linear_cka:.4f}、视觉 {aggregate.loc['vision'].centered_linear_cka:.4f}；对应 Procrustes 残差为 {aggregate.loc['text'].procrustes_residual:.4f}、{aggregate.loc['audio'].procrustes_residual:.4f}、{aggregate.loc['vision'].procrustes_residual:.4f}。逐样本 54 个模态比较中，{len(invalid)} 条为 `insufficient_variation`；具体原因记录在 `status_reason`。

## 产物与复现

- 核心数据：`q1_samples.csv`、`q1_features.npz`、`q1_alignment.csv`、`q1_feature_manifest.csv`；
- 审计与证据：`q1_overlap_similarity.csv`、`q1_extraction_reproduction.json`、`q1_overlap_reproduction.json`、`q1_checksums.sha256`；
- 图表：9 张候选数据图及 1 张流程图，均提供 SVG、300 DPI PNG、合同与灰度核验图。

```powershell
.venv\\Scripts\\python.exe -X utf8 scripts\\q1_run_all.py --config config/q1_features.yaml --output-dir results/q1 --figure-dir figures/q1 --status validated
```

本成果仍是本地 Q1 结果，不是冻结的 I01/I02 交换包；跨用户发布须另行完成版本化握手与 wc 审批。
"""
    (result_dir / "q1_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"figure_count": 10, "layout_issue_count": len(audit), "coverage": coverage.to_dict()}))


if __name__ == "__main__":
    main()
