"""Generate traceable Q1 figures and a concise Chinese result report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".matplotlib-cache"))

import av
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import yaml

COLORS = {"text": "#0072B2", "audio": "#E69F00", "vision": "#CC79A7",
          "Negative": "#D55E00", "Neutral": "#777777", "Positive": "#0072B2"}


def setup_style() -> None:
    """Repository-local, portable publication style."""
    plt.rcParams.update({"figure.figsize": (7.2, 4.5), "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.unicode_minus": False, "svg.fonttype": "none",
                         "svg.hashsalt": "q1-review-remediation", "savefig.bbox": "tight"})


def export(fig, base: Path, qa_dir: Path) -> list[dict]:
    fig.tight_layout()
    fig.canvas.draw()
    issues = []
    for index, axis in enumerate(fig.axes):
        box = axis.get_position()
        if box.width <= 0 or box.height <= 0 or box.x0 < 0 or box.y0 < 0 or box.x1 > 1 or box.y1 > 1:
            issues.append({"severity": "error", "message": f"invalid axes bounds at index {index}: {box.bounds}"})
    svg_path = base.with_suffix(".svg")
    fig.savefig(svg_path, metadata={"Date": None})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    png_path = base.with_suffix(".png")
    fig.savefig(png_path, dpi=300, metadata={"Date": None})
    with Image.open(png_path) as rendered:
        gray = ImageOps.grayscale(rendered)
        gray.save(qa_dir / f"{base.name}_grayscale.png")
        if rendered.width < 600 or rendered.height < 300:
            issues.append({"severity": "error", "message": f"render too small: {rendered.size}"})
        if gray.getextrema()[0] == gray.getextrema()[1]:
            issues.append({"severity": "error", "message": "render is visually constant"})
    if not fig.axes:
        issues.append({"severity": "error", "message": "figure has no axes"})
    plt.close(fig)
    return issues


def style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.65)


def representation_change(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cosine change to the preceding valid representation."""
    output = np.full(len(values), np.nan, dtype=np.float64)
    previous = None
    for index, (value, is_valid) in enumerate(zip(values, valid)):
        if not is_valid:
            continue
        if previous is None:
            output[index] = 0.0
        else:
            denominator = np.linalg.norm(previous) * np.linalg.norm(value)
            output[index] = 1.0 - float(previous @ value / max(denominator, 1e-12))
        previous = value
    return output


def decode_selected_frames(path: Path, indexes: list[int]) -> dict[int, np.ndarray]:
    wanted = set(index for index in indexes if index >= 0)
    frames: dict[int, np.ndarray] = {}
    if not wanted:
        return frames
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                frames[index] = frame.to_ndarray(format="rgb24")
            if len(frames) == len(wanted) or index > max(wanted):
                break
    return frames


def decode_audio_waveform(path: Path, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    rate = 0
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.empty(0), np.empty(0)
        for frame in container.decode(audio=0):
            rate = int(frame.sample_rate)
            values = frame.to_ndarray().astype(np.float32)
            chunks.append(values.mean(axis=0) if values.ndim == 2 else values.reshape(-1))
    if not chunks or rate <= 0:
        return np.empty(0), np.empty(0)
    signal = np.concatenate(chunks)
    step = max(1, int(np.ceil(len(signal) / max_points)))
    indexes = np.arange(0, len(signal), step)
    return indexes / rate, signal[indexes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-dir", default="results/q1")
    parser.add_argument("--figure-dir", default="figures/q1")
    args = parser.parse_args()
    setup_style()
    result_dir = REPO / args.q1_dir
    figure_dir = REPO / args.figure_dir
    qa_dir = figure_dir.parent / f"{figure_dir.name}_qa"
    figure_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(result_dir / "q1_samples.csv")
    manifest = pd.read_csv(result_dir / "q1_feature_manifest.csv")
    alignment = pd.read_csv(result_dir / "q1_alignment.csv")
    quality = pd.read_csv(result_dir / "q1_data_quality.csv")
    overlap = pd.read_csv(result_dir / "q1_overlap_similarity.csv")
    permutation = pd.read_csv(result_dir / "q1_overlap_permutation.csv")
    features = np.load(result_dir / "q1_features.npz")
    profile = samples.merge(quality[["sample_id", "alignment_timeline_duration_sec"]], on="sample_id")
    profile["word_count"] = profile.text.fillna("").str.split().str.len()
    audit: list[dict] = []

    def save(fig, name: str) -> None:
        nonlocal audit
        audit += export(fig, figure_dir / name, qa_dir)

    fig, ax = plt.subplots()
    counts = samples.label_class.value_counts().reindex(["Negative", "Neutral", "Positive"])
    ax.barh(counts.index, counts.values, color=[COLORS[x] for x in counts.index])
    for y, value in enumerate(counts):
        ax.text(value + 1, y, f"n={value}", va="center")
    style_ax(ax, "Samples (count)", "Sentiment class", "Attachment-1 class composition")
    ax.set_xlim(0, counts.max() * 1.18)
    save(fig, "raw_q1_class_distribution")

    fig, ax = plt.subplots()
    durations = profile.alignment_timeline_duration_sec
    ax.hist(durations, bins=12, color=COLORS["text"], edgecolor="white")
    ax.axvline(durations.median(), color="#D55E00", linestyle="--", label="Median")
    ax.legend(frameon=False)
    style_ax(ax, "Decoded timeline duration (s)", "Samples (count)", "Decodable-duration distribution (n=100)")
    save(fig, "raw_q1_duration_histogram")

    fig, ax = plt.subplots()
    for label, group in profile.groupby("label_class"):
        ax.scatter(group.word_count, group.alignment_timeline_duration_sec, s=20, alpha=0.72,
                   color=COLORS[label], label=label)
    ax.legend(frameon=False)
    style_ax(ax, "Transcript length (words)", "Decoded timeline duration (s)",
             "Transcript length versus decodable duration")
    save(fig, "raw_q1_words_duration_scatter")

    fig, ax = plt.subplots()
    ax.hist(samples.sequence_length, bins=np.arange(4.5, 51.5, 3), color=COLORS["audio"], edgecolor="white")
    style_ax(ax, "Aligned anchors (count)", "Samples (count)", "Text-anchor sequence lengths after grouping")
    save(fig, "process_q1_anchor_lengths")

    fig, ax = plt.subplots()
    indexed_samples = samples.set_index("sample_id")
    data = [manifest[manifest.modality == modality].valid_position_count /
            indexed_samples.loc[manifest[manifest.modality == modality].sample_id, "sequence_length"].to_numpy()
            for modality in ("text", "audio", "vision")]
    box = ax.boxplot(data, tick_labels=["Text", "Audio", "Vision"], patch_artist=True, showfliers=False)
    for patch, modality in zip(box["boxes"], ("text", "audio", "vision")):
        patch.set_facecolor(COLORS[modality])
    rng = np.random.default_rng(20260923)
    for i, values in enumerate(data, 1):
        ax.scatter(rng.normal(i, 0.035, len(values)), values, s=8, alpha=0.28, color="#333333")
    style_ax(ax, "Modality", "Valid anchors / sequence length", "Per-sample modality validity (box=IQR; n=100)")
    ax.set_ylim(-0.02, 1.05)
    save(fig, "process_q1_validity_boxstrip")

    fig, ax = plt.subplots()
    ax.hist(quality.face_detection_rate, bins=np.linspace(0, 1, 11), color=COLORS["vision"], edgecolor="white")
    ax.axvline(quality.face_detection_rate.median(), color="#333333", linestyle="--", label="Median")
    ax.legend(frameon=False)
    style_ax(ax, "Face-detection rate among sampled frames", "Samples (count)",
             "Visual face-detection quality distribution")
    save(fig, "process_q1_face_detection_rate")

    typical = str(samples.iloc[(samples.sequence_length - samples.sequence_length.median()).abs().argmin()].sample_id)
    rows = alignment[alignment.sample_id.astype(str) == typical].reset_index(drop=True)
    typical_index = {str(value): i for i, value in enumerate(features["sample_id"])}[typical]
    selected = np.unique(np.linspace(0, len(rows) - 1, min(6, len(rows)), dtype=int))
    selected_rows = rows.iloc[selected]
    frame_indexes = selected_rows.video_start_frame_src.fillna(-1).astype(int).tolist()
    source_path = REPO / str(samples[samples.sample_id.astype(str) == typical].iloc[0].source_path)
    thumbnails = decode_selected_frames(source_path, frame_indexes)
    audio_times, audio_signal = decode_audio_waveform(source_path)
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.1, 1.65, 1.2]})
    text_ax, audio_ax, video_ax, feature_ax = axes
    for _, row in selected_rows.iterrows():
        center = (row.start_sec + row.end_sec) / 2
        snippet = str(row.token_text).replace("\n", " ")
        if len(snippet) > 18:
            snippet = snippet[:17] + "…"
        text_ax.text(center, 0.5, snippet, ha="center", va="center", fontsize=7,
                     bbox={"boxstyle": "round,pad=0.25", "fc": "#EAF2F8", "ec": COLORS["text"]})
        text_ax.axvline(row.start_sec, color="#BBBBBB", linewidth=0.5)
        audio_ax.axvspan(row.start_sec, row.end_sec, color=COLORS["audio"], alpha=0.08)
        frame_index = int(row.video_start_frame_src) if pd.notna(row.video_start_frame_src) else -1
        if frame_index in thumbnails:
            video_ax.add_artist(AnnotationBbox(OffsetImage(thumbnails[frame_index], zoom=0.10),
                                               (center, 0.56), frameon=True, pad=0.1))
        video_ax.text(center, 0.04, f"src #{frame_index}", ha="center", va="bottom", fontsize=6)
    text_ax.set_ylim(0, 1); text_ax.set_yticks([])
    text_ax.set_title(f"Readable anchor excerpts: {typical.replace('$_$', ' / ')}")
    if len(audio_signal):
        audio_ax.plot(audio_times, audio_signal, color=COLORS["audio"], linewidth=0.45)
    audio_ax.set_ylabel("Audio\nwaveform")
    audio_ax.spines[["top", "right"]].set_visible(False)
    audio_ax.grid(axis="x", color="#DDDDDD", linewidth=0.4, alpha=0.6)
    video_ax.set_ylim(0, 1); video_ax.set_yticks([]); video_ax.set_ylabel("Video keyframes")
    centers = ((rows.start_sec + rows.end_sec) / 2).to_numpy()
    for modality, marker, line in [("text", "o", "-"), ("audio", "s", "--"), ("vision", "^", ":")]:
        valid = features[f"valid_{modality}"][typical_index][:len(rows)]
        changes = representation_change(features[modality][typical_index][:len(rows)], valid)
        feature_ax.plot(centers, changes, marker=marker, linestyle=line, linewidth=1,
                        markersize=3, color=COLORS[modality], label=modality.title())
    feature_ax.legend(frameon=False, ncols=3)
    style_ax(feature_ax, "Time (s)", "Cosine change from prior valid anchor",
             "Aligned descriptor dynamics")
    save(fig, "process_q1_typical_timeline")

    fig, ax = plt.subplots()
    coverage = manifest.groupby("modality").apply(
        lambda x: x.valid_position_count.sum() / x.sequence_length.sum(), include_groups=False
    ).reindex(["text", "audio", "vision"])
    ax.bar(["Text", "Audio", "Vision"], coverage, color=[COLORS[x] for x in coverage.index])
    for x, value in enumerate(coverage):
        ax.text(x, value + 0.015, f"{value:.1%}", ha="center")
    style_ax(ax, "Modality", "Valid-anchor coverage", "Q1 full-sample feature coverage")
    ax.set_ylim(0, 1.1)
    save(fig, "result_q1_coverage")

    individual = overlap[overlap.sample_id != "__aggregate__"]
    aggregate = overlap[overlap.sample_id == "__aggregate__"].set_index("modality")
    null = permutation.set_index("modality")
    fig, ax = plt.subplots()
    for i, modality in enumerate(("text", "audio", "vision")):
        observed = aggregate.loc[modality, "centered_linear_cka"]
        null_mean = null.loc[modality, "null_mean"]
        ax.errorbar(i, null_mean,
                    yerr=[[null_mean - null.loc[modality, "null_q025"]],
                          [null.loc[modality, "null_q975"] - null_mean]],
                    fmt="o", color="#777777", capsize=4, label="Permutation null 95%" if i == 0 else None)
        ax.scatter(i, observed, marker="D", s=48, color=COLORS[modality], zorder=3,
                   label="Observed aggregate CKA" if i == 0 else None)
    ax.set_xticks(range(3), ["Text", "Audio", "Vision"])
    ax.legend(frameon=False)
    style_ax(ax, "Modality", "Centered linear CKA", "Aggregate CKA versus 200-permutation null")
    ax.set_ylim(0, max(0.15, float(max(aggregate.centered_linear_cka.max(), null.null_q975.max())) * 1.25))
    save(fig, "result_q1_overlap_cka")

    fig, ax = plt.subplots()
    proc_data = [individual[individual.modality == modality].procrustes_residual.dropna()
                 for modality in ("text", "audio", "vision")]
    box = ax.boxplot(proc_data, tick_labels=[f"{name}\nn={len(values)}" for name, values in
                                            zip(("Text", "Audio", "Vision"), proc_data)],
                     patch_artist=True, showfliers=False)
    for patch, modality in zip(box["boxes"], ("text", "audio", "vision")):
        patch.set_facecolor(COLORS[modality])
    for i, values in enumerate(proc_data, 1):
        ax.scatter(rng.normal(i, 0.04, len(values)), values, s=16, alpha=0.55, color="#333333")
    style_ax(ax, "Modality", "Procrustes residual (lower is closer)",
             "Cross-representation residuals (box=IQR; points=samples)")
    save(fig, "result_q1_overlap_procrustes")

    fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.set_axis_off()
    nodes = [(0.34, .94, "Attachment-1\n100 videos + labels"), (0.34, .82, "Decode actual streams\n+ quality audit"),
             (0.34, .68, "Proportional text anchors\n(aligned T≤50)"),
             (0.11, .50, "Text hashing baseline\n128-D"), (0.34, .50, "Audio descriptors\n74-D"),
             (0.57, .50, "Visual descriptors\n35-D @ 10 fps"),
             (0.34, .34, "Explicit masks + time/source indexes"),
             (0.34, .19, "Aligned-50 + independent\naudio/vision-500"), (0.82, .68, "Attachment-2\n18 shared IDs"),
             (0.77, .36, "Aggregate CKA + null\n+ Procrustes audit"),
             (0.56, .05, "Reports + figures +\nfull-chain hashes")]
    edges = [(0,1),(1,2),(2,3),(2,4),(2,5),(3,6),(4,6),(5,6),(6,7),(7,9),(8,9),(7,10),(9,10)]
    for source, target in edges:
        x1, y1, _ = nodes[source]; x2, y2, _ = nodes[target]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops={"arrowstyle": "->", "color": "#34495E", "shrinkA": 28, "shrinkB": 28})
    for x, y, label in nodes:
        ax.add_patch(FancyBboxPatch((x-.11, y-.045), .22, .09, boxstyle="round,pad=0.012",
                                    facecolor="#EAF2F8", edgecolor="#34495E", linewidth=1, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=7, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(-.02, 1)
    save(fig, "flow_overall_model")

    contracts = [
        ("raw_q1_class_distribution", "raw", "Attachment-1 class composition", "q1_samples.csv", "class counts"),
        ("raw_q1_duration_histogram", "raw", "decodable timeline duration", "q1_data_quality.csv", "histogram and median"),
        ("raw_q1_words_duration_scatter", "raw", "text length versus decodable duration", "q1_samples.csv + q1_data_quality.csv", "sample scatter"),
        ("process_q1_anchor_lengths", "process", "post-grouping anchor lengths", "q1_samples.csv", "length histogram"),
        ("process_q1_validity_boxstrip", "process", "per-modality valid ratios", "q1_feature_manifest.csv", "box and sample points"),
        ("process_q1_face_detection_rate", "process", "visual face-detection quality", "q1_data_quality.csv", "sample distribution"),
        ("process_q1_typical_timeline", "process", "readable aligned example", "q1_alignment.csv + q1_features.npz + source MP4", "six anchors, keyframes and descriptor changes"),
        ("result_q1_coverage", "result", "full-sample valid coverage", "q1_feature_manifest.csv", "labeled bars"),
        ("result_q1_overlap_cka", "result", "aggregate representation agreement relative to null", "q1_overlap_similarity.csv + q1_overlap_permutation.csv", "observed CKA and permutation interval"),
        ("result_q1_overlap_procrustes", "result", "cross-representation residuals", "q1_overlap_similarity.csv", "boxplots and sample points"),
        ("flow_overall_model", "flow", "implemented extraction and audit order", "scripts/q1_*.py", "dual resolutions and audit branch"),
    ]
    figure_contracts = {"schema_version": "q1-figure-contracts@1.1",
        "shared_style": {"language": "English", "png_dpi": 300, "formats": ["svg", "png"],
                         "grayscale_qa_dir": f"../{qa_dir.name}",
                         "notes": "The typical-timeline SVG intentionally embeds source-video raster thumbnails."},
        "figures": [{"name": name, "category": category, "claim": claim, "source": source, "evidence": evidence}
                    for name, category, claim, source, evidence in contracts]}
    (figure_dir / "figure_contracts.yaml").write_text(
        yaml.safe_dump(figure_contracts, allow_unicode=True, sort_keys=False), encoding="utf-8")

    truncation_count = int(quality.container_truncation_suspected.sum())
    short_visual_count = int((alignment.video_sampled_frame_count <= 1).sum())
    short_visual_rate = short_visual_count / len(alignment)
    invalid = individual[individual.status != "ok"]
    summary = {"sample_count": int(len(samples)), "label_counts": counts.to_dict(),
               "decoded_duration_sec": durations.describe().to_dict(),
               "declared_duration_truncation_suspected_count": truncation_count,
               "visual_anchors_with_at_most_one_frame": short_visual_count,
               "visual_anchors_with_at_most_one_frame_rate": short_visual_rate,
               "sequence_length": samples.sequence_length.describe().to_dict(),
               "coverage": coverage.to_dict(), "face_detection_rate": quality.face_detection_rate.describe().to_dict(),
               "overlap_count": 18, "aggregate_similarity": aggregate.reset_index().to_dict("records"),
               "permutation_null": permutation.to_dict("records"), "figure_count": len(contracts),
               "grayscale_qa_count": len(contracts), "layout_issue_count": len(audit), "layout_issues": audit}
    (result_dir / "q1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# E题第一问：复杂场景下多模态情感预测的特征提取与时序对齐

## 结论

附件1的 100 条标签与 100 个 MP4 一一对应并全部进入结果集。可比较视图为文本、音频、视觉的 aligned-50 张量，形状分别为 `100×50×128`、`100×50×74`、`100×50×35`；另保留音频和视觉各自的 independent-500 序列，避免把原始高频时间结构永久压缩到文本锚点。所有缺失、有效和填充状态均由布尔掩码表达，禁止根据特征是否为零推断。

实际可解码时间轴平均 {durations.mean():.3f} 秒（范围 {durations.min():.3f}--{durations.max():.3f} 秒），文本锚点平均 {samples.sequence_length.mean():.2f} 个。有效锚点覆盖率为文本 {coverage.text:.1%}、音频 {coverage.audio:.1%}、视觉 {coverage.vision:.1%}。视觉以 10 fps 抽样，并单独保存源帧号、PTS 毫秒与样本级人脸检测率。

10 fps 下仅含0或1个视觉采样帧的锚点为 {short_visual_count}/{len(alignment)}（{short_visual_rate:.1%}），相较评审复算的旧版 731/1917（38.1%）明显下降；仍需完整动态证据时应使用 independent-500 视觉序列，而非只看 aligned-50 池化结果。

## 数据质量与对齐

1. 对齐总时间取实际解码音频与视频时长的较大值，不再用容器声明时长截断；另一模态不存在的位置由显式 mask 表达。
2. 共有 {truncation_count}/100 个 MP4 的 `mvhd` 声明时长比实际可解码音视频时间轴长 0.2 秒以上；该阈值用于排除正常的 AAC/容器尾差。`q1_data_quality.csv` 逐样本保存声明时长、实际解码时长和截断标志。这种附件1与附件2之间的采集/处理差异属于潜在域偏移。
3. 文本锚点按字符权重生成连续半开比例区间，明确标为 `method=proportional,is_fallback=true`，置信度仅由区间内音频/视觉实际覆盖率给出，不宣称 forced alignment。空文本采用 50 个均匀时间格并将 `valid_text=False`。

## 特征边界与附件2审计

当前 128 维文本 hashing、74 维音频描述符和 35 维视觉描述符是确定、可审计的 Q1 基线，并非附件2中 BERT/COVAREP/OpenFace 的同空间替代品。题面允许开源或预训练工具，但不强制第一问重建附件2特征空间；后续模型若要求语义表征，应消费附件2或在依赖与权重可用后替换相应后端，不得仅靠补零或投影伪装维数兼容。

18 个重叠样本的总体中心化线性 CKA 为文本 {aggregate.loc['text'].centered_linear_cka:.4f}、音频 {aggregate.loc['audio'].centered_linear_cka:.4f}、视觉 {aggregate.loc['vision'].centered_linear_cka:.4f}。图中与 200 次确定性置换零基线对照；逐样本 54 个模态比较中 {len(invalid)} 条因变化不足无法估计。结果支持“当前描述符与附件2不可互换”，而不是证明二者等价。

## 产物与复现

- 数据与审计：`q1_features.npz`、`q1_features_unaligned.npz`、`q1_alignment.csv`、`q1_data_quality.csv`、`q1_feature_statistics.csv`、`q1_overlap_similarity.csv`、`q1_overlap_permutation.csv`；
- 图表：10 张数据图和 1 张流程图，均提供 SVG、300 DPI PNG、图表合同与灰度核验图；
- 复现：`python -X utf8 scripts/q1_run_all.py --config config/q1_features.yaml --output-dir results/q1 --figure-dir figures/q1 --status validated`。

本成果仍是 Q1 本地结果，不是冻结的 I01/I02 交换包；跨用户发布须另行完成版本化握手与 wc 审批。
"""
    (result_dir / "q1_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"figure_count": len(contracts), "layout_issue_count": len(audit),
                      "coverage": coverage.to_dict(), "truncation_suspected": truncation_count}))


if __name__ == "__main__":
    main()
