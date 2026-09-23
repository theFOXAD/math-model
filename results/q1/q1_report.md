# E题第一问：多模态特征提取与时序对齐结果

## 结论

附件1的 100 条标签与 100 个 MP4 严格一一对应，全部进入结果集。三模态统一为最长 50 个文本锚点：文本、音频、视觉张量形状分别为 `100×50×128`、`100×50×74`、`100×50×35`，均为 `float32`；有效、注入缺失与填充状态由独立布尔掩码表达。

容器平均时长 7.875 秒（范围 2.257--29.288 秒），文本锚点平均 19.17 个（范围 5--50）。按非填充锚点统计，有效覆盖率为：文本 100.0%、音频 100.0%、视觉 95.31%。未覆盖位置由显式 `valid_*` 掩码标记，不从零值特征推断缺失。

## 方法与验证

1. 文本使用确定性 SHA-256 signed hashing 生成 128 维描述符；音频使用 log-Mel、MFCC、差分与统计生成 74 维描述符；视觉使用 HSV、脸部几何、运动、亮度与边缘生成 35 维描述符。
2. 以文本锚点字符数为权重生成连续半开比例区间，明确标记 `method=proportional,is_fallback=true`，不宣称 forced alignment。
3. 100 条样本全部完成；特征均有限；mask 为 bool 且不与 padding 冲突；时间区间覆盖音频、视频流与容器时长的共同有界区间，索引连续且有界。

## 与附件2的 18 条重叠样本

以本结果显式 mask 和附件2对齐 mask 选择位置。总体中心化线性 CKA 为文本 0.2091、音频 0.0814、视觉 0.0307；对应 Procrustes 残差为 0.9707、0.9650、0.9822。逐样本 54 个模态比较中，1 条为 `insufficient_variation`；具体原因记录在 `status_reason`。

## 产物与复现

- 核心数据：`q1_samples.csv`、`q1_features.npz`、`q1_alignment.csv`、`q1_feature_manifest.csv`；
- 审计与证据：`q1_overlap_similarity.csv`、`q1_extraction_reproduction.json`、`q1_overlap_reproduction.json`、`q1_checksums.sha256`；
- 图表：9 张候选数据图及 1 张流程图，均提供 SVG、300 DPI PNG、合同与灰度核验图。

```powershell
.venv\Scripts\python.exe -X utf8 scripts\q1_run_all.py --config config/q1_features.yaml --output-dir results/q1 --figure-dir figures/q1 --status validated
```

本成果仍是本地 Q1 结果，不是冻结的 I01/I02 交换包；跨用户发布须另行完成版本化握手与 wc 审批。
