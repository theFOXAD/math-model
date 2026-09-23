# E题第一问：多模态特征提取与时序对齐结果

## 结论

附件1的 100 条标签与 100 个 MP4 严格一一对应，全部进入结果集。三模态统一为最长 50 个文本锚点：文本、音频、视觉张量形状分别为 `100×50×128`、`100×50×74`、`100×50×35`，均为 `float32`；有效、注入缺失与填充状态由独立布尔掩码表达。

视频平均时长 7.958 秒（范围 2.402--29.433 秒），文本锚点平均 19.17 个（范围 5--50）。按非填充锚点统计，有效覆盖率为：文本 100.0%、音频 99.79%、视觉 96.24%。未能覆盖的音视频位置保留在样本中并由 `valid_audio`、`valid_vision` 标记，未用全零特征反推缺失。

## 方法

1. 文本：将连续词组作为统一锚点；超过 50 词时确定性均衡分组，不截断。每个锚点使用字符三元组、词形和上下文的 SHA-256 signed hashing，生成 128 维归一化描述符。
2. 音频：解码为单声道 16 kHz，以 25 ms 窗、10 ms 步长计算 40 维三角 Mel 滤波器组 log 能量、13 维 MFCC、13 维 MFCC 一阶差分及 8 维韵律/质量统计，共 74 维。
3. 视觉：按 5 fps 采样，聚合 24 维 HSV、6 维人脸几何、3 维运动和 2 维亮度/边缘描述，共 35 维。人脸未检出不等于视觉流缺失。
4. 对齐：以锚点字符数为权重，将视频时长按比例分割为连续半开区间，并映射到共享边界的音频样点和采样视频帧。该方法明确标记为 `proportional` 与 fallback，不宣称是 forced alignment。

## 验证

- 100 条样本全部完成；所有特征为有限数，无 NaN/Inf。
- 时间区间首端为 0、末端覆盖完整时长，相邻时间和音频样点区间连续、互不重叠，所有音视频索引有界。
- `valid_text` 的计数与非填充位置数一致；全部 mask 为 bool，所有有效/注入缺失位置均不落在 padding 区。
- 同一真实样本重复运行得到逐元素相同的数组及相同 NPZ SHA-256。
- P1 独立质检通过，无 P0/P1 问题。

## 与附件2的 18 条重叠样本

交集样本标签一致。将共同有效位置展平后，总体中心化线性 CKA 为文本 0.2091、音频 0.0812、视觉 0.0562；对应 Procrustes 残差为 0.9707、0.9652、0.9786。它们说明轻量自主描述符与附件2参照表征并不等价，不能把两者无条件替换。逐样本 54 个模态比较中，1 条视觉参照没有共同有效位置，按合同记录为 `insufficient_variation`，其余均正常计算。

## 产物与复现

- 核心数据：`q1_samples.csv`、`q1_features.npz`、`q1_alignment.csv`、`q1_feature_manifest.csv`；
- 重叠审计：`q1_overlap_similarity.csv`；
- 复现证据：`复现清单.json`、`q1_overlap_reproduction.json`、`q1_checksums.sha256`；
- 图表：`figures/q1/` 下 9 张候选数据图及 1 张流程图，均提供纯矢量 SVG 与 300 DPI PNG。

运行命令：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\q1_extract_features.py --output-dir results/q1
.venv\Scripts\python.exe -X utf8 scripts\q1_overlap_audit.py --q1-dir results/q1
.venv\Scripts\python.exe -X utf8 scripts\q1_make_figures.py
```

本目录是第一问题目成果，不是冻结的跨用户 I01/I02 交换包；后续接口生产必须另行执行版本化发布与消费者握手。
