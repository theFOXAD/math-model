# Active Context

## 当前状态

- 通用数学建模 Git 协作仓库已完成初始化。
- `main`、`csh`、`xjj`、`wc` 四个分支从同一初始提交建立。
- Memory Bank、多 AI 客户端入口、PR 模板和 Git 协作规范已经就绪。
- 竞赛题目为 E 题“复杂场景下多模态情感预测的数学建模与算法设计”。
- 当前执行身份为 `csh`，任务分支为 `fix/csh-q1-review-remediation`。
- 外部评审整改已完成并提交为 `e6a8e05`：aligned-50 与 independent-500 双分辨率、实际解码时间轴、源帧/PTS、人脸率、空文本回退、逐维统计、CKA置换零基线和11幅图均已重算。
- 独立 P1/P2 均 PASS（P0=0、P1=0）；根复现清单状态为 `validated`，manifest 50项、checksum 51行均零失配，但未获 wc 审批，仍不得标记 frozen 或进入论文。
- 2026-09-24 15:32（Asia/Shanghai）额度为 AMBER：五小时剩余31%，每周剩余51%。

## 最近决策

- `main` 只接受审查后的合并请求，不直接开发。
- 三名成员分别使用 `csh`、`xjj`、`wc` 作为个人集成分支。
- `csh` 负责 `src/data`、`src/features`、`scripts/q1_*`、`config/q1_*`，生产数据、掩码和对齐索引。
- 跨成员数据交换采用版本化 exchange bundle；缺失模态必须使用显式掩码，禁止由零值推断。
- 大型原始附件不进入 Git，确定题目后使用外部共享存储，并在代码中通过配置路径引用。
- 提交代码时必须同时提交相关 Memory Bank 更新。

## 最近验证

- 2026-09-23：Codex five-hour 与 weekly 额度均为 GREEN，启动门禁通过。
- 已读取 `AGENTS.md`、`systemPatterns.md`、`techContext.md` 和 Git 工作流；启动前工作树干净。
- 已核验 E 题原始包 SHA-256、附件1--4文件数量与大小；附件1的100条标签与100个视频一一对应。
- 附件1中18条样本与附件2标签表重叠，标签完全一致，其中11条属于 train、7条属于 test。
- 问题一 `题目分析报告.md` 与 `术语表格.md` 已通过独立 M1 终检；冻结快照分别为 `318888...67C4` 与 `50524E...7CCE`。
- 单样本与空转写 P1 复验 PASS；aligned 张量为 text `[100,50,128]`、audio `[100,50,74]`、vision `[100,50,35]`，independent 音视频为 `[100,500,74]`、`[100,500,35]`。
- 88/100 文件的 `mvhd` 声明时长比实际解码轴长超过0.2秒；声明总时长1120.144秒，实际解码轴总长786.278秒。
- 1917个锚点中≤1个视觉采样帧的仅168个（8.8%，旧版38.1%）；文本/音频/视觉有效覆盖率为100.0%/99.95%/99.11%。
- 18条交集审计输出57条数据行，56条可估计、1条 `insufficient_variation`；聚合CKA均与200次置换零基线对照。
- 10张数据图与1张流程图均提供SVG、300 DPI PNG和灰度图；22个正式图文件通过 strict figure audit。
- 最终 `validated` 单命令全链通过；manifest 50项与checksum 51行均零缺失、零失配。

## 下一步

1. 非作者审阅 `e6a8e05` 后，决定是合并到个人 `csh` 分支还是提交 PR；本轮未推送远端。
2. 若下游需要语义/专业声学视觉空间，另开原子任务引入锁定的 BERT/COVAREP/OpenFace 后端；不得把当前可审计基线伪称为附件2同空间。
3. I01/I02 作为后续独立任务执行消费者加载与冻结握手。

## 已知问题

- `csh` 尚无可消费的 frozen dataset/alignment exchange bundle；本轮没有写 I01/I02。
- BERT/COVAREP/OpenFace 与 forced/CTC 工具当前不可用；结果明确标为轻量可审计基线与 `method=proportional,is_fallback=true`。
- P2 已通过，但任何论文采用或 I09 冻结仍必须等待 wc 审批。
