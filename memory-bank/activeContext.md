# Active Context

## 当前状态

- 通用数学建模 Git 协作仓库已完成初始化。
- `main`、`csh`、`xjj`、`wc` 四个分支从同一初始提交建立。
- Memory Bank、多 AI 客户端入口、PR 模板和 Git 协作规范已经就绪。
- 竞赛题目已确定为 E 题“复杂场景下多模态情感识别”。
- 当前执行身份为 `csh`，任务分支为 `feat/csh-q1-data-alignment`。

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

## 下一步

1. 完成问题一模型合同并通过 M1 独立质检。
2. 实现并校验附件1的三模态特征提取和 proportional 对齐最小闭环。
3. 通过 P1 后运行100条全量处理，再发布 draft `dataset-bundle@1` 与 `alignment-bundle@1`。

## 已知问题

- `csh` 尚无可消费的 dataset/alignment exchange bundle。
- forced/CTC 对齐工具尚不可用，首版将使用显式标记的 proportional fallback。
- 问题一 Python 依赖尚未锁定，当前仅完成只读数据审计。
