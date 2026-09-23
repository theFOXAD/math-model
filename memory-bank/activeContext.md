# Active Context

## 当前状态

- 通用数学建模 Git 协作仓库已完成初始化。
- `main`、`csh`、`xjj`、`wc` 四个分支从同一初始提交建立。
- Memory Bank、多 AI 客户端入口、PR 模板和 Git 协作规范已经就绪。
- 竞赛题目已确定为 E 题“复杂场景下多模态情感识别”。
- 当前执行身份为 `csh`，任务分支为 `feat/csh-session-bootstrap`。

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

## 下一步

1. 登记 E 题源数据位置、版本和哈希清单，不提交原始压缩包。
2. 设计并校验 `dataset-bundle@1` 与 `alignment-bundle@1` 的最小生产骨架。
3. 与 `xjj`、`wc` 完成小批次消费握手后再冻结接口产物。

## 已知问题

- `csh` 尚无可消费的 dataset/alignment exchange bundle。
- 源数据位置、版本和校验值尚未登记。
- 尚未确定 Python/R/Matlab 的最终技术栈和版本。
