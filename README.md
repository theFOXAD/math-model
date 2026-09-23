# 数学建模团队协作仓库

本仓库用于尚在选题阶段的数学建模团队协作。项目采用 Memory Bank 保存跨成员、跨电脑和跨 AI 工具的共享上下文；确定题目后再补充数据、模型和任务分工。

## 快速开始

```bash
git clone <repository-url>
cd math-modeling-collaboration
git switch <你的分支>
git pull --rebase origin <你的分支>
```

开始任务前依次阅读：

1. `AGENTS.md`
2. `memory-bank/activeContext.md`
3. `memory-bank/systemPatterns.md`
4. `docs/GIT_WORKFLOW.md`

完成任务时必须同步更新 `memory-bank/activeContext.md` 和 `memory-bank/progress.md`。

## 长期分支

- `main`：唯一稳定主分支，仅接收经过审查的合并。
- `csh`：csh 的个人集成分支。
- `xjj`：xjj 的个人集成分支。
- `wc`：wc 的个人集成分支。

个人分支不是绕过审查的直推通道。具体开发应从个人分支创建短期任务分支，例如 `feat/csh-quality-score`，完成后先合回个人分支，再由个人分支向 `main` 发起合并请求。

完整规范见 [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)。
