# Git 流程与分支合并规范

## 1. 分支模型

仓库包含四个长期分支：

```text
main
├── csh
├── xjj
└── wc
```

- `main`：稳定、可复现、可用于论文集成的唯一主干。
- `csh`、`xjj`、`wc`：对应成员的个人集成分支，用于汇总本人已完成并自测的任务。
- 短期任务分支：从个人分支创建，命名为 `<type>/<owner>-<topic>`，例如 `feat/csh-quality-score`、`fix/xjj-scaling-validation`、`docs/wc-methodology`。

禁止直接向 `main` 提交。个人分支也应避免多人同时直接修改；协作任务使用独立短期分支。

## 2. 开始任务

以 csh 为例：

```bash
git switch csh
git fetch origin --prune
git rebase origin/main
git push --force-with-lease origin csh
git switch -c feat/csh-quality-score
```

若团队不允许重写已共享个人分支历史，将 `rebase` 改为：

```bash
git merge --no-ff origin/main
```

一旦个人分支被多人共同使用，禁止普通 `--force` 推送；确需变基时只允许 `--force-with-lease`，并提前通知协作者。

## 3. 日常提交

提交应小而完整，一个提交只表达一个逻辑目的。采用 Conventional Commits：

```text
feat(q1): 增加质量指标方向统一与稳健聚合
fix(q2): 修正跨模型族损失截距偏差
test(q3): 增加预算约束和KKT残差测试
docs(git): 补充分支合并规范
refactor(data): 统一附件读取接口
chore(env): 锁定Python依赖版本
```

允许的常用类型：`feat`、`fix`、`test`、`docs`、`refactor`、`perf`、`chore`。禁止使用“update”“改一下”“final”等无法审查的提交信息。

提交前执行：

```bash
git status
git diff --check
# 运行当前模块测试、分析脚本或最小复现实验
git add <明确的文件列表>
git commit -m "type(scope): 简明说明"
```

不要习惯性使用 `git add .`，避免把数据、缓存和无关改动带入提交。

## 4. Memory Bank 交接

每个可合并任务必须同步更新：

- `memory-bank/activeContext.md`：本次改动、关键决策、验证结果、下一步；
- `memory-bank/progress.md`：勾选完成项并登记新问题；
- `systemPatterns.md`：仅当架构或团队约定变化时更新；
- `techContext.md`：依赖、版本或环境变化时更新。

个人任务分支中的 `activeContext.md` 可以描述本任务状态。合并到 `main` 时，不能简单采用 ours/theirs，应人工合并仍有效的整体进度和待办。

## 5. 从任务分支合并到个人分支

```bash
git switch feat/csh-quality-score
git fetch origin
git rebase csh
# 解决冲突并完成测试
git switch csh
git merge --ff-only feat/csh-quality-score
git push origin csh
```

个人分支内部优先使用 fast-forward，保持任务历史清晰。任务合并并推送后可删除短期分支：

```bash
git branch -d feat/csh-quality-score
git push origin --delete feat/csh-quality-score
```

## 6. 从个人分支合并到 main

1. 从 `csh`、`xjj` 或 `wc` 向 `main` 发起 PR/MR。
2. 填写仓库 PR 模板，列出验证命令、结果和 Memory Bank 更新。
3. 至少一名非作者成员审查；涉及公共架构、核心公式或最终论文结论时，建议两名成员审查。
4. 合并前同步最新 `main`，解决冲突并重新运行验证。
5. 所有必要检查通过后，使用 **Squash merge** 合入 `main`；若一组提交本身具有长期审计价值，可经审查同意使用 `--no-ff` merge。
6. 禁止在 `main` 上使用 rebase、force push 或修改已发布历史。

本地等价操作：

```bash
git switch main
git pull --ff-only origin main
git merge --squash csh
git commit -m "feat(model): 合并首个建模模块"
git push origin main
```

## 7. 合并准入条件

合入 `main` 前必须同时满足：

- 任务目标和边界清楚，变更无无关文件；
- 代码或分析脚本可运行，关键结果可复现；
- 测试、静态检查或人工数值复核通过；
- 没有密钥、个人绝对路径、原始大数据和大型生成文件；
- 新增结论注明数据性质、适用范围和不确定性；
- 文档与代码一致；
- `activeContext.md` 与 `progress.md` 已更新；
- 至少一名非作者批准。

不得为了赶进度跳过检查后再“补审”。未满足条件的工作保留在个人分支。

## 8. 冲突处理规范

1. 在来源分支处理冲突，不在 `main` 上临时修补。
2. 先理解双方语义，再编辑冲突；不得无脑使用 `--ours` 或 `--theirs`。
3. 代码冲突解决后运行受影响测试和最小复现实验。
4. `activeContext.md` 冲突应合并双方仍有效的进度、决策和待办，并删除已失效内容。
5. 数据字典、公式或指标口径冲突必须由原作者之一复核。
6. 将冲突解决及复核结果写入 PR 说明。

## 9. 紧急修复

从 `main` 创建 `hotfix/<topic>`，由至少一名成员快速审查后合回 `main`，再将修复同步到三个个人分支：

```bash
git switch main
git switch -c hotfix/<topic>
# 修改、测试、提交、合并到 main
git switch csh && git merge main
git switch xjj && git merge main
git switch wc  && git merge main
```

紧急修复不豁免验证和 Memory Bank 更新。

## 10. 标签与里程碑

在可复现的重要节点打带注释标签：

```bash
git tag -a v0.1-data-audit -m "完成附件审计"
git tag -a v0.2-models -m "四问主模型可复现"
git tag -a v1.0-submission -m "竞赛提交版本"
git push origin --tags
```

竞赛提交后不得移动 `v1.0-submission`。如需修订，创建 `v1.0.1`。

## 11. 分工与文件所有权

文件所有权用于明确首要审查人，不表示其他成员不能修改：

- `csh`：个人集成分支，具体模块待选题后确定；
- `xjj`：个人集成分支，具体模块待选题后确定；
- `wc`：个人集成分支，具体模块待选题后确定；
- 公共配置、论文总稿和 Memory Bank：至少两人审查。

若实际分工不同，修改本节并同步 `activeContext.md`。
