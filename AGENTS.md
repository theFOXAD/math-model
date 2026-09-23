# AI 协作规则

任何 AI 或自动化 Agent 在修改项目前必须：

1. 阅读 `memory-bank/activeContext.md`、`memory-bank/systemPatterns.md` 和 `memory-bank/techContext.md`；
2. 确认当前 Git 分支，不得直接在 `main` 上开发；
3. 检查工作区现有改动，不覆盖其他成员未提交的工作；
4. 只修改当前任务范围内的文件，不执行破坏性 Git 命令；
5. 完成后运行与改动相称的验证，并记录验证结果；
6. 更新 `memory-bank/activeContext.md` 和 `memory-bank/progress.md`；
7. 遵守 `docs/GIT_WORKFLOW.md` 的提交、审查与合并规则。

## Memory Bank 约束

- `activeContext.md` 保持简洁，只记录当前状态、最近决策、验证结果和下一步任务；建议不超过 100 行。
- 不在 Memory Bank 中粘贴大段代码、原始数据或聊天记录。
- 架构变更同步更新 `systemPatterns.md`；依赖或环境变更同步更新 `techContext.md`。
- 冲突解决时保留双方仍然有效的事实和待办，不机械选择任意一侧。

## 数据与科研规范

- 原始大数据、压缩包、模型产物和临时图表不得提交 Git；只提交可复现脚本、配置、汇总结果和必要的小型样例。
- 真实、半合成、估算和外推数据必须明确区分。
- 报告数值必须能追溯到脚本、输入和参数；不得把探索性结果写成最终结论。
- AI 生成内容需要人工核验，并按竞赛要求披露 AI 使用范围。
