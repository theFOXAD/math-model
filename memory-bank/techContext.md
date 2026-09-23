# Tech Context

## 当前状态

技术栈尚待团队确认。建议以 Python 3.11 为主，必要时使用 Matlab 或 R 复核特定算法。

## 建议依赖

- 数据：pandas、numpy、pyarrow
- 统计与优化：scipy、statsmodels、scikit-learn
- 贝叶斯/层级模型（可选）：pymc
- 可视化：matplotlib、seaborn
- 测试与质量：pytest、ruff

正式引入依赖时必须锁定版本，并在本文件记录版本选择和兼容性说明。

## 环境约束

- 大型 `.jsonl.xz` 应流式读取，避免一次性解压和载入内存。
- 数据根目录由环境变量或配置文件提供，不写死个人绝对路径。
- 不提交 API Key、账号凭据、个人目录或本地虚拟环境。
- 运行脚本应从仓库根目录执行，并给出最小可复现命令。
