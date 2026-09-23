# System Patterns

## 推荐目录分层

```text
config/             # 路径、随机种子、实验参数
data/               # 仅保留说明、小型样例和外部数据清单
docs/               # 协作规范、模型说明、数据字典
notebooks/          # 探索性分析；最终结论应沉淀到脚本
src/
  data/             # 读取、清洗、校验
  features/         # 质量指标、配比变换、特征工程
  models/           # 统计、优化、预测等建模模块
  evaluation/       # 指标、交叉验证、不确定性分析
  visualization/    # 统一制图
tests/              # 单元测试和小型集成测试
scripts/            # 可复现运行入口
outputs/            # 自动生成，不纳入 Git
memory-bank/        # 团队共享记忆
```

## 架构约定

- 原始数据只读；清洗结果写入独立输出目录。
- 路径、阈值、随机种子和模型参数集中配置，禁止散落硬编码。
- 探索 notebook 不作为最终执行入口；稳定逻辑迁移到 `src/`，由 `scripts/` 调用。
- 每张论文图表由唯一脚本生成，并记录输入、参数和版本。
- 数据连接必须显式检查主键唯一性、行数变化和未匹配比例。
- 随机实验固定种子；交叉验证划分保存或可确定性重建。
- 公共函数写清输入、输出、单位和异常条件。

## 命名约定

- Python 文件、函数和变量使用 `snake_case`；类使用 `PascalCase`。
- 图表名使用 `fig_<question>_<topic>`，表格名使用 `tab_<question>_<topic>`。
- 实验配置使用 `question_method_variant`，例如 `q2_scaling_hierarchical`。
