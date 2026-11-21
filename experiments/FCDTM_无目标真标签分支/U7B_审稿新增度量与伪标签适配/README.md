# U7B 审稿新增方法与伪标签适配

本目录保存单独实验的代码、固定协议、审计、分数、评价、图表和通俗分析。目标真实标签只用于评分封存之后的评价；源模型不微调。采用25个历史二分类UNet、50任务、8类别；D42主结果，P37／E50敏感性。包括18个方法条目：FCDTM硬／软、FD、DS、GBC-PL、AC、Entropy、ATC-MC／NE、DoC-gap、DoC-Acc-SrcCal、LogME-ICML-PL、H-score-PL、LEEP-PL、NCE-Pred-PL、NLEEP-PL、OTCE-PL-SrcCal、MMD-RBF-global。

## 入口与结果

- 协议.md及三个补充协议：公式、标签预算、执行修复与数值边界。
- 实验结果分析.md：正式结果及通俗解释；phase0只是早期阶段，不替代full结果。
- artifacts/predictions_full.csv：最终封存分数，未包含目标真值。
- artifacts/逐项评价_full.csv：独立评价后的分数与真实F1／AD；不能输入预测程序。
- artifacts/相关汇总_full.csv、模型选择宏汇总_full.csv：主结果与覆盖。
- artifacts/figures：全部方法／场景的散点图、矢量PDF图册及指标核验索引。
- artifacts/seal_full.json、audit_*.json、execution_full.json：封存与25个原生访问审计。

## 执行

在服务器FCDTM环境中，首次运行使用`code/run_cpu_clean.py`：从相邻U7A冻结清单启动，四模型并发、模型内串行评分。新复现必须使用新的实验输出目录，不覆盖已封存artifacts。当前实际执行经GPU故障及CPU并行初始化停滞后，使用`code/run_cached_resume.py`恢复；该入口只适用于本次尚未完成封存的缓存恢复，不用于覆盖正式结果。

评分完成、访问审计通过后自动封存，独立`code/evaluate_full.py`先调用两个不读目标真值的后处理器：源域DoC回归、单伪类别的精确零规范化；各自保留原始封存并重新封存。随后才读取目标参考。最后运行`code/plot_full.py`绘图。

大型逐像素特征、原始系统调用trace和第三方依赖保留在服务器本实验文件夹，GitHub保存哈希索引、访问审计摘要、可复现代码和论文图表，不上传训练数据／模型检查点。复算需要原始数据和U7A依赖目录。固定路径在相邻U7A/code/common.py中，移植时应在新复现实验记录中说明路径变更。

full2048每图2048点，比较16项；shared256每条源／目标集合各256点，比较全部18项；两面板不混合排序。8／16／24指无标签图数，不是标注图数。NCE的恒零与缺类造成的N/A均完整报告，不能将其误解为零预测误差。原始有标签方法的理论目标与本次冻结模型伪标签适配不同；本轮不是其有标签训练流程的逐位复现。
