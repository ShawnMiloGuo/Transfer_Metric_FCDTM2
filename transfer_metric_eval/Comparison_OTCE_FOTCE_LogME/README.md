# 比较实验：OTCE / F-OTCE / LogME

本目录在不修改既有源代码的前提下，新增对 OTCE、F-OTCE 与 LogME 三种方法的对比实验脚本与说明。

- compare_metrics.py：统一对比入口，加载特征与标签，调用各方法计算分数，汇总为 CSV。
- metric_f_otce.py：F-OTCE 的实现（依赖 feature/metric_otce.py 的耦合与条件熵计算）。
- metric_logme.py：LogME 的实现（不依赖外部包，内置一个简化版实现）。
- run_compare.sh：批量运行示例脚本，输出到 `transfer_metric_eval/Comparison_OTCE_FOTCE_LogME/result` 目录。

注意：
- 不修改任何现有文件；新增代码均位于本目录。
- 默认读取已经提取的分割头特征或兼容 `feature/transfer_metric_FD_DS.py` 的产出格式（numpy npy 或 csv）。
- F-OTCE 的具体公式可能存在不同文献变体。当前实现以“条件熵（CE）与耦合代价（W）联合”的常见思路为原型，若需对齐特定论文公式，请提供论文或公式以便精确替换。

## 使用方式

1) 准备数据：
- 源域/目标域的特征文件（n×d，numpy `.npy` 或逗号分隔 `.csv`）
- 源域/目标域的标签文件（n×1，numpy `.npy` 或单列 `.csv`）

2) 运行示例：
```bash
cd transfer_metric_eval/Comparison_OTCE_FOTCE_LogME
bash run_compare.sh \
  --src_x /path/src_features.npy \
  --src_y /path/src_labels.npy \
  --tar_x /path/tar_features.npy \
  --tar_y /path/tar_labels.npy \
  --out result/compare_dwq_s2_to_xj_s2.csv
```

3) 输出：
- CSV 列包含：`method, score, W, CE, extra` 等字段，便于后续相关性评估脚本读取。

## 与现有评估的衔接
- 若要与 `transfer_metric_eval/Correlation/FD_DS_correlation_cross_domain.py` 形式对齐，只需将本目录输出的 CSV 嵌入其读取流程，即可计算与目标域 F1/OA 等的相关性。

## 目录结构
```
transfer_metric_eval/Comparison_OTCE_FOTCE_LogME/
├─ README.md                     # 本说明
├─ compare_metrics.py            # 统一入口
├─ metric_f_otce.py              # F-OTCE 实现
├─ metric_logme.py               # LogME 实现
└─ run_compare.sh                # 批量运行示例
```
