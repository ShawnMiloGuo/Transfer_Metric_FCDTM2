# FCDTM - 迁移学习特征度量工具包

**Feature-based Cross-Domain Transfer Metric**

一个模块化的迁移学习度量计算工具包，用于评估语义分割模型在不同域之间的迁移性能。

---

## 目录

- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装与依赖](#安装与依赖)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [扩展新度量](#扩展新度量)
- [输出文件说明](#输出文件说明)
- [常见问题](#常见问题)

---

## 功能特性

- **五种度量方法**：
  - **FD (Fréchet Distance)**：基于特征分布的Fréchet距离
  - **DS (Dispersion Score)**：基于类别间特征分散度
  - **GBC (Geometric Bayesian Classifier)**：基于几何贝叶斯分类器
  - **OTCE (Optimal Transport for Conditional Estimation)**：基于最优传输理论的迁移度量
  - **LogME (Log Maximum Evidence)**：基于最大证据的贝叶斯迁移度量

- **模块化设计**：
  - 配置、模型、特征提取、度量计算、可视化完全解耦
  - 易于扩展新的度量方法

- **灵活的运行方式**：
  - 支持单独运行任意度量方法
  - 支持批量运行所有度量
  - 支持命令行参数和配置文件

---

## 项目结构

```
FCDTM/
├── config.py              # 配置管理模块
│                          # - 路径配置（模型、数据、结果）
│                          # - 计算参数配置
│                          # - 任务配置（跨区域/跨传感器）
│
├── model.py               # 模型管理模块
│                          # - 模型加载
│                          # - 特征提取钩子
│                          # - 权重差异分析
│
├── feature_extractor.py   # 特征提取模块
│                          # - FDFeatureExtractor: FD度量特征提取
│                          # - DSFeatureExtractor: DS度量特征提取
│                          # - GBCFeatureExtractor: GBC度量特征提取
│                          # - OTCEFeatureExtractor: OTCE度量特征提取
│                          # - LogMEFeatureExtractor: LogME度量特征提取
│                          # - 分类指标计算（OA, F1, mIoU等）
│
├── metrics/               # 度量计算模块
│   ├── __init__.py       # 模块入口，注册所有度量
│   ├── base.py           # 度量基类和通用工具
│   ├── fd.py             # FD度量实现
│   ├── ds.py             # DS度量实现
│   ├── gbc.py            # GBC度量实现
│   ├── otce.py           # OTCE度量实现
│   └── logme.py          # LogME度量实现
│
├── visualization.py       # 可视化模块
│                          # - 散点图绘制
│                          # - 相关性热力图
│
├── main.py               # 主程序入口
│                          # - 命令行参数解析
│                          # - 任务调度
│                          # - 结果保存
│
├── run.sh                # Shell运行脚本
│                          # - 简化的命令行接口
│                          # - 集中配置管理
│
└── README.md             # 本文档
```

---

## 安装与依赖

### 环境要求

- Python >= 3.8
- PyTorch >= 1.8
- CUDA (推荐，用于GPU加速)

### 依赖安装

```bash
pip install torch torchvision numpy scipy matplotlib tqdm torchmetrics pandas seaborn
```

### 项目依赖

本项目依赖以下自定义模块（需放置在项目根目录或Python路径中）：

- `component/utils.py` - 工具函数
- `component/dataset.py` - 数据集读取器
- `metric_gbc.py` - GBC度量计算（可选）

---

## 快速开始

### 1. 配置路径

编辑 `run.sh` 文件顶部的配置：

```bash
# 路径配置
MODEL_ROOT="/path/to/your/models"      # 模型文件根目录
DATA_ROOT="/path/to/your/data"          # 数据集根目录
RESULT_ROOT="/path/to/your/results"     # 结果输出目录
```

### 2. 运行度量计算

```bash
# 单独运行 FD 度量
bash run.sh --metric FD --task dwq_s2_xj_s2

# 单独运行 DS 度量
bash run.sh --metric DS --task dwq_s2_xj_s2

# 单独运行 GBC 度量
bash run.sh --metric GBC --task dwq_s2_xj_s2

# 单独运行 OTCE 度量
bash run.sh --metric OTCE --task dwq_s2_xj_s2

# 单独运行 LogME 度量
bash run.sh --metric LogME --task dwq_s2_xj_s2

# 运行所有度量
bash run.sh --all --task dwq_s2_xj_s2
```

### 3. 查看结果

结果保存在 `$RESULT_ROOT/{metric_type}/{task_name}/` 目录下：

```
result/
└── FD/
    └── dwq_s2_xj_s2/
        ├── config.json              # 运行配置
        ├── column_names.json        # 结果列名
        ├── result_FD_dwq_s2_xj_s2.csv    # CSV格式结果
        ├── result_FD_dwq_s2_xj_s2.json   # JSON格式结果
        └── fig/                     # 可视化图表
            └── scatter_*.png
```

---

## 配置说明

### 方式一：修改 `run.sh`（推荐）

在 `run.sh` 顶部集中修改所有配置：

```bash
# ============================================================================
# 配置参数（在此集中修改）
# ============================================================================

# 路径配置
MODEL_ROOT="/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/model"
DATA_ROOT="/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/data"
RESULT_ROOT="/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/result"

# 计算参数
BATCH_SIZE=1              # 批次大小
MAX_IMAGES=100            # 最大处理图像数量

# 特征提取参数
FEATURE_LAYER="up4"       # 特征提取层 (up4/outc/down4)
ONLY_FOREGROUND=false     # 是否只提取前景特征
EXCLUDE_ZERO=false        # 是否排除零值特征
USE_PREDICTION=false      # 是否使用预测标签

# 目标域处理
PROCESS_ALL_TARGET=true   # 是否处理所有目标域数据

# 数据集参数
USE_TRAIN_SET=true        # 使用训练集还是验证集
FOREGROUND_RATIO=0.2      # 前景像素比例阈值
```

### 方式二：命令行参数

```bash
python main.py \
    --metric_type FD \
    --task_name dwq_s2_xj_s2 \
    --model_root /path/to/models \
    --data_root /path/to/data \
    --result_root /path/to/results \
    --batch_size 4 \
    --max_images 200 \
    --feature_layer up4
```

### 方式三：修改 `config.py` 默认值

在 `config.py` 的 `Config` 类中修改默认值：

```python
@dataclass
class Config:
    # 路径配置
    model_root: str = "/your/default/model/path"
    data_root: str = "/your/default/data/path"
    result_root: str = "/your/default/result/path"
    
    # 计算参数
    batch_size: int = 4
    max_images: int = 200
    ...
```

---

## 使用方法

### Shell 脚本方式（推荐）

```bash
# 显示帮助
bash run.sh --help

# 运行 FD 度量，指定任务
bash run.sh --metric FD --task dwq_s2_xj_s2

# 运行 DS 度量，指定批次大小
bash run.sh --metric DS --task dwq_l8_xj_l8 --batch 4

# 运行所有度量
bash run.sh --all --task xj_s2_xj_l8
```

### Python 命令行方式

```bash
# 基本用法
python main.py --metric_type FD --task_name dwq_s2_xj_s2

# 完整参数
python main.py \
    --metric_type FD \
    --task_name dwq_s2_xj_s2 \
    --model_root /path/to/models \
    --data_root /path/to/data \
    --result_root /path/to/results \
    --batch_size 1 \
    --max_images 100 \
    --feature_layer up4 \
    --foreground_ratio 0.2
```

### Python API 方式

```python
from config import Config
from main import TransferMetricRunner

# 创建配置
config = Config(
    metric_type="FD",
    task_name="dwq_s2_xj_s2",
    model_root="/path/to/models",
    data_root="/path/to/data",
    result_root="/path/to/results",
    batch_size=1,
    max_images=100
)

# 运行度量计算
runner = TransferMetricRunner(config)
results = runner.run()

# 获取结果
for key, rows in results.items():
    print(f"{key}: {len(rows)} records")
```

### 可用任务列表

| 任务名称 | 源域 | 目标域 | 类型 |
|---------|------|--------|------|
| `dwq_s2_xj_s2` | 大湾区 Sentinel2 | 新疆 Sentinel2 | 跨区域 |
| `dwq_l8_xj_l8` | 大湾区 Landsat8 | 新疆 Landsat8 | 跨区域 |
| `dwq_s2_dwq_l8` | 大湾区 Sentinel2 | 大湾区 Landsat8 | 跨传感器 |
| `xj_s2_xj_l8` | 新疆 Sentinel2 | 新疆 Landsat8 | 跨传感器 |

---

## 扩展新度量

### 步骤 1：创建度量实现文件

在 `metrics/` 目录下创建新文件，如 `metrics/new_metric.py`：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新度量方法实现"""

from .base import BaseMetric, MetricResult
from feature_extractor import BaseFeatureExtractor
from model import ModelManager
import torch


class NewMetric(BaseMetric):
    """
    新度量方法
    
    在此描述度量方法的原理和用途。
    """
    
    # 度量类型名称（必须）
    METRIC_NAME = "NEW"
    
    # 结果列名（必须）
    COLUMN_NAMES = [
        # 在此定义输出列名
        "score_1",
        "score_2",
        "score_3",
    ]
    
    # 绘图时的度量指标列索引
    METRIC_PLOT_INDICES = [0]
    
    # 绘图时的精度指标列索引  
    ACCURACY_PLOT_INDICES = [4, 5]
    
    def compute(
        self,
        model: torch.nn.Module,
        model_manager: ModelManager,
        source_loader,
        target_loader
    ) -> List[MetricResult]:
        """
        计算度量
        
        参数:
            model: 预训练模型
            model_manager: 模型管理器
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
        
        返回:
            度量结果列表
        """
        self.clear_results()
        
        # 1. 提取源域特征
        # ... 使用 feature_extractor 模块
        
        # 2. 提取目标域特征
        # ...
        
        # 3. 计算度量分数
        # ...
        
        # 4. 创建并返回结果
        result = MetricResult(
            source_domain=self.config.source_dataset,
            target_domain=self.config.target_dataset,
            class_index=class_idx,
            class_name=class_name,
            metric_scores={
                "score_1": value1,
                "score_2": value2,
                "score_3": value3,
            }
        )
        self.add_result(result)
        
        return self.results
```

### 步骤 2：注册度量方法

编辑 `metrics/__init__.py`：

```python
from .new_metric import NewMetric

def get_metric(metric_type: str):
    metrics = {
        "FD": FDMetric,
        "DS": DSMetric,
        "GBC": GBCMetric,
        "NEW": NewMetric,  # 添加新度量
    }
    ...
```

### 步骤 3：运行新度量

```bash
bash run.sh --metric NEW --task dwq_s2_xj_s2
```

---

## 输出文件说明

### 目录结构

```
result/
└── {metric_type}/                    # FD, DS, 或 GBC
    └── {task_name}/                  # 如 dwq_s2_xj_s2
        ├── config.json              # 运行配置快照
        ├── column_names.json        # 结果列名定义
        ├── result_{metric}_{task}.csv    # CSV格式结果
        ├── result_{metric}_{task}.json   # JSON格式结果
        └── fig/                     # 可视化图表目录
            ├── scatter_FD_score_OA_delta.png
            └── ...
```

### 结果字段说明

#### FD 度量结果

| 字段名 | 说明 |
|--------|------|
| `source`, `target` | 源域/目标域名称 |
| `class_index`, `class_name` | 类别索引和名称 |
| `OA_source`, `F1_source`, `precision_source` | 源域分类指标 |
| `OA_target`, `F1_target`, `precision_target` | 目标域分类指标 |
| `OA_delta`, `F1_delta` | 精度下降值 |
| `FD_score` | Fréchet距离分数 |
| `FD_raw`, `FD_absolute` | 加权FD分数 |

#### DS 度量结果

| 字段名 | 说明 |
|--------|------|
| `dispersion_score` | 分散度分数 |
| `log_dispersion_score` | 对数分散度分数 |
| `weighted_dispersion_score` | 加权分散度分数 |

#### GBC 度量结果

| 字段名 | 说明 |
|--------|------|
| `diagonal_GBC` | 对角GBC分数 |
| `spherical_GBC` | 球形GBC分数 |

#### OTCE 度量结果

| 字段名 | 说明 |
|--------|------|
| `OT_global` | 全局最优传输距离 |
| `OT_weighted` | 类别加权最优传输距离 |
| `OTCE_score` | OTCE综合分数（越小可迁移性越好） |
| `OT_class_0`, `OT_class_1` | 各类别最优传输距离 |
| `mean_discrepancy` | 均值差异 |
| `coral_distance` | CORAL协方差距离 |
| `MMD_linear` | 线性核MMD距离 |

#### LogME 度量结果

| 字段名 | 说明 |
|--------|------|
| `LogME_score` | LogME分数（越大可迁移性越好） |
| `LogME_fast` | 快速计算的LogME分数 |
| `target_within_class_dist` | 目标域类内距离 |
| `target_between_class_dist` | 目标域类间距离 |
| `target_fisher_ratio` | 目标域Fisher可分性比率 |
| `center_shift` | 源域到目标域的中心偏移 |

---

## 常见问题

### Q1: 如何添加新的迁移任务？

在 `config.py` 的 `TASK_CONFIGS` 字典中添加：

```python
TASK_CONFIGS = {
    "new_task": TaskConfig(
        name="new_task",
        source_dataset="source_name",
        target_dataset="target_name",
        class_indices=[1, 2, 3, 6, 7, 8],
        description="任务描述"
    ),
    ...
}
```

### Q2: 如何使用不同的特征提取层？

```bash
# 使用解码器第4层（默认）
bash run.sh --metric FD

# 使用编码器第4层
python main.py --metric_type FD --feature_layer down4

# 使用输出层
python main.py --metric_type FD --feature_layer outc
```

### Q3: 进度条不显示在终端？

确保使用 `run.sh` 脚本，或直接运行 Python：

```bash
python main.py --metric_type FD --task_name dwq_s2_xj_s2
```

不要将 stderr 重定向到文件。

### Q4: 如何查看详细日志？

日志保存在 `logs/` 目录：

```bash
# 查看最新日志
tail -f logs/FD_dwq_s2_xj_s2_*.log
```

### Q5: 模型文件找不到？

检查 `MODEL_ROOT` 路径是否正确，模型文件应按以下结构组织：

```
model/
├── train_dwq_sentinel2_cls_1/
│   └── unet_xxx_best_val.pth
├── train_dwq_sentinel2_cls_2/
│   └── unet_xxx_best_val.pth
└── ...
```

### Q6: 内存不足怎么办？

减小批次大小或最大图像数：

```bash
bash run.sh --metric FD --batch 1
# 或修改 run.sh 中的 MAX_IMAGES=50
```

---

## 联系方式

如有问题或建议，请联系作者或提交 Issue。

---

## 更新日志

- **2026-03-23**: 新增 OTCE 和 LogME 度量方法，支持五种迁移度量对比
- **2026-03-23**: 模块化重构，支持单独运行各度量方法
- **2024-12-19**: 初始版本
