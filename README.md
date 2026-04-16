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
- [运行模式详解](#运行模式详解)
- [扩展新度量](#扩展新度量)
- [输出文件说明](#输出文件说明)
- [常见问题](#常见问题)

---

## 功能特性

- **七种度量方法**：
  - **FD (Fréchet Distance)**：原始Fréchet Distance算法，仅输出FD_sum
  - **FCDTM (Fréchet Class Difference Transfer Metric)**：FCDTM最优度量，专注mean_dif_absolute_y0_y1_diff组合方式
  - **FCDTM-Test (FCDTM Test)**：FCDTM研发过程中的测试模型，包含所有组合方式
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
  - **支持两种处理模式**：汇总模式 / 批次模式

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
│   ├── fd.py             # FD度量实现（原始算法，仅FD_sum）
│   ├── fcdtm.py          # FCDTM实现（最优度量）
│   ├── fcdtm_test.py     # FCDTM-Test实现（研发测试模型）
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
├── postprocess/          # 后处理分析模块
│   ├── config.py        # 后处理配置
│   ├── loader.py        # 结果数据加载器
│   ├── visualization.py # 相关性可视化
│   ├── analyze_correlation.py  # 相关性分析主程序
│   └── run_analysis.sh  # 后处理运行脚本
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
# 计算单个度量值（汇总所有目标域数据）
bash run.sh --metric FD --task dwq_s2_xj_s2

# 每4张图片计算一个度量值（批次模式）
bash run.sh --metric FD --task dwq_s2_xj_s2 --batch 4 --batch_target

# 运行所有度量
bash run.sh --all
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
BATCH_SIZE=1              # 批次大小（用于批次模式）
MAX_IMAGES=100            # 最大处理图像数量

# 特征提取参数
FEATURE_LAYER="up4"       # 特征提取层 (up4/outc/down4)
ONLY_FOREGROUND=true      # 是否只提取前景特征（默认true，需与训练时保持一致）
EXCLUDE_ZERO=false        # 是否排除零值特征
USE_PREDICTION=false      # 是否使用预测标签

# 目标域处理
PROCESS_ALL_TARGET=true   # true=汇总模式, false=批次模式

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
    --feature_layer up4 \
    --batch_target  # 启用批次模式
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

### 重要参数说明

#### ONLY_FOREGROUND（前景特征提取）

**默认值为 `true`**，表示只提取前景类别的特征。

**重要**：此参数应与模型训练时的设置保持一致：
- 如果训练时使用 `--only_foreground` 参数，则推理时也应设置 `ONLY_FOREGROUND=true`
- 如果训练时未使用 `--only_foreground` 参数，则应设置 `ONLY_FOREGROUND=false`

| 训练配置 | ONLY_FOREGROUND | 说明 |
|---------|----------------|------|
| 使用 `--only_foreground` | `true` | 只提取前景类别特征 |
| 未使用 `--only_foreground` | `false` | 提取所有类别特征（包括背景） |

**注意**：错误设置此参数会导致特征维度不匹配，计算结果不正确。

#### 批次模式与汇总模式

- **汇总模式**（`PROCESS_ALL_TARGET=true`，默认）：处理完所有目标域数据后，计算单个度量值
- **批次模式**（`PROCESS_ALL_TARGET=false`）：每 `batch_size` 张图片计算一个度量值

---

## 使用方法

### Shell 脚本方式（推荐）

```bash
# 显示帮助
bash run.sh --help

# 汇总模式：计算单个度量值
bash run.sh --metric FD --task dwq_s2_xj_s2

# 批次模式：每 batch_size 张图片计算一个度量值
bash run.sh --metric FD --task dwq_s2_xj_s2 --batch 4 --batch_target

# 运行所有度量（包括FD、FCDTM、FCDTM-Test、DS、GBC、OTCE、LogME）
bash run.sh --all --task xj_s2_xj_l8
```

### Python 命令行方式

```bash
# 汇总模式（默认）
python main.py --metric_type FD --task_name dwq_s2_xj_s2

# 批次模式
python main.py --metric_type FD --task_name dwq_s2_xj_s2 --batch_size 4 --batch_target

# 完整参数
python main.py \
    --metric_type FD \
    --task_name dwq_s2_xj_s2 \
    --model_root /path/to/models \
    --data_root /path/to/data \
    --result_root /path/to/results \
    --batch_size 4 \
    --batch_target \
    --max_images 100 \
    --feature_layer up4 \
    --foreground_ratio 0.2
```

### FCDTM-Test 专用命令

FCDTM-Test 是研发 FCDTM 算法过程中的终极测试模型，包含了所有可能的组合方式。

```bash
# 汇总模式
bash run.sh --metric FCDTM-Test --task dwq_s2_xj_s2

# 批次模式
bash run.sh --metric FCDTM-Test --task dwq_s2_xj_s2 --batch 4 --batch_target

# 或使用 Python
python main.py --metric_type FCDTM-Test --task_name dwq_s2_xj_s2 --batch_size 4 --batch_target
```

### FCDTM 专用命令（推荐）

**FCDTM** 是从 FCDTM-Test 中提取的最优度量方法，只输出 `mean_dif_absolute_y0_y1_diff` 这一列。

**核心公式**：
```
FCDTM = Σ ((mean_target - mean_source) × y0_y1_diff)
```

其中：
- `mean_target`：目标域特征的均值
- `mean_source`：源域特征的均值
- `y0_y1_diff`：模型最后一层权重在类别间的差异

**使用示例**：

```bash
# 汇总模式：计算单个 FCDTM 值
bash run.sh --metric FCDTM --task dwq_s2_xj_s2

# 批次模式：每 batch_size 张图片计算一个 FCDTM 值
bash run.sh --metric FCDTM --task dwq_s2_xj_s2 --batch 4 --batch_target

# 同时运行 FCDTM 和 FCDTM-Test
bash run.sh --metric "FCDTM FCDTM-Test" --task dwq_s2_xj_s2 --batch 4 --batch_target

# 或使用 Python
python main.py --metric_type FCDTM --task_name dwq_s2_xj_s2 --batch_size 4 --batch_target
```

**输出说明**：

FCDTM 的输出 CSV 文件包含以下列：

| 列名 | 说明 |
|------|------|
| `source`, `target` | 源域和目标域名称 |
| `class_index`, `class_name` | 类别索引和名称 |
| `OA_s`, `F1_s`, `precision_s` | 源域精度指标 |
| `OA_t`, `F1_t`, `precision_t` | 目标域精度指标 |
| `OA_delta`, `F1_delta`, `precision_delta` | 精度增量（源-目） |
| `mean_dif_absolute_y0_y1_diff` | **FCDTM 核心度量值** |

### Python API 方式

```python
from config import Config
from main import TransferMetricRunner

# 创建配置（汇总模式）
config = Config(
    metric_type="FD",
    task_name="dwq_s2_xj_s2",
    model_root="/path/to/models",
    data_root="/path/to/data",
    result_root="/path/to/results",
    process_all_target=True  # 汇总模式
)

# 创建配置（批次模式）
config = Config(
    metric_type="FD",
    task_name="dwq_s2_xj_s2",
    batch_size=4,
    process_all_target=False  # 批次模式
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

## 运行模式详解

### 模式一：汇总模式（默认）

**特点**：处理所有目标域数据，计算单个度量值

**适用场景**：
- 需要整体评估迁移性能
- 对比不同迁移任务的整体效果
- 快速获取迁移度量

**使用方法**：
```bash
# 方式1：不指定 --batch_target
bash run.sh --metric FD --task dwq_s2_xj_s2

# 方式2：设置 PROCESS_ALL_TARGET=true
```

**输出结果**：
- 每个迁移任务每个类别 **1 个度量值**
- CSV 文件有 1 行数据（或双向迁移时 2 行）

### 模式二：批次模式

**特点**：每 `batch_size` 张图片计算一个度量值

**适用场景**：
- 分析迁移性能的变化趋势
- 绘制散点图进行相关性分析
- 需要更多数据点进行统计分析

**使用方法**：
```bash
# 每4张图片计算一个FD值
bash run.sh --metric FD --task dwq_s2_xj_s2 --batch 4 --batch_target

# 每1张图片计算一个FD值（最多数据点）
bash run.sh --metric FD --task dwq_s2_xj_s2 --batch 1 --batch_target
```

**输出结果**：
- 每个迁移任务每个类别 **N 个度量值**（N = 目标域图片数 / batch_size）
- CSV 文件有 N 行数据
- 可用于绘制散点图分析度量值与精度变化的相关性

### 模式对比

| 特性 | 汇总模式 | 批次模式 |
|------|---------|---------|
| 参数 | 默认 / 无需 `--batch_target` | 添加 `--batch_target` |
| 输出数量 | 每任务每类别 1 个值 | 每任务每类别 N 个值 |
| 计算速度 | 较快 | 较慢 |
| 适用分析 | 整体评估 | 趋势分析、相关性分析 |
| 散点图 | 不适用 | 适用 |

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
from typing import List
import numpy as np


class NewMetric(BaseMetric):
    """
    新度量方法
    
    在此描述度量方法的原理和用途。
    """
    
    # 度量类型名称（必须）
    METRIC_NAME = "NEW"
    
    # 结果列名（必须）
    COLUMN_NAMES = [
        # 源域指标
        "OA_source", "F1_source", "mIoU_source", "precision_source", "recall_source",
        # 目标域指标
        "OA_target", "F1_target", "mIoU_target", "precision_target", "recall_target",
        # 增量指标
        "OA_delta", "F1_delta", "mIoU_delta", "precision_delta", "recall_delta",
        # 度量分数
        "score_1", "score_2", "score_3",
    ]
    
    # 绘图时的度量指标列索引（相对于COLUMN_NAMES）
    METRIC_PLOT_INDICES = [15]  # score_1
    
    # 绘图时的精度指标列索引
    ACCURACY_PLOT_INDICES = [10, 11]  # OA_delta, F1_delta
    
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
        
        device = model_manager.device
        
        # ========== 提取源域特征（一次性提取所有） ==========
        # ... 使用 feature_extractor 模块
        
        # ========== 处理目标域 ==========
        if self.config.process_all_target:
            # 汇总模式：处理所有目标域数据
            # ... 提取所有目标域特征
            # ... 计算度量
            result = self._compute_single_result(...)
            self.add_result(result)
        else:
            # 批次模式：按批次处理目标域
            target_iter = iter(target_loader)
            n_batches = len(target_loader)
            
            for batch_idx in range(n_batches):
                try:
                    # ... 提取当前批次特征
                    # ... 计算度量
                    result = self._compute_single_result(...)
                    self.add_result(result)
                except StopIteration:
                    break
        
        return self.results
    
    def _compute_single_result(self, ...) -> MetricResult:
        """计算单个结果"""
        return MetricResult(
            source_domain=self.config.source_dataset,
            target_domain=self.config.target_dataset,
            class_index=0,
            class_name="",
            # 源域指标
            OA_source=...,
            F1_source=...,
            # ... 其他指标
            # 度量分数
            metric_scores={
                "score_1": value1,
                "score_2": value2,
                "score_3": value3,
            }
        )
```

### 步骤 2：注册度量方法

编辑 `metrics/__init__.py`：

```python
from .new_metric import NewMetric

def get_metric(metric_type: str):
    metrics = {
        "FD": FDMetric,
        "FCDTM-Test": FCDTMTestMetric,
        "DS": DSMetric,
        "GBC": GBCMetric,
        "OTCE": OTCEMetric,
        "LogME": LogMEMetric,
        "NEW": NewMetric,  # 添加新度量
    }
    ...
```

### 步骤 3：更新配置

在 `config.py` 的 `MetricType` 枚举中添加：

```python
class MetricType(Enum):
    FD = "FD"
    DS = "DS"
    GBC = "GBC"
    OTCE = "OTCE"
    LogME = "LogME"
    NEW = "NEW"  # 添加新度量
```

### 步骤 4：运行新度量

```bash
# 汇总模式
bash run.sh --metric NEW --task dwq_s2_xj_s2

# 批次模式
bash run.sh --metric NEW --task dwq_s2_xj_s2 --batch 4 --batch_target
```

---

## 输出文件说明

### 目录结构

```
result/
├── FD/                           # 原始FD算法结果
│   └── {task_name}/
│       └── result_FD_{task}.csv
├── FCDTM-Test/                   # FCDTM研发测试结果
│   └── {task_name}/
│       └── result_FCDTM-Test_{task}.csv
├── DS/
│   └── {task_name}/
│       └── result_DS_{task}.csv
├── GBC/
│   └── {task_name}/
│       └── result_GBC_{task}.csv
├── OTCE/
│   └── {task_name}/
│       └── result_OTCE_{task}.csv
└── LogME/
    └── {task_name}/
        └── result_LogME_{task}.csv
```

**支持两种目录结构**：
1. **层级结构**：`result/{metric_type}/{task_name}/*.csv`
2. **扁平结构**：`result/*.csv`（自动从文件名推断任务信息）

### 结果字段说明

#### FD 度量结果

原始 Fréchet Distance 算法，仅输出 FD_sum。

| 字段名 | 说明 |
|--------|------|
| `source`, `target` | 源域/目标域名称 |
| `class_index`, `class_name` | 类别索引和名称 |
| `OA_s`, `F1_s`, `precision_s` | 源域分类指标 |
| `OA_t`, `F1_t`, `precision_t` | 目标域分类指标 |
| `OA_delta`, `F1_delta`, `precision_delta` | 精度下降值 |
| `OA_delta_relative`, `F1_delta_relative` | 相对精度下降 |
| `FD_sum` | **核心指标**：原始 Fréchet Distance 分数 |

### FCDTM-Test 度量结果

FCDTM-Test 是研发 FCDTM 算法过程中的终极测试模型，包含了所有可能的组合方式。

| 字段名 | 说明 |
|--------|------|
| `source`, `target` | 源域/目标域名称 |
| `class_index`, `class_name` | 类别索引和名称 |
| `OA_s`, `F1_s`, `precision_s` | 源域分类指标 |
| `OA_t`, `F1_t`, `precision_t` | 目标域分类指标 |
| `OA_delta`, `F1_delta`, `precision_delta` | 精度下降值 |
| 均值差异基础统计 | `mean_dif_absolute_sum` 等 |
| 均值差异×权重差异 | `mean_dif_absolute_y0_y1_diff` 等（共16种） |
| FD分数 | `FD_sum`, `FD_y0_y1_diff` 等（共5种） |

**核心公式**：
```
最优组合 = Σ(|mean_t - mean_s| × y0_y1_diff)
```
其中：
- `|mean_t - mean_s|`：目标域与源域特征均值差异的绝对值
- `y0_y1_diff`：模型最后一层权重的类别间差异

该组合在实验中与迁移后精度下降的相关性最高。

#### DS 度量结果

| 字段名 | 说明 |
|--------|------|
| `dispersion_score` | 分散度分数 |
| `log_dispersion_score` | 对数分散度分数 |
| `weighted_dispersion_score` | 加权分散度分数 |
| `weighted_log_dispersion_score` | 加权对数分散度分数 |

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

## 后处理分析

### 概述

后处理分析模块 (`postprocess/`) 提供了迁移度量结果的相关性分析和可视化功能，用于分析度量指标与精度下降之间的相关性。

### 目录结构

```
postprocess/
├── __init__.py              # 模块入口
├── config.py                # 后处理配置
├── loader.py                # 数据加载器
├── visualization.py         # 可视化模块
├── analyze_correlation.py   # 主程序
└── run_analysis.sh          # 运行脚本
```

### 功能特性

- **相关性分析**：计算度量指标与精度下降的 Pearson/Spearman 相关系数
- **按类别分析**：支持按不同类别（Cropland, Forest, Water等）和迁移方向分别分析
- **可视化**：
  - **热力图**（一张）：行=度量指标，列=类别+迁移方向，值=相关系数
  - **散点图**（每指标一张）：不同类别用不同颜色区分
- **批量处理**：支持批量分析多个度量类型和任务
- **灵活目录结构**：支持层级结构和扁平结构

### 使用方法

#### 方式一：Shell 脚本（推荐）

```bash
# 进入后处理目录
cd postprocess

# 默认配置运行（分析FD、FCDTM-Test、DS、GBC）
./run_analysis.sh

# 指定度量类型
./run_analysis.sh --metric_types FD FCDTM-Test

# 指定单个度量
./run_analysis.sh --metric_types FCDTM-Test

# 完整参数
./run_analysis.sh \
    --result_root ../result \
    --output_dir ../analysis \
    --metric_types FD FCDTM-Test DS \
    --batch_sizes 1 \
    --correlation_methods pearson
```

#### 方式二：Python 命令行

```bash
python postprocess/analyze_correlation.py \
    --result_root ./result \
    --output_dir ./analysis \
    --metric_types FD FCDTM-Test DS \
    --batch_sizes 1 \
    --correlation_methods pearson
```

#### 方式三：Python API

```python
from postprocess.config import PostprocessConfig
from postprocess.loader import load_all_csv_files, merge_all_data
from postprocess.visualization import CorrelationVisualizer

# 创建配置
config = PostprocessConfig(
    result_root="./result",
    output_dir="./analysis",
)

# 加载数据（支持层级结构和扁平结构）
loader = load_all_csv_files("./result")
df = merge_all_data(loader)

# 可视化
visualizer = CorrelationVisualizer(config)

# 绘制热力图（行=度量指标，列=类别+迁移方向）
from postprocess.config import METRIC_SCORE_COLUMNS
metric_cols = METRIC_SCORE_COLUMNS["FCDTM-Test"]

fig, corr_df = visualizer.draw_heatmap_metrics_vs_class_task(
    df, metric_cols, "F1_delta",
    method="pearson",
    save_path="./heatmap_F1_delta.png"
)

# 绘制散点图
visualizer.draw_all_scatter_by_class(
    df, metric_cols, "F1_delta",
    output_dir="./fig"
)
```

### 输出文件

```
analysis/
├── fig/
│   ├── heatmap_F1_delta_pearson.png              # 热力图
│   │                                                 # 行：度量指标
│   │                                                 # 列：类别+迁移方向
│   │                                                 # 值：相关系数
│   ├── correlation_F1_delta_pearson.csv          # 相关性矩阵CSV
│   ├── F1_delta_scatter_mean_dif_absolute_y0_y1_diff.png  # 散点图
│   └── F1_delta_scatter_FCDTM_score.png          # 散点图
└── csv/
    └── correlation_F1_delta_pearson.csv          # 相关性矩阵
```

### 热力图结构示例

```
                              Cropland_dwq_s2  Forest_dwq_s2  Water_dwq_s2  ...
mean_dif_absolute_y0_y1_diff    0.85           0.92           0.78        ...
FCDTM_score                     0.82           0.89           0.75        ...
FD_sum                          0.75           0.80           0.68        ...
```

**说明**：
- 行（纵坐标）：度量指标
- 列（横坐标）：类别+迁移方向（如 Cropland_dwq_s2 表示 Cropland 类别在 dwq→s2 迁移任务中）
- 值：相关系数（-1 到 1），反映该度量指标与精度下降的相关程度

### FCDTM-Test 度量指标列表

FCDTM-Test 包含所有可能的组合方式，共 25 个指标：

| 类别 | 指标名 | 说明 |
|------|--------|------|
| 均值差异基础 | `mean_dif_absolute_sum` | 均值绝对差异之和 |
| | `mean_dif_absolute_abs_sum` | 均值绝对差异绝对值之和 |
| | `mean_dif_relative_sum` | 均值相对差异之和 |
| | `mean_dif_relative_abs_sum` | 均值相对差异绝对值之和 |
| 加权差异 | `mean_dif_absolute_y0_y1_diff` | 均值差异 × 权重差异 |
| | `mean_dif_absolute_y0_y1_diff_abs` | 均值差异 × 权重差异绝对值 |
| | `mean_dif_absolute_y0_y1_diff_normalized` | 均值差异 × 归一化权重差异 |
| | `mean_dif_absolute_y0_y1_diff_abs_normalized` | 均值差异 × 归一化权重差异绝对值 |
| | ... | （共16个加权差异指标） |
| FD分数 | `FD_sum` | 原始Fréchet距离 |
| | `FD_y0_y1_diff` | 加权FD |
| | `FD_y0_y1_diff_abs` | 绝对加权FD |
| | `FD_y0_y1_diff_normalized` | 归一化加权FD |
| | `FD_y0_y1_diff_abs_normalized` | 归一化绝对加权FD |

### FD 度量指标列表

FD 仅包含原始算法，共 1 个指标：

| 指标名 | 说明 |
|--------|------|
| `FD_sum` | 原始 Fréchet Distance 分数 |

### 相关性矩阵示例

```
                              Cropland_dwq_s2  Forest_dwq_s2  Water_dwq_s2
FD_sum                          0.75           0.80           0.68
mean_dif_absolute_y0_y1_diff    0.85           0.92           0.78
FD_y0_y1_diff                   0.82           0.89           0.75
```

---

## 常见问题

### Q1: 如何选择汇总模式还是批次模式？

**汇总模式**：
- 快速获取迁移性能评估
- 对比不同迁移任务的整体效果
- 只需要一个代表性数值

**批次模式**：
- 需要分析度量值的变化趋势
- 需要绘制散点图分析相关性
- 需要更多数据点进行统计分析

### Q2: BATCH_SIZE 参数如何影响结果？

| 参数设置 | 汇总模式 | 批次模式 |
|---------|---------|---------|
| `--batch 1` | 无影响 | 每张图片计算一个度量值（最多数据点） |
| `--batch 4` | 无影响 | 每4张图片计算一个度量值 |
| `--batch 10` | 无影响 | 每10张图片计算一个度量值 |

**注意**：在批次模式下，`batch_size` 决定了每个度量值对应的图片数量。

### Q3: 如何添加新的迁移任务？

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

### Q4: 如何使用不同的特征提取层？

```bash
# 使用解码器第4层（默认）
bash run.sh --metric FD

# 使用编码器第4层
python main.py --metric_type FD --feature_layer down4

# 使用输出层
python main.py --metric_type FD --feature_layer outc
```

### Q5: 为什么批次模式下结果数量不对？

检查以下几点：
1. 确保添加了 `--batch_target` 参数
2. 检查 `PROCESS_ALL_TARGET` 是否设置为 `false`
3. 确认目标域数据加载器的长度

```bash
# 正确的批次模式命令
bash run.sh --metric FD --task dwq_s2_xj_s2 --batch 4 --batch_target
```

### Q6: 进度条不显示在终端？

确保使用 `run.sh` 脚本，或直接运行 Python：

```bash
python main.py --metric_type FD --task_name dwq_s2_xj_s2
```

不要将 stderr 重定向到文件。

### Q7: 如何查看详细日志？

日志保存在 `logs/` 目录：

```bash
# 查看最新日志
tail -f logs/FD_dwq_s2_xj_s2_*.log
```

### Q8: 模型文件找不到？

检查 `MODEL_ROOT` 路径是否正确，模型文件应按以下结构组织：

```
model/
├── train_dwq_sentinel2_cls_1/
│   └── unet_xxx_best_val.pth
├── train_dwq_sentinel2_cls_2/
│   └── unet_xxx_best_val.pth
└── ...
```

### Q9: 内存不足怎么办？

减小批次大小或最大图像数：

```bash
# 批次模式：减小 batch_size
bash run.sh --metric FD --batch 1 --batch_target

# 汇总模式：减小 max_images
# 修改 run.sh 中的 MAX_IMAGES=50
```

### Q10: 如何验证度量索引是否正确？

运行索引验证脚本：

```bash
python test_all_metrics_v2.py
```

---

## 联系方式

如有问题或建议，请联系作者或提交 Issue。

---
## 张拓实验调用代码
### 模型计算
```bash
bash ./FCDTM/run.sh --metric "FCDTM" --task dwq_s2_xj_s2 --batch 4 --batch_target
```
### 后处理
```bash
bash ./postprocess/run_analysis.sh --batch_sizes 4 --metric_types "FD FCDTM FCDTM—Test"
```


## 更新日志

- **2026-03-25**: 重构后处理分析模块，适配新的CSV结果格式，新增 `postprocess/` 目录
- **2026-03-24**: 修复 BATCH_SIZE 参数未生效问题，新增两种运行模式（汇总模式/批次模式）
- **2026-03-23**: 新增 OTCE 和 LogME 度量方法，支持五种迁移度量对比
- **2026-03-23**: 模块化重构，支持单独运行各度量方法
- **2024-12-19**: 初始版本
