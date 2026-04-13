#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后处理配置模块

定义后处理分析的参数配置，包括:
- 结果文件路径
- 相关性计算参数
- 可视化参数
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class CorrelationMethod(Enum):
    """相关性计算方法"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"


# 预定义的迁移任务配置
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dwq_s2_xj_s2": {
        "source": "dwq_sentinel2",
        "target": "xj_sentinel2",
        "description": "大湾区 Sentinel2 -> 新疆 Sentinel2 (跨区域)"
    },
    "dwq_l8_xj_l8": {
        "source": "dwq_landsat8",
        "target": "xj_landsat8",
        "description": "大湾区 Landsat8 -> 新疆 Landsat8 (跨区域)"
    },
    "dwq_s2_dwq_l8": {
        "source": "dwq_sentinel2",
        "target": "dwq_landsat8",
        "description": "大湾区 Sentinel2 -> 大湾区 Landsat8 (跨传感器)"
    },
    "xj_s2_xj_l8": {
        "source": "xj_sentinel2",
        "target": "xj_landsat8",
        "description": "新疆 Sentinel2 -> 新疆 Landsat8 (跨传感器)"
    },
}

# 支持的度量类型
METRIC_TYPES = ["FD", "FCDTM", "FCDTM-Test", "DS", "GBC", "OTCE", "LogME"]

# 默认的精度指标列
DEFAULT_ACCURACY_COLUMNS = [
    "OA_delta", "F1_delta", "precision_delta",
    "OA_delta_relative", "F1_delta_relative", "precision_delta_relative"
]

# 精度指标列（用于分析）
ACCURACY_COLUMNS = DEFAULT_ACCURACY_COLUMNS

# 默认的度量分数列（按度量类型）- 完整列表
METRIC_SCORE_COLUMNS = {
    # FD: 原始Fréchet Distance算法，仅输出FD_sum
    "FD": [
        "FD_sum",  # 原始FD分数
    ],
    # FCDTM: FCDTM最优度量，专注mean_dif_absolute_y0_y1_diff组合方式
    "FCDTM": [
        # FCDTM 核心度量
        "mean_dif_absolute_y0_y1_diff",  # 核心度量：均值差异绝对值 × 权重差异
        # FCDTM 综合分数
        "FCDTM_score",  # 综合分数 (等价于 FD_y0_y1_diff)
    ],
    # FCDTM-Test: FCDTM研发过程中的测试模型，包含所有组合方式
    "FCDTM-Test": [
        # 均值差异基础统计
        "mean_dif_absolute_sum",
        "mean_dif_absolute_abs_sum",
        "mean_dif_relative_sum",
        "mean_dif_relative_abs_sum",
        # 均值差异 × 权重差异 (y0_y1_diff)
        "mean_dif_absolute_y0_y1_diff",
        "mean_dif_absolute_abs_y0_y1_diff",
        "mean_dif_relative_y0_y1_diff",
        "mean_dif_relative_abs_y0_y1_diff",
        # 均值差异 × 权重差异绝对值 (y0_y1_diff_abs)
        "mean_dif_absolute_y0_y1_diff_abs",
        "mean_dif_absolute_abs_y0_y1_diff_abs",
        "mean_dif_relative_y0_y1_diff_abs",
        "mean_dif_relative_abs_y0_y1_diff_abs",
        # 均值差异 × 归一化权重差异 (y0_y1_diff_normalized)
        "mean_dif_absolute_y0_y1_diff_normalized",
        "mean_dif_absolute_abs_y0_y1_diff_normalized",
        "mean_dif_relative_y0_y1_diff_normalized",
        "mean_dif_relative_abs_y0_y1_diff_normalized",
        # 均值差异 × 归一化权重差异绝对值 (y0_y1_diff_abs_normalized)
        "mean_dif_absolute_y0_y1_diff_abs_normalized",
        "mean_dif_absolute_abs_y0_y1_diff_abs_normalized",
        "mean_dif_relative_y0_y1_diff_abs_normalized",
        "mean_dif_relative_abs_y0_y1_diff_abs_normalized",
        # FD 分数
        "FD_sum",
        "FD_y0_y1_diff",
        "FD_y0_y1_diff_abs",
        "FD_y0_y1_diff_normalized",
        "FD_y0_y1_diff_abs_normalized",
    ],
    "DS": [
        "dispersion_score", "log_dispersion_score",
        "weighted_dispersion_score", "weighted_log_dispersion_score",
    ],
    "GBC": [
        "diagonal_GBC", "spherical_GBC",
    ],
    "OTCE": [
        "OT_global", "OT_weighted", "OTCE_score",
        "mean_discrepancy", "coral_distance", "MMD_linear",
    ],
    "LogME": [
        "LogME_score", "LogME_fast",
        "target_within_class_dist", "target_between_class_dist",
        "target_fisher_ratio", "center_shift",
    ],
}

# 所有度量分数列（合并所有度量类型）
METRIC_COLUMNS = []
for cols in METRIC_SCORE_COLUMNS.values():
    METRIC_COLUMNS.extend(cols)

# 类别名称映射
CLASS_NAMES = {
    0: "background",
    1: "Cropland",
    2: "Forest",
    3: "Grassland",
    4: "Shrubland",
    5: "Wetland",
    6: "Water",
    7: "Built-up",
    8: "Bareland"
}


@dataclass
class PostprocessConfig:
    """
    后处理分析配置
    
    属性:
        result_root: 结果文件根目录
        metric_types: 要分析的度量类型列表
        task_names: 要分析的任务名称列表
        batch_sizes: 要分析的批次大小列表
        correlation_methods: 相关性计算方法列表
        accuracy_columns: 精度指标列名列表
        output_dir: 输出目录
        figure_format: 图片格式
        figure_dpi: 图片分辨率
    """
    # 路径配置
    result_root: str = "./results"
    output_dir: str = "./analysis"
    
    # 分析范围
    metric_types: List[str] = field(default_factory=lambda: ["FD", "DS", "GBC"])
    task_names: List[str] = field(default_factory=lambda: list(TASK_CONFIGS.keys()))
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4])
    
    # 相关性参数
    correlation_methods: List[str] = field(default_factory=lambda: ["pearson", "spearman"])
    accuracy_columns: List[str] = field(default_factory=lambda: DEFAULT_ACCURACY_COLUMNS)
    
    # 可视化参数
    figure_format: str = "png"
    figure_dpi: int = 150
    colormap: str = "RdBu_r"
    
    # 字体设置
    font_family: str = "sans-serif"
    font_size: int = 12
    
    def __post_init__(self):
        """初始化后处理"""
        self._validate()
    
    def _validate(self):
        """验证配置参数"""
        for metric in self.metric_types:
            if metric not in METRIC_TYPES:
                raise ValueError(f"未知的度量类型: {metric}, 可选: {METRIC_TYPES}")
        
        for task in self.task_names:
            if task not in TASK_CONFIGS:
                raise ValueError(f"未知的任务: {task}, 可选: {list(TASK_CONFIGS.keys())}")
    
    def get_result_path(self, metric_type: str, task_name: str) -> str:
        """
        获取结果文件路径
        
        参数:
            metric_type: 度量类型
            task_name: 任务名称
        
        返回:
            结果目录路径
        """
        return os.path.join(self.result_root, metric_type, task_name)
    
    def get_output_path(self, *parts: str) -> str:
        """
        获取输出文件路径
        
        参数:
            *parts: 路径组成部分
        
        返回:
            输出文件路径
        """
        return os.path.join(self.output_dir, *parts)
    
    def get_metric_score_columns(self, metric_type: str) -> List[str]:
        """
        获取指定度量类型的分数列名
        
        参数:
            metric_type: 度量类型
        
        返回:
            分数列名列表
        """
        return METRIC_SCORE_COLUMNS.get(metric_type, [])
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """
        获取任务信息
        
        参数:
            task_name: 任务名称
        
        返回:
            任务信息字典
        """
        return TASK_CONFIGS.get(task_name, {})
    
    @classmethod
    def from_args(cls, args: Optional[List[str]] = None) -> "PostprocessConfig":
        """
        从命令行参数创建配置
        
        参数:
            args: 命令行参数列表
        
        返回:
            PostprocessConfig实例
        """
        parser = argparse.ArgumentParser(
            description="迁移度量结果后处理分析工具",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # 路径参数
        parser.add_argument("--result_root", type=str, default=cls.result_root,
                          help="结果文件根目录")
        parser.add_argument("--output_dir", type=str, default=cls.output_dir,
                          help="输出目录")
        
        # 分析范围
        parser.add_argument("--metric_types", type=str, nargs="+",
                          default=["FD", "DS", "GBC"],
                          help="要分析的度量类型")
        parser.add_argument("--task_names", type=str, nargs="+",
                          default=list(TASK_CONFIGS.keys()),
                          help="要分析的任务名称")
        parser.add_argument("--batch_sizes", type=int, nargs="+",
                          default=[1, 4],
                          help="要分析的批次大小")
        
        # 相关性参数
        parser.add_argument("--correlation_methods", type=str, nargs="+",
                          default=["pearson", "spearman"],
                          help="相关性计算方法")
        
        # 可视化参数
        parser.add_argument("--figure_format", type=str, default="png",
                          choices=["png", "pdf", "svg"],
                          help="图片格式")
        parser.add_argument("--figure_dpi", type=int, default=150,
                          help="图片分辨率")
        
        parsed = parser.parse_args(args)
        
        return cls(
            result_root=parsed.result_root,
            output_dir=parsed.output_dir,
            metric_types=parsed.metric_types,
            task_names=parsed.task_names,
            batch_sizes=parsed.batch_sizes,
            correlation_methods=parsed.correlation_methods,
            figure_format=parsed.figure_format,
            figure_dpi=parsed.figure_dpi,
        )


def print_config(config: PostprocessConfig):
    """打印配置摘要"""
    print("=" * 60)
    print("后处理分析配置")
    print("=" * 60)
    print(f"结果根目录: {config.result_root}")
    print(f"输出目录: {config.output_dir}")
    print(f"度量类型: {config.metric_types}")
    print(f"任务名称: {config.task_names}")
    print(f"批次大小: {config.batch_sizes}")
    print(f"相关性方法: {config.correlation_methods}")
    print("=" * 60)


if __name__ == "__main__":
    config = PostprocessConfig.from_args()
    print_config(config)
