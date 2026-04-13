#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
度量计算模块

包含七种迁移度量方法的实现:
- FD (Fréchet Distance) - 原始算法，仅输出FD_sum
- FCDTM (Fréchet Class Difference Transfer Metric) - 最优度量，专注mean_dif_absolute_y0_y1_diff
- FCDTM-Test (FCDTM Test) - 研发过程中的测试模型，包含所有组合方式
- DS (Dispersion Score)
- GBC (Geometric Bayesian Classifier)
- OTCE (Optimal Transport for Conditional Estimation)
- LogME (Log Maximum Evidence)
"""

from .base import BaseMetric, MetricResult
from .fd import FDMetric
from .fcdtm import FCDTMMetric
from .fcdtm_test import FCDTMTestMetric
from .ds import DSMetric
from .gbc import GBCMetric
from .otce import OTCEMetric
from .logme import LogMEMetric


def get_metric(metric_type: str):
    """
    获取度量计算器
    
    参数:
        metric_type: 度量类型 ("FD", "FCDTM", "FCDTM-Test", "DS", "GBC", "OTCE", "LogME")
    
    返回:
        度量计算器类
    """
    metrics = {
        "FD": FDMetric,           # 原始FD算法
        "FCDTM": FCDTMMetric,     # FCDTM最优度量
        "FCDTM-Test": FCDTMTestMetric,  # FCDTM测试模型
        "DS": DSMetric,
        "GBC": GBCMetric,
        "OTCE": OTCEMetric,
        "LogME": LogMEMetric,
    }
    
    if metric_type not in metrics:
        raise ValueError(f"未知的度量类型: {metric_type}, 可选: {list(metrics.keys())}")
    
    return metrics[metric_type]


__all__ = [
    'BaseMetric',
    'MetricResult',
    'FDMetric',
    'FCDTMMetric',
    'FCDTMTestMetric',
    'DSMetric',
    'GBCMetric',
    'OTCEMetric',
    'LogMEMetric',
    'get_metric'
]
