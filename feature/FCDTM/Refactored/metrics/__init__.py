#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
度量计算模块

包含五种迁移度量方法的实现:
- FD (Fréchet Distance)
- DS (Dispersion Score)
- GBC (Geometric Bayesian Classifier)
- OTCE (Optimal Transport for Conditional Estimation)
- LogME (Log Maximum Evidence)
"""

from .base import BaseMetric, MetricResult
from .fd import FDMetric
from .ds import DSMetric
from .gbc import GBCMetric
from .otce import OTCEMetric
from .logme import LogMEMetric


def get_metric(metric_type: str):
    """
    获取度量计算器
    
    参数:
        metric_type: 度量类型 ("FD", "DS", "GBC", "OTCE", "LogME")
    
    返回:
        度量计算器类
    """
    metrics = {
        "FD": FDMetric,
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
    'DSMetric',
    'GBCMetric',
    'OTCEMetric',
    'LogMEMetric',
    'get_metric'
]
