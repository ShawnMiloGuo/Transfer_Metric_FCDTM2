#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
度量基类模块

定义度量计算的抽象接口和通用功能。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
import numpy as np


@dataclass
class MetricResult:
    """
    度量计算结果
    
    存储单次度量计算的所有结果数据。
    """
    # 基本信息
    source_domain: str
    target_domain: str
    class_index: int
    class_name: str
    
    # 源域指标
    OA_source: float = 0.0
    F1_source: float = 0.0
    mIoU_source: float = 0.0
    precision_source: float = 0.0
    recall_source: float = 0.0
    
    # 目标域指标
    OA_target: float = 0.0
    F1_target: float = 0.0
    mIoU_target: float = 0.0
    precision_target: float = 0.0
    recall_target: float = 0.0
    
    # 增量指标
    OA_delta: float = 0.0
    F1_delta: float = 0.0
    mIoU_delta: float = 0.0
    precision_delta: float = 0.0
    recall_delta: float = 0.0
    
    # 度量分数（由子类填充）
    metric_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_list(self, column_names: List[str]) -> List[Any]:
        """
        转换为列表格式
        
        参数:
            column_names: 列名列表
        
        返回:
            按列名顺序排列的值列表
        """
        result = []
        for col in column_names:
            if hasattr(self, col):
                result.append(getattr(self, col))
            elif col in self.metric_scores:
                result.append(self.metric_scores[col])
            else:
                result.append(None)
        return result


class BaseMetric(ABC):
    """
    度量计算基类
    
    定义度量计算的通用接口，所有具体度量方法继承此类。
    """
    
    # 度量类型名称
    METRIC_NAME: str = "base"
    
    # 结果列名定义（子类应覆盖）
    COLUMN_NAMES: List[str] = []
    
    # 绘图时的度量指标列索引
    METRIC_PLOT_INDICES: List[int] = []
    
    # 绘图时的精度指标列索引
    ACCURACY_PLOT_INDICES: List[int] = []
    
    def __init__(self, config):
        """
        初始化
        
        参数:
            config: 配置对象
        """
        self.config = config
        self.results: List[MetricResult] = []
    
    @abstractmethod
    def compute(
        self,
        model,
        model_manager,
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
        pass
    
    def get_column_names(self) -> List[str]:
        """获取结果列名"""
        return ["source", "target", "class_index", "class_name"] + self.COLUMN_NAMES
    
    def get_plot_indices(self) -> tuple:
        """
        获取绘图索引
        
        注意：返回的索引已经考虑了基础信息列的偏移量（4列）
        """
        # 基础信息列数量（source, target, class_index, class_name）
        BASE_COLUMN_OFFSET = 4
        
        # 将相对索引转换为绝对索引
        metric_indices = [idx + BASE_COLUMN_OFFSET for idx in self.METRIC_PLOT_INDICES]
        accuracy_indices = [idx + BASE_COLUMN_OFFSET for idx in self.ACCURACY_PLOT_INDICES]
        
        return metric_indices, accuracy_indices
    
    def clear_results(self):
        """清除结果"""
        self.results = []
    
    def add_result(self, result: MetricResult):
        """添加结果"""
        self.results.append(result)
    
    def get_results_as_rows(self) -> List[List]:
        """获取结果行列表"""
        columns = self.get_column_names()
        return [r.to_list(columns) for r in self.results]


# ============================================================================
# 距离计算工具函数
# ============================================================================

def calculate_frechet_distance(
    source_mean: np.ndarray,
    target_mean: np.ndarray,
    source_cov: np.ndarray,
    target_cov: np.ndarray
) -> float:
    """
    计算Fréchet距离
    
    公式: FD = ||μ_s - μ_t||² + Tr(Σ_s + Σ_t - 2√(Σ_s·Σ_t))
    
    参数:
        source_mean: 源域均值
        target_mean: 目标域均值
        source_cov: 源域协方差
        target_cov: 目标域协方差
    
    返回:
        Fréchet距离值
    """
    from scipy.linalg import sqrtm
    
    # 均值差的L2范数平方
    mean_diff_sq = np.sum((source_mean - target_mean) ** 2)
    
    # 协方差矩阵乘积的平方根
    cov_sqrt = sqrtm(source_cov @ target_cov)
    
    # 处理复数
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    
    # 迹项
    trace_term = np.trace(source_cov + target_cov - 2 * cov_sqrt)
    trace_term = abs(trace_term)
    
    return float(mean_diff_sq + trace_term)


def calculate_dispersion_score(
    overall_mean: np.ndarray,
    class_means: Dict[int, np.ndarray],
    class_samples: Dict[int, int],
    weights: Optional[Dict[int, np.ndarray]] = None
) -> tuple:
    """
    计算分散度分数
    
    参数:
        overall_mean: 全局均值
        class_means: 各类别均值
        class_samples: 各类别样本数
        weights: 特征权重
    
    返回:
        (原始分数, 对数分数)
    """
    n_classes = len(class_means)
    weighted_sum = 0.0
    
    for c in range(n_classes):
        diff = overall_mean - class_means[c]
        
        if weights and c in weights:
            diff = diff * weights[c]
        
        weighted_sum += class_samples[c] * np.linalg.norm(diff, ord=2)
    
    score = weighted_sum / (n_classes - 1)
    log_score = np.log(score) if score > 0 else float('-inf')
    
    return score, log_score
