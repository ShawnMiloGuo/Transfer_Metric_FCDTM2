#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OTCE (Optimal Transport for Conditional Estimation) 度量实现

基于最优传输理论的迁移学习度量方法。通过计算源域和目标域特征分布之间的
Wasserstein 距离，结合条件分布估计迁移性能。

参考文献:
    Tan, J., et al. "OTCE: A Transferability Metric for Cross-Domain 
    Cross-Task Representations." CVPR 2021.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .base import BaseMetric, MetricResult
from feature_extractor import OTCEFeatureExtractor
from model import ModelManager


def compute_optimal_transport(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    reg: float = 1.0,
    max_iter: int = 100
) -> Tuple[float, np.ndarray]:
    """
    计算最优传输距离
    
    使用 Sinkhorn 算法计算源域和目标域之间的 Wasserstein 距离。
    
    参数:
        source_features: 源域特征 [N_s, D]
        target_features: 目标域特征 [N_t, D]
        source_labels: 源域标签 [N_s]
        target_labels: 目标域标签 [N_t]
        reg: 熵正则化参数
        max_iter: 最大迭代次数
    
    返回:
        (传输距离, 传输矩阵)
    """
    n_source = source_features.shape[0]
    n_target = target_features.shape[0]
    
    # 计算成本矩阵（特征距离）
    cost_matrix = cdist(source_features, target_features, metric='euclidean')
    
    # 归一化成本矩阵
    cost_matrix = cost_matrix / (cost_matrix.max() + 1e-8)
    
    # 均匀分布权重
    source_weights = np.ones(n_source) / n_source
    target_weights = np.ones(n_target) / n_target
    
    # Sinkhorn 算法
    transport_matrix = sinkhorn_knopp(
        source_weights, target_weights, cost_matrix, reg, max_iter
    )
    
    # 计算传输距离
    transport_distance = np.sum(transport_matrix * cost_matrix)
    
    return transport_distance, transport_matrix


def sinkhorn_knopp(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float,
    max_iter: int = 100,
    tol: float = 1e-9
) -> np.ndarray:
    """
    Sinkhorn-Knopp 算法计算最优传输
    
    参数:
        a: 源分布权重
        b: 目标分布权重
        M: 成本矩阵
        reg: 熵正则化参数
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回:
        传输矩阵
    """
    # 初始化
    K = np.exp(-M / reg)
    u = np.ones_like(a)
    
    for i in range(max_iter):
        u_prev = u.copy()
        
        # 更新
        v = b / (K.T @ u + 1e-8)
        u = a / (K @ v + 1e-8)
        
        # 检查收敛
        if np.max(np.abs(u - u_prev)) < tol:
            break
    
    # 计算传输矩阵
    transport_matrix = np.diag(u) @ K @ np.diag(v)
    
    return transport_matrix


def compute_conditional_ot(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    计算条件最优传输距离
    
    分别计算每个类别的最优传输距离，并结合类别先验得到综合度量。
    
    参数:
        source_features: 源域特征
        target_features: 目标域特征
        source_labels: 源域标签
        target_labels: 目标域标签
        num_classes: 类别数
    
    返回:
        各类OT距离和综合度量
    """
    results = {}
    class_distances = []
    
    # 全局最优传输
    global_dist, _ = compute_optimal_transport(
        source_features, target_features,
        source_labels, target_labels
    )
    results['OT_global'] = float(global_dist)
    
    # 按类别计算条件最优传输
    for c in range(num_classes):
        source_mask = source_labels == c
        target_mask = target_labels == c
        
        if np.sum(source_mask) > 0 and np.sum(target_mask) > 0:
            class_source = source_features[source_mask]
            class_target = target_features[target_mask]
            
            # 下采样以加速计算
            if len(class_source) > 1000:
                idx = np.random.choice(len(class_source), 1000, replace=False)
                class_source = class_source[idx]
            if len(class_target) > 1000:
                idx = np.random.choice(len(class_target), 1000, replace=False)
                class_target = class_target[idx]
            
            # 类内特征距离（使用均值距离近似）
            class_dist = np.linalg.norm(
                np.mean(class_source, axis=0) - np.mean(class_target, axis=0)
            )
            results[f'OT_class_{c}'] = float(class_dist)
            class_distances.append(class_dist)
    
    # 类别加权平均
    if class_distances:
        results['OT_weighted'] = float(np.mean(class_distances))
    
    # OTCE 综合分数（距离越小，可迁移性越好）
    results['OTCE_score'] = float(global_dist)
    
    return results


def compute_domain_discrepancy(
    source_features: np.ndarray,
    target_features: np.ndarray
) -> Dict[str, float]:
    """
    计算域差异度量
    
    包括 MMD (Maximum Mean Discrepancy) 和 CORAL (Correlation Alignment)。
    
    参数:
        source_features: 源域特征
        target_features: 目标域特征
    
    返回:
        域差异度量字典
    """
    results = {}
    
    # 1. 均值差异
    source_mean = np.mean(source_features, axis=0)
    target_mean = np.mean(target_features, axis=0)
    results['mean_discrepancy'] = float(np.linalg.norm(source_mean - target_mean))
    
    # 2. 协方差差异 (CORAL)
    source_cov = np.cov(source_features, rowvar=False)
    target_cov = np.cov(target_features, rowvar=False)
    
    if source_cov.ndim == 0:
        source_cov = np.array([[source_cov]])
        target_cov = np.array([[target_cov]])
    
    results['coral_distance'] = float(np.linalg.norm(source_cov - target_cov, ord='fro'))
    
    # 3. MMD 近似（使用线性核）
    mmd = float(np.linalg.norm(source_mean - target_mean) ** 2)
    mmd += float(np.trace(source_cov + target_cov - 2 * np.sqrt(
        np.abs(source_cov @ target_cov)
    )))
    results['MMD_linear'] = mmd
    
    return results


class OTCEMetric(BaseMetric):
    """
    OTCE (Optimal Transport for Conditional Estimation) 度量计算器
    
    基于最优传输理论计算源域和目标域之间的分布差异，
    用于预测模型迁移性能。
    """
    
    METRIC_NAME = "OTCE"
    
    COLUMN_NAMES = [
        # 源域指标 (使用 _s 后缀)
        "OA_s", "F1_s", "mIoU_s", "precision_s", "recall_s",
        # 目标域指标 (使用 _t 后缀)
        "OA_t", "F1_t", "mIoU_t", "precision_t", "recall_t",
        # 增量指标
        "OA_delta", "F1_delta", "mIoU_delta", "precision_delta", "recall_delta",
        # OTCE 度量分数
        "OT_global", "OT_weighted", "OTCE_score",
        # 域差异度量
        "mean_discrepancy", "coral_distance", "MMD_linear",
        # 类别OT距离
        "OT_class_0", "OT_class_1",
    ]
    
    METRIC_PLOT_INDICES = [17]  # OTCE_score (索引17，从0开始)
    ACCURACY_PLOT_INDICES = [10, 11]  # OA_delta, F1_delta
    
    def __init__(self, config):
        super().__init__(config)
        self.ot_reg = getattr(config, 'ot_reg', 1.0)
        self.ot_max_iter = getattr(config, 'ot_max_iter', 100)
        self.sample_size = getattr(config, 'ot_sample_size', 2000)
    
    def compute(
        self,
        model: torch.nn.Module,
        model_manager: ModelManager,
        source_loader,
        target_loader
    ) -> List[MetricResult]:
        """
        计算 OTCE 度量
        
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
        extractor = OTCEFeatureExtractor(model, device)
        
        # 获取配置参数
        use_pred = self.config.use_prediction_labels
        max_images = self.config.max_images
        process_all = self.config.process_all_target
        
        # ========== 提取源域特征 ==========
        print("提取源域特征（带标签）...")
        source_metrics, source_features, source_labels = extractor.extract_with_labels(
            source_loader,
            use_prediction_labels=False,
            max_images=max_images,
            single_batch=False
        )
        
        # 下采样源域特征以加速计算
        if len(source_features) > self.sample_size:
            idx = np.random.choice(len(source_features), self.sample_size, replace=False)
            source_features = source_features[idx]
            source_labels = source_labels[idx]
        
        # ========== 处理目标域 ==========
        if process_all:
            # 模式1：处理所有目标域数据
            print("提取目标域特征（全部）...")
            target_metrics, target_features, target_labels = extractor.extract_with_labels(
                target_loader,
                use_prediction_labels=use_pred,
                max_images=max_images,
                single_batch=False
            )
            
            result = self._compute_single_otce(
                source_metrics, target_metrics,
                source_features, source_labels,
                target_features, target_labels
            )
            self.add_result(result)
            
        else:
            # 模式2：按批次处理目标域
            print(f"按批次处理目标域 (batch_size={self.config.batch_size})...")
            
            target_iter = iter(target_loader)
            n_batches = len(target_loader)
            
            for batch_idx in tqdm(range(n_batches), desc="计算OTCE度量"):
                try:
                    target_metrics, target_features, target_labels = extractor.extract_with_labels(
                        target_iter,
                        use_prediction_labels=use_pred,
                        max_images=self.config.batch_size,
                        single_batch=True
                    )
                    
                    result = self._compute_single_otce(
                        source_metrics, target_metrics,
                        source_features, source_labels,
                        target_features, target_labels
                    )
                    self.add_result(result)
                    
                except StopIteration:
                    print(f"目标域数据已处理完毕，共 {batch_idx} 个批次")
                    break
        
        return self.results
    
    def _compute_single_otce(
        self,
        source_metrics,
        target_metrics,
        source_features,
        source_labels,
        target_features,
        target_labels
    ) -> MetricResult:
        """计算单个OTCE结果"""
        # 下采样目标域特征
        if len(target_features) > self.sample_size:
            idx = np.random.choice(len(target_features), self.sample_size, replace=False)
            target_feat_arr = target_features[idx]
            target_label_arr = target_labels[idx]
        else:
            target_feat_arr = target_features
            target_label_arr = target_labels
        
        # 计算最优传输度量
        try:
            ot_results = compute_conditional_ot(
                source_features, target_feat_arr,
                source_labels, target_label_arr,
                num_classes=2
            )
        except Exception as e:
            print(f"OT计算错误: {e}")
            ot_results = {
                'OT_global': float(0.0),
                'OT_weighted': float(0.0),
                'OTCE_score': float(0.0),
                'OT_class_0': float(0.0),
                'OT_class_1': float(0.0),
            }
        
        # 计算域差异度量
        try:
            domain_results = compute_domain_discrepancy(
                source_features, target_feat_arr
            )
            # 确保所有值都是 Python float，避免 JSON 序列化问题
            domain_results = {k: float(v) for k, v in domain_results.items()}
        except Exception as e:
            print(f"域差异计算错误: {e}")
            domain_results = {
                'mean_discrepancy': float(0.0),
                'coral_distance': float(0.0),
                'MMD_linear': float(0.0),
            }
        
        # 创建结果
        result = MetricResult(
            source_domain=self.config.source_dataset,
            target_domain=self.config.target_dataset,
            class_index=0,
            class_name="",
            # 源域指标 (使用 _s 后缀)
            OA_s=source_metrics.overall_accuracy,
            F1_s=source_metrics.f1_score,
            mIoU_s=source_metrics.mean_iou,
            precision_s=source_metrics.precision,
            recall_s=source_metrics.recall,
            # 目标域指标 (使用 _t 后缀)
            OA_t=target_metrics.overall_accuracy,
            F1_t=target_metrics.f1_score,
            mIoU_t=target_metrics.mean_iou,
            precision_t=target_metrics.precision,
            recall_t=target_metrics.recall,
            # 增量指标
            OA_delta=source_metrics.overall_accuracy - target_metrics.overall_accuracy,
            F1_delta=source_metrics.f1_score - target_metrics.f1_score,
            mIoU_delta=source_metrics.mean_iou - target_metrics.mean_iou,
            precision_delta=source_metrics.precision - target_metrics.precision,
            recall_delta=source_metrics.recall - target_metrics.recall,
            # 度量分数
            metric_scores={
                **ot_results,
                **domain_results,
            }
        )
        
        return result
