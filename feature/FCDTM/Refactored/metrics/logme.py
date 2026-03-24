#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LogME (Log Maximum Evidence) 度量实现

基于贝叶斯推理的迁移学习度量方法。通过计算特征与标签之间的最大证据
（Maximum Evidence），评估预训练特征在目标任务上的可迁移性。

参考文献:
    You, K., et al. "LogME: Practical Assessment of Pre-trained Models 
    for Transfer Learning." ICML 2021.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.linalg import solve_triangular
from tqdm import tqdm

from .base import BaseMetric, MetricResult
from feature_extractor import LogMEFeatureExtractor
from model import ModelManager


def compute_logme(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
    max_iter: int = 100,
    tol: float = 1e-5
) -> Dict[str, float]:
    """
    计算 Log Maximum Evidence
    
    使用高斯过程回归模型，通过边缘似然最大化来评估特征质量。
    
    参数:
        features: 特征矩阵 [N, D]
        labels: 标签向量 [N]
        num_classes: 类别数
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回:
        LogME 分数及相关指标
    """
    results = {}
    n_samples, n_features = features.shape
    
    # 归一化特征
    features = features - np.mean(features, axis=0)
    features = features / (np.std(features, axis=0) + 1e-8)
    
    # 计算核矩阵
    K = features @ features.T  # [N, N]
    
    # 添加正则化项
    alpha = 1.0  # 初始噪声方差
    beta = 1.0   # 初始信号方差
    
    # 迭代优化
    for i in range(max_iter):
        # 计算后验协方差
        try:
            # 使用 Cholesky 分解提高数值稳定性
            L = np.linalg.cholesky(beta * K + alpha * np.eye(n_samples) + 1e-6 * np.eye(n_samples))
            
            # 计算 log evidence
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # 对于分类任务，将标签转换为 one-hot
            total_logme = 0.0
            for c in range(num_classes):
                binary_labels = (labels == c).astype(np.float64)
                
                # 求解线性系统
                alpha_vec = solve_triangular(L, binary_labels, lower=True)
                alpha_vec = solve_triangular(L.T, alpha_vec, lower=False)
                
                # 计算证据
                data_fit = np.dot(binary_labels, alpha_vec)
                evidence = -0.5 * (data_fit + log_det + n_samples * np.log(2 * np.pi))
                total_logme += evidence
            
            # 更新超参数（简化的 EM 更新）
            gamma = n_samples - alpha * np.trace(np.linalg.solve(L @ L.T, np.eye(n_samples)))
            new_alpha = gamma / (np.dot(alpha_vec, K @ alpha_vec) + 1e-8)
            
            if abs(new_alpha - alpha) < tol:
                break
            
            alpha = new_alpha
            
        except np.linalg.LinAlgError:
            # 如果 Cholesky 分解失败，使用伪逆
            alpha = alpha * 0.9
            continue
    
    results['LogME_score'] = float(total_logme / num_classes)
    
    return results


def compute_logme_regression(
    features: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-5
) -> Dict[str, float]:
    """
    计算回归任务的 LogME（用于连续标签）
    
    参数:
        features: 特征矩阵 [N, D]
        labels: 标签向量 [N]
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回:
        LogME 分数
    """
    n_samples, n_features = features.shape
    
    # 归一化
    features = features - np.mean(features, axis=0)
    features = features / (np.std(features, axis=0) + 1e-8)
    labels = labels - np.mean(labels)
    
    # 特征分解
    try:
        U, S, Vt = np.linalg.svd(features, full_matrices=False)
    except:
        return {'LogME_regression': 0.0}
    
    # 初始化超参数
    alpha = 1.0
    beta = 1.0
    
    for i in range(max_iter):
        # 计算证据
        gamma = np.sum(S ** 2 / (S ** 2 + alpha / beta + 1e-8))
        
        # 后验均值
        sigma_sq = 1.0 / (beta + alpha * S ** 2)
        mu = beta * sigma_sq * S * (U.T @ labels)
        
        # 更新参数
        new_alpha = gamma / (np.sum(mu ** 2) + 1e-8)
        new_beta = (n_samples - gamma) / (np.sum((labels - features @ Vt.T @ mu) ** 2) + 1e-8)
        
        if abs(new_alpha - alpha) < tol and abs(new_beta - beta) < tol:
            break
        
        alpha = new_alpha
        beta = new_beta
    
    # 计算最终证据
    log_evidence = 0.5 * (
        n_samples * np.log(beta) +
        n_features * np.log(alpha) -
        np.sum(np.log(alpha + beta * S ** 2 + 1e-8)) -
        beta * np.sum((labels - features @ Vt.T @ mu) ** 2) -
        alpha * np.sum(mu ** 2) -
        n_samples * np.log(2 * np.pi)
    )
    
    return {'LogME_regression': float(log_evidence)}


def compute_logme_fast(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    快速 LogME 计算（使用特征分解）
    
    通过 SVD 分解加速计算，适用于大规模数据。
    
    参数:
        features: 特征矩阵 [N, D]
        labels: 标签向量 [N]
        num_classes: 类别数
    
    返回:
        LogME 分数
    """
    results = {}
    n_samples, n_features = features.shape
    
    # 归一化
    features = features - np.mean(features, axis=0)
    features = features / (np.std(features, axis=0) + 1e-8)
    
    # SVD 分解
    try:
        U, S, Vt = np.linalg.svd(features, full_matrices=False)
    except:
        results['LogME_fast'] = 0.0
        return results
    
    # 限制奇异值数量
    k = min(100, len(S))
    U_k = U[:, :k]
    S_k = S[:k]
    
    total_logme = 0.0
    
    for c in range(num_classes):
        binary_labels = (labels == c).astype(np.float64)
        
        # 计算 alpha
        alpha_sq = np.dot(U_k.T, binary_labels) ** 2
        
        # 计算证据
        log_evidence = 0.5 * (
            np.sum(alpha_sq / (S_k ** 2 + 1e-8)) -
            np.sum(np.log(S_k ** 2 + 1e-8)) -
            n_samples * np.log(2 * np.pi)
        )
        
        total_logme += log_evidence
    
    results['LogME_fast'] = float(total_logme / num_classes)
    
    return results


def compute_feature_statistics(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    计算特征统计量
    
    包括类内/类间距离、特征可分性等指标。
    
    参数:
        source_features: 源域特征
        target_features: 目标域特征
        source_labels: 源域标签
        target_labels: 目标域标签
        num_classes: 类别数
    
    返回:
        特征统计量字典
    """
    results = {}
    
    # 类内距离（目标域）
    target_within_class = []
    for c in range(num_classes):
        mask = target_labels == c
        if np.sum(mask) > 1:
            class_features = target_features[mask]
            center = np.mean(class_features, axis=0)
            distances = np.linalg.norm(class_features - center, axis=1)
            target_within_class.append(np.mean(distances))
    
    if target_within_class:
        results['target_within_class_dist'] = np.mean(target_within_class)
    
    # 类间距离（目标域）
    class_centers = []
    for c in range(num_classes):
        mask = target_labels == c
        if np.sum(mask) > 0:
            class_centers.append(np.mean(target_features[mask], axis=0))
    
    if len(class_centers) >= 2:
        between_class_dist = np.linalg.norm(class_centers[0] - class_centers[1])
        results['target_between_class_dist'] = between_class_dist
    
    # Fisher 可分性判据
    if target_within_class and len(class_centers) >= 2:
        fisher_ratio = between_class_dist / (np.mean(target_within_class) + 1e-8)
        results['target_fisher_ratio'] = fisher_ratio
    
    # 特征有效性（源域到目标域的转移）
    source_center = np.mean(source_features, axis=0)
    target_center = np.mean(target_features, axis=0)
    results['center_shift'] = np.linalg.norm(source_center - target_center)
    
    return results


class LogMEMetric(BaseMetric):
    """
    LogME (Log Maximum Evidence) 度量计算器
    
    基于贝叶斯推理计算特征与标签之间的最大证据，
    用于评估预训练特征的迁移能力。
    """
    
    METRIC_NAME = "LogME"
    
    COLUMN_NAMES = [
        # 源域指标
        "OA_source", "F1_source", "mIoU_source", "precision_source", "recall_source",
        # 目标域指标
        "OA_target", "F1_target", "mIoU_target", "precision_target", "recall_target",
        # 增量指标
        "OA_delta", "F1_delta", "mIoU_delta", "precision_delta", "recall_delta",
        # LogME 度量分数
        "LogME_score", "LogME_fast",
        # 特征统计量
        "target_within_class_dist", "target_between_class_dist", "target_fisher_ratio",
        "center_shift",
    ]
    
    METRIC_PLOT_INDICES = [16, 17]  # LogME_score, LogME_fast
    ACCURACY_PLOT_INDICES = [12, 13]  # OA_delta, F1_delta
    
    def __init__(self, config):
        super().__init__(config)
        self.max_iter = getattr(config, 'logme_max_iter', 100)
        self.sample_size = getattr(config, 'logme_sample_size', 3000)
    
    def compute(
        self,
        model: torch.nn.Module,
        model_manager: ModelManager,
        source_loader,
        target_loader
    ) -> List[MetricResult]:
        """
        计算 LogME 度量
        
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
        extractor = LogMEFeatureExtractor(model, device)
        
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
        
        # 转换为数组
        source_feat_arr = np.array(source_features, dtype=np.float32)
        source_label_arr = np.array(source_labels, dtype=np.int64)
        
        # 下采样以加速计算
        if len(source_feat_arr) > self.sample_size:
            idx = np.random.choice(len(source_feat_arr), self.sample_size, replace=False)
            source_feat_arr = source_feat_arr[idx]
            source_label_arr = source_label_arr[idx]
        
        # ========== 处理目标域 ==========
        n_batches = len(target_loader)
        
        for batch_idx in tqdm(range(n_batches), desc="计算LogME度量"):
            
            target_metrics, target_features, target_labels = extractor.extract_with_labels(
                iter(target_loader),
                use_prediction_labels=use_pred,
                max_images=max_images,
                single_batch=not process_all
            )
            
            # 转换为数组
            target_feat_arr = np.array(target_features, dtype=np.float32)
            target_label_arr = np.array(target_labels, dtype=np.int64)
            
            # 下采样
            if len(target_feat_arr) > self.sample_size:
                idx = np.random.choice(len(target_feat_arr), self.sample_size, replace=False)
                target_feat_arr = target_feat_arr[idx]
                target_label_arr = target_label_arr[idx]
            
            # 计算 LogME 分数
            try:
                logme_results = compute_logme(
                    target_feat_arr, target_label_arr,
                    num_classes=2,
                    max_iter=self.max_iter
                )
            except Exception as e:
                print(f"LogME计算错误: {e}")
                logme_results = {'LogME_score': 0.0}
            
            # 计算快速 LogME
            try:
                logme_fast_results = compute_logme_fast(
                    target_feat_arr, target_label_arr,
                    num_classes=2
                )
            except Exception as e:
                print(f"LogME快速计算错误: {e}")
                logme_fast_results = {'LogME_fast': 0.0}
            
            # 计算特征统计量
            try:
                feature_stats = compute_feature_statistics(
                    source_feat_arr, target_feat_arr,
                    source_label_arr, target_label_arr,
                    num_classes=2
                )
            except Exception as e:
                print(f"特征统计计算错误: {e}")
                feature_stats = {}
            
            # 创建结果
            result = MetricResult(
                source_domain=self.config.source_dataset,
                target_domain=self.config.target_domain,
                class_index=0,
                class_name="",
                batch_index=batch_idx,
                source_accuracy=source_metrics.overall_accuracy,
                source_f1=source_metrics.f1_score,
                source_precision=source_metrics.precision,
                target_accuracy=target_metrics.overall_accuracy,
                target_f1=target_metrics.f1_score,
                target_precision=target_metrics.precision,
                accuracy_delta=source_metrics.overall_accuracy - target_metrics.overall_accuracy,
                f1_delta=source_metrics.f1_score - target_metrics.f1_score,
                precision_delta=source_metrics.precision - target_metrics.precision,
                metric_scores={
                    "mIoU_source": source_metrics.mean_iou,
                    "recall_source": source_metrics.recall,
                    "mIoU_target": target_metrics.mean_iou,
                    "precision_target": target_metrics.precision,
                    "recall_target": target_metrics.recall,
                    "mIoU_delta": source_metrics.mean_iou - target_metrics.mean_iou,
                    "recall_delta": source_metrics.recall - target_metrics.recall,
                    **logme_results,
                    **logme_fast_results,
                    **feature_stats,
                }
            )
            
            self.add_result(result)
            
            if process_all:
                break
        
        return self.results
