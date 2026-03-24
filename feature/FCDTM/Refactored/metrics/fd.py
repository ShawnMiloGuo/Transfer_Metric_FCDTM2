#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FD (Fréchet Distance) 度量实现

基于特征分布的Fréchet距离，用于衡量源域和目标域特征分布的差异。
"""

import torch
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .base import BaseMetric, MetricResult, calculate_frechet_distance
from feature_extractor import FDFeatureExtractor
from model import ModelManager


class FDMetric(BaseMetric):
    """
    Fréchet Distance 度量计算器
    
    计算源域和目标域特征分布之间的Fréchet距离。
    支持加权计算，使用模型最后一层权重作为特征重要性权重。
    """
    
    METRIC_NAME = "FD"
    
    # 结果列名定义（与原始代码顺序一致）
    COLUMN_NAMES = [
        # 增量指标
        "OA_delta", "F1_delta", "precision_delta",
        "OA_delta_relative", "F1_delta_relative",
        # 源域指标
        "OA_source", "F1_source", "precision_source",
        # 目标域指标
        "OA_target", "F1_target", "precision_target",
        # 均值差异
        "mean_diff_sum", "mean_diff_abs_sum",
        "mean_diff_relative_sum", "mean_diff_relative_abs_sum",
        # FD分数
        "FD_score",
        # 加权FD分数
        "FD_raw_difference", "FD_absolute_difference", 
        "FD_normalized_difference", "FD_normalized_absolute",
    ]
    
    METRIC_PLOT_INDICES = [16, 17, 20, 22]  # FD相关列
    ACCURACY_PLOT_INDICES = [0, 1]  # OA_delta, F1_delta
    
    def compute(
        self,
        model: torch.nn.Module,
        model_manager: ModelManager,
        source_loader,
        target_loader
    ) -> List[MetricResult]:
        """
        计算FD度量
        
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
        extractor = FDFeatureExtractor(model, device)
        
        # 获取配置参数
        target_label = 1 if self.config.only_foreground else None
        use_pred = self.config.use_prediction_labels
        exclude_zeros = self.config.exclude_zero_features
        max_images = self.config.max_images
        process_all = self.config.process_all_target
        
        # 获取权重差异用于加权计算
        weight_diff = model_manager.get_last_layer_weight_diff(model)
        
        # ========== 提取源域特征 ==========
        print("提取源域特征...")
        source_metrics, source_stats, _ = extractor.extract(
            source_loader,
            target_label_index=target_label,
            use_prediction_labels=False,
            max_images=max_images,
            exclude_zero_features=exclude_zeros,
            single_batch=False
        )
        
        # ========== 处理目标域 ==========
        target_iter = iter(target_loader)
        
        # 如果处理全部目标域
        if process_all:
            print("提取目标域特征（全部）...")
            target_metrics, target_stats, _ = extractor.extract(
                target_iter,
                target_label_index=target_label,
                use_prediction_labels=use_pred,
                max_images=max_images,
                exclude_zero_features=exclude_zeros,
                single_batch=False
            )
        
        # 按批次计算
        n_batches = len(target_loader)
        for batch_idx in tqdm(range(n_batches), desc="计算FD度量"):
            
            if not process_all:
                target_metrics, target_stats, _ = extractor.extract(
                    iter(target_loader),
                    target_label_index=target_label,
                    use_prediction_labels=use_pred,
                    max_images=max_images,
                    exclude_zero_features=exclude_zeros,
                    single_batch=True
                )
            
            # 计算均值差异
            mean_diff = target_stats.mean - source_stats.mean
            mean_diff_abs = np.abs(mean_diff)
            mean_diff_rel = mean_diff / (source_stats.mean + 1e-8)
            mean_diff_rel_abs = np.abs(mean_diff_rel)
            
            # 计算FD
            fd_score = calculate_frechet_distance(
                source_stats.mean, target_stats.mean,
                source_stats.covariance, target_stats.covariance
            )
            
            # 计算加权FD
            fd_weighted = {}
            for key, weight in weight_diff.items():
                w = weight.numpy()
                fd_w = calculate_frechet_distance(
                    source_stats.mean * w,
                    target_stats.mean * w,
                    source_stats.covariance,
                    target_stats.covariance
                )
                fd_weighted[f"FD_{key}"] = fd_w
            
            # 计算相对变化
            oa_rel = (source_metrics.overall_accuracy - target_metrics.overall_accuracy) / (source_metrics.overall_accuracy + 1e-8)
            f1_rel = (source_metrics.f1_score - target_metrics.f1_score) / (source_metrics.f1_score + 1e-8)
            
            # 创建结果（与原始代码结构一致）
            result = MetricResult(
                source_domain=self.config.source_dataset,
                target_domain=self.config.target_dataset,
                class_index=0,  # 由外部填充
                class_name="",  # 由外部填充
                # 源域指标
                OA_source=source_metrics.overall_accuracy,
                F1_source=source_metrics.f1_score,
                precision_source=source_metrics.precision,
                # 目标域指标
                OA_target=target_metrics.overall_accuracy,
                F1_target=target_metrics.f1_score,
                precision_target=target_metrics.precision,
                # 增量指标
                OA_delta=source_metrics.overall_accuracy - target_metrics.overall_accuracy,
                F1_delta=source_metrics.f1_score - target_metrics.f1_score,
                precision_delta=source_metrics.precision - target_metrics.precision,
                # 其他度量分数
                metric_scores={
                    "OA_delta_relative": oa_rel,
                    "F1_delta_relative": f1_rel,
                    "mean_diff_sum": float(np.sum(mean_diff)),
                    "mean_diff_abs_sum": float(np.sum(mean_diff_abs)),
                    "mean_diff_relative_sum": float(np.sum(mean_diff_rel)),
                    "mean_diff_relative_abs_sum": float(np.sum(mean_diff_rel_abs)),
                    "FD_score": fd_score,
                    **fd_weighted
                }
            )
            
            self.add_result(result)
            
            if process_all:
                break
        
        return self.results
