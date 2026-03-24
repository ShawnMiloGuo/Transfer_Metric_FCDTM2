#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DS (Dispersion Score) 度量实现

基于类别间特征分散度的度量，衡量目标域中不同类别特征的分离程度。
"""

import torch
import numpy as np
from typing import List
from tqdm import tqdm

from .base import BaseMetric, MetricResult, calculate_dispersion_score
from feature_extractor import DSFeatureExtractor
from model import ModelManager


class DSMetric(BaseMetric):
    """
    Dispersion Score 度量计算器
    
    计算目标域中各类别特征的分散程度，分散度越高表示特征区分性越好。
    """
    
    METRIC_NAME = "DS"
    
    # 结果列名定义（与原始代码顺序一致）
    COLUMN_NAMES = [
        # 源域指标
        "OA_source", "F1_source", "mIoU_source", "precision_source", "recall_source",
        # 目标域指标
        "OA_target", "F1_target", "mIoU_target", "precision_target", "recall_target",
        # 增量指标
        "OA_delta", "F1_delta", "mIoU_delta", "precision_delta", "recall_delta",
        # 分散度分数
        "dispersion_score", "log_dispersion_score",
        "weighted_dispersion_score", "weighted_log_dispersion_score",
    ]
    
    METRIC_PLOT_INDICES = [16, 17]  # dispersion_score
    ACCURACY_PLOT_INDICES = [10, 11]  # OA_delta, F1_delta
    
    def compute(
        self,
        model: torch.nn.Module,
        model_manager: ModelManager,
        source_loader,
        target_loader
    ) -> List[MetricResult]:
        """
        计算DS度量
        
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
        extractor = DSFeatureExtractor(model, device)
        
        # 获取配置参数
        use_pred = self.config.use_prediction_labels
        max_images = self.config.max_images
        process_all = self.config.process_all_target
        num_classes = 2
        
        # 获取权重差异用于加权
        weight_diff = model_manager.get_last_layer_weight_diff(model)
        weight_abs_norm = weight_diff['normalized_absolute']
        weight_dict = {i: weight_abs_norm.numpy() for i in range(num_classes)}
        
        # ========== 提取源域特征 ==========
        print("提取源域特征（按类别）...")
        source_metrics, source_class_stats, _ = extractor.extract_by_class(
            source_loader,
            use_prediction_labels=False,
            max_images=max_images,
            single_batch=False
        )
        
        # ========== 处理目标域 ==========
        n_batches = len(target_loader)
        
        for batch_idx in tqdm(range(n_batches), desc="计算DS度量"):
            
            target_metrics, target_class_stats, _ = extractor.extract_by_class(
                iter(target_loader),
                use_prediction_labels=use_pred,
                max_images=max_images,
                single_batch=not process_all
            )
            
            # 计算全局均值
            total_samples = sum(s.num_samples for s in target_class_stats.values())
            if total_samples > 0:
                overall_mean = sum(
                    target_class_stats[i].mean * target_class_stats[i].num_samples
                    for i in target_class_stats.keys()
                ) / total_samples
            else:
                overall_mean = np.zeros(64)  # 默认特征维度
            
            # 计算样本数
            class_samples = {i: target_class_stats[i].num_samples for i in target_class_stats.keys()}
            
            # 计算分散度
            class_means = {i: target_class_stats[i].mean for i in target_class_stats.keys()}
            
            dispersion, log_dispersion = calculate_dispersion_score(
                overall_mean, class_means, class_samples
            )
            
            weighted_disp, weighted_log_disp = calculate_dispersion_score(
                overall_mean, class_means, class_samples, weight_dict
            )
            
            # 创建结果（与原始代码结构一致）
            result = MetricResult(
                source_domain=self.config.source_dataset,
                target_domain=self.config.target_dataset,
                class_index=0,
                class_name="",
                # 源域指标
                OA_source=source_metrics.overall_accuracy,
                F1_source=source_metrics.f1_score,
                mIoU_source=source_metrics.mean_iou,
                precision_source=source_metrics.precision,
                recall_source=source_metrics.recall,
                # 目标域指标
                OA_target=target_metrics.overall_accuracy,
                F1_target=target_metrics.f1_score,
                mIoU_target=target_metrics.mean_iou,
                precision_target=target_metrics.precision,
                recall_target=target_metrics.recall,
                # 增量指标
                OA_delta=source_metrics.overall_accuracy - target_metrics.overall_accuracy,
                F1_delta=source_metrics.f1_score - target_metrics.f1_score,
                mIoU_delta=source_metrics.mean_iou - target_metrics.mean_iou,
                precision_delta=source_metrics.precision - target_metrics.precision,
                recall_delta=source_metrics.recall - target_metrics.recall,
                # 度量分数
                metric_scores={
                    "dispersion_score": dispersion,
                    "log_dispersion_score": log_dispersion,
                    "weighted_dispersion_score": weighted_disp,
                    "weighted_log_dispersion_score": weighted_log_disp,
                }
            )
            
            self.add_result(result)
            
            if process_all:
                break
        
        return self.results
