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
        # 源域指标 (使用 _s 后缀)
        "OA_s", "F1_s", "mIoU_s", "precision_s", "recall_s",
        # 目标域指标 (使用 _t 后缀)
        "OA_t", "F1_t", "mIoU_t", "precision_t", "recall_t",
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
        if process_all:
            # 模式1：处理所有目标域数据
            print("提取目标域特征（全部）...")
            target_metrics, target_class_stats, _ = extractor.extract_by_class(
                target_loader,
                use_prediction_labels=use_pred,
                max_images=max_images,
                single_batch=False
            )
            
            result = self._compute_single_ds(
                source_metrics, target_metrics,
                target_class_stats, weight_dict
            )
            self.add_result(result)
            
        else:
            # 模式2：按批次处理目标域
            print(f"按批次处理目标域 (batch_size={self.config.batch_size})...")
            
            target_iter = iter(target_loader)
            n_batches = len(target_loader)
            
            for batch_idx in tqdm(range(n_batches), desc="计算DS度量"):
                try:
                    target_metrics, target_class_stats, _ = extractor.extract_by_class(
                        target_iter,
                        use_prediction_labels=use_pred,
                        max_images=self.config.batch_size,
                        single_batch=True
                    )
                    
                    result = self._compute_single_ds(
                        source_metrics, target_metrics,
                        target_class_stats, weight_dict
                    )
                    self.add_result(result)
                    
                except StopIteration:
                    print(f"目标域数据已处理完毕，共 {batch_idx} 个批次")
                    break
        
        return self.results
    
    def _compute_single_ds(
        self,
        source_metrics,
        target_metrics,
        target_class_stats: dict,
        weight_dict: dict
    ) -> MetricResult:
        """计算单个DS结果"""
        num_classes = 2
        
        # 计算全局均值
        total_samples = sum(s.num_samples for s in target_class_stats.values())
        if total_samples > 0:
            overall_mean = sum(
                target_class_stats[i].mean * target_class_stats[i].num_samples
                for i in target_class_stats.keys()
            ) / total_samples
        else:
            overall_mean = np.zeros(64)
        
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
                "dispersion_score": dispersion,
                "log_dispersion_score": log_dispersion,
                "weighted_dispersion_score": weighted_disp,
                "weighted_log_dispersion_score": weighted_log_disp,
            }
        )
        
        return result
