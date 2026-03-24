#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GBC (Geometric Bayesian Classifier) 度量实现

基于几何贝叶斯分类器的度量，评估特征空间中类别可分性。
"""

import torch
from typing import List
from tqdm import tqdm

from .base import BaseMetric, MetricResult
from feature_extractor import GBCFeatureExtractor
from model import ModelManager

# 导入GBC计算函数
try:
    from metric_gbc import get_gbc_score
except ImportError:
    # 如果没有安装，提供警告
    print("警告: 未找到 metric_gbc 模块，GBC度量将不可用")
    def get_gbc_score(features, labels, mode):
        raise NotImplementedError("metric_gbc 模块未安装")


class GBCMetric(BaseMetric):
    """
    Geometric Bayesian Classifier 度量计算器
    
    计算特征空间中基于贝叶斯分类器的几何度量。
    """
    
    METRIC_NAME = "GBC"
    
    COLUMN_NAMES = [
        # 源域指标
        "OA_source", "F1_source", "mIoU_source", "precision_source", "recall_source",
        # 目标域指标
        "OA_target", "F1_target", "mIoU_target", "precision_target", "recall_target",
        # 增量指标
        "OA_delta", "F1_delta", "mIoU_delta", "precision_delta", "recall_delta",
        # GBC分数
        "diagonal_GBC", "spherical_GBC",
    ]
    
    METRIC_PLOT_INDICES = [18]  # spherical_GBC
    ACCURACY_PLOT_INDICES = [14, 15]  # OA_delta, F1_delta
    
    def compute(
        self,
        model: torch.nn.Module,
        model_manager: ModelManager,
        source_loader,
        target_loader
    ) -> List[MetricResult]:
        """
        计算GBC度量
        
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
        extractor = GBCFeatureExtractor(model, device)
        
        # 获取配置参数
        use_pred = self.config.use_prediction_labels
        max_images = self.config.max_images
        process_all = self.config.process_all_target
        
        # ========== 提取源域特征 ==========
        print("提取源域特征...")
        source_metrics, _, _ = extractor.extract_with_labels(
            source_loader,
            use_prediction_labels=False,
            max_images=max_images,
            single_batch=False
        )
        
        # ========== 处理目标域 ==========
        n_batches = len(target_loader)
        
        for batch_idx in tqdm(range(n_batches), desc="计算GBC度量"):
            
            target_metrics, target_features, target_labels = extractor.extract_with_labels(
                iter(target_loader),
                use_prediction_labels=use_pred,
                max_images=max_images,
                single_batch=not process_all
            )
            
            # 计算GBC分数
            try:
                diagonal_gbc = get_gbc_score(target_features, target_labels, 'diagonal')
                spherical_gbc = get_gbc_score(target_features, target_labels, 'spherical')
            except Exception as e:
                print(f"GBC计算错误: {e}")
                diagonal_gbc = 0.0
                spherical_gbc = 0.0
            
            # 创建结果
            result = MetricResult(
                source_domain=self.config.source_dataset,
                target_domain=self.config.target_dataset,
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
                    "diagonal_GBC": diagonal_gbc,
                    "spherical_GBC": spherical_gbc,
                }
            )
            
            self.add_result(result)
            
            if process_all:
                break
        
        return self.results
