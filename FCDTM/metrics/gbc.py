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
    
    # 结果列名定义（与原始代码顺序一致）
    COLUMN_NAMES = [
        # 源域指标 (使用 _s 后缀)
        "OA_s", "F1_s", "mIoU_s", "precision_s", "recall_s",
        # 目标域指标 (使用 _t 后缀)
        "OA_t", "F1_t", "mIoU_t", "precision_t", "recall_t",
        # 增量指标
        "OA_delta", "F1_delta", "mIoU_delta", "precision_delta", "recall_delta",
        # GBC分数
        "diagonal_GBC", "spherical_GBC",
    ]
    
    METRIC_PLOT_INDICES = [16]  # spherical_GBC
    ACCURACY_PLOT_INDICES = [10, 11]  # OA_delta, F1_delta
    
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
        if process_all:
            # 模式1：处理所有目标域数据
            print("提取目标域特征（全部）...")
            target_metrics, target_features, target_labels = extractor.extract_with_labels(
                target_loader,
                use_prediction_labels=use_pred,
                max_images=max_images,
                single_batch=False
            )
            
            result = self._compute_single_gbc(
                source_metrics, target_metrics,
                target_features, target_labels
            )
            self.add_result(result)
            
        else:
            # 模式2：按批次处理目标域
            print(f"按批次处理目标域 (batch_size={self.config.batch_size})...")
            
            target_iter = iter(target_loader)
            n_batches = len(target_loader)
            
            for batch_idx in tqdm(range(n_batches), desc="计算GBC度量"):
                try:
                    target_metrics, target_features, target_labels = extractor.extract_with_labels(
                        target_iter,
                        use_prediction_labels=use_pred,
                        max_images=self.config.batch_size,
                        single_batch=True
                    )
                    
                    result = self._compute_single_gbc(
                        source_metrics, target_metrics,
                        target_features, target_labels
                    )
                    self.add_result(result)
                    
                except StopIteration:
                    print(f"目标域数据已处理完毕，共 {batch_idx} 个批次")
                    break
        
        return self.results
    
    def _compute_single_gbc(
        self,
        source_metrics,
        target_metrics,
        target_features,
        target_labels
    ) -> MetricResult:
        """计算单个GBC结果"""
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
                "diagonal_GBC": diagonal_gbc,
                "spherical_GBC": spherical_gbc,
            }
        )
        
        return result
