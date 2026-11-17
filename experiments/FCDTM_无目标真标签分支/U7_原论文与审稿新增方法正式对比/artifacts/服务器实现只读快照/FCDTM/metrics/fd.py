#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FD (Fréchet Distance) 度量实现 - 原始版本

基于特征分布的Fréchet距离，用于衡量源域和目标域特征分布的差异。
这是最原始的FD算法实现，仅输出FD_sum分数。
"""

import torch
import numpy as np
from typing import List
from tqdm import tqdm

from .base import BaseMetric, MetricResult, calculate_frechet_distance
from feature_extractor import FDFeatureExtractor
from model import ModelManager


class FDMetric(BaseMetric):
    """
    原始 Fréchet Distance 度量计算器
    
    仅计算源域和目标域特征分布之间的Fréchet距离。
    这是最初的算法实现，仅输出 FD_sum 分数。
    """
    
    METRIC_NAME = "FD"
    
    # 结果列名定义（原始版本，仅包含基本指标）
    COLUMN_NAMES = [
        # 增量指标
        "OA_delta", "F1_delta", "precision_delta",
        # 相对增量指标
        "OA_delta_relative", "F1_delta_relative", "precision_delta_relative",
        # 源域指标 (使用 _s 后缀)
        "OA_s", "F1_s", "precision_s",
        # 目标域指标 (使用 _t 后缀)
        "OA_t", "F1_t", "precision_t",
        # FD 分数（仅此一项）
        "FD_sum",
    ]
    
    # 绘图时的度量指标列索引
    METRIC_PLOT_INDICES = [12]  # FD_sum
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
        
        # ========== 提取源域特征（一次性提取所有源域特征） ==========
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
        if process_all:
            # 模式1：处理所有目标域数据，计算单个FD值
            print("提取目标域特征（全部）...")
            target_metrics, target_stats, _ = extractor.extract(
                target_loader,
                target_label_index=target_label,
                use_prediction_labels=use_pred,
                max_images=max_images,
                exclude_zero_features=exclude_zeros,
                single_batch=False
            )
            
            # 计算FD
            result = self._compute_single_fd(
                source_stats, target_stats,
                source_metrics, target_metrics
            )
            self.add_result(result)
            
        else:
            # 模式2：按批次处理目标域，每批次计算一个FD值
            print(f"按批次处理目标域 (batch_size={self.config.batch_size})...")
            
            # 创建目标域迭代器
            target_iter = iter(target_loader)
            n_batches = len(target_loader)
            
            for batch_idx in tqdm(range(n_batches), desc="计算FD度量"):
                try:
                    # 提取当前批次的特征
                    target_metrics, target_stats, _ = extractor.extract(
                        target_iter,
                        target_label_index=target_label,
                        use_prediction_labels=use_pred,
                        max_images=self.config.batch_size,
                        exclude_zero_features=exclude_zeros,
                        single_batch=True
                    )
                    
                    # 计算FD
                    result = self._compute_single_fd(
                        source_stats, target_stats,
                        source_metrics, target_metrics
                    )
                    self.add_result(result)
                    
                except StopIteration:
                    print(f"目标域数据已处理完毕，共 {batch_idx} 个批次")
                    break
        
        return self.results
    
    def _compute_single_fd(
        self,
        source_stats,
        target_stats,
        source_metrics,
        target_metrics
    ) -> MetricResult:
        """
        计算单个FD结果（原始版本）
        
        参数:
            source_stats: 源域特征统计
            target_stats: 目标域特征统计
            source_metrics: 源域评估指标
            target_metrics: 目标域评估指标
        
        返回:
            MetricResult对象
        """
        # 计算原始FD分数（Fréchet Distance）
        fd_sum = calculate_frechet_distance(
            source_stats.mean, target_stats.mean,
            source_stats.covariance, target_stats.covariance
        )
        
        # 计算相对变化
        oa_rel = (source_metrics.overall_accuracy - target_metrics.overall_accuracy) / (source_metrics.overall_accuracy + 1e-8)
        f1_rel = (source_metrics.f1_score - target_metrics.f1_score) / (source_metrics.f1_score + 1e-8)
        precision_rel = (source_metrics.precision - target_metrics.precision) / (source_metrics.precision + 1e-8)
        
        # 创建结果
        result = MetricResult(
            source_domain=self.config.source_dataset,
            target_domain=self.config.target_dataset,
            class_index=0,  # 由外部填充
            class_name="",  # 由外部填充
            # 源域指标 (使用 _s 后缀)
            OA_s=source_metrics.overall_accuracy,
            F1_s=source_metrics.f1_score,
            precision_s=source_metrics.precision,
            # 目标域指标 (使用 _t 后缀)
            OA_t=target_metrics.overall_accuracy,
            F1_t=target_metrics.f1_score,
            precision_t=target_metrics.precision,
            # 增量指标
            OA_delta=source_metrics.overall_accuracy - target_metrics.overall_accuracy,
            F1_delta=source_metrics.f1_score - target_metrics.f1_score,
            precision_delta=source_metrics.precision - target_metrics.precision,
            # 其他度量分数
            metric_scores={
                # 相对增量指标
                "OA_delta_relative": oa_rel,
                "F1_delta_relative": f1_rel,
                "precision_delta_relative": precision_rel,
                # FD 分数（仅此一项）
                "FD_sum": fd_sum,
            }
        )
        
        return result
