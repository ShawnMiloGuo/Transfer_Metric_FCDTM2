#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCDTM (Fréchet Class Difference Transfer Metric) 度量实现

FCDTM是FD度量的改进版本，专注于mean_dif_absolute_y0_y1_diff组合方式。
该方式在实验中表现最佳，能更准确地预测迁移学习效果。

核心思想：
- 使用均值差异的绝对值 (|mean_t - mean_s|)
- 乘以类别权重差异 (y0_y1_diff)，衡量不同类别间的特征分布变化
"""

import torch
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .base import BaseMetric, MetricResult
from feature_extractor import FDFeatureExtractor
from model import ModelManager


class FCDTMMetric(BaseMetric):
    """
    FCDTM 度量计算器
    
    FCDTM (Fréchet Class Difference Transfer Metric) 是专门针对
    mean_dif_absolute_y0_y1_diff 组合方式的最优度量。
    
    该度量结合了:
    1. 特征分布的均值差异绝对值 (|mean_t - mean_s|)
    2. 模型最后一层权重的类别间差异 (y0_y1_diff)
    
    在实验中，该度量与迁移后精度下降的相关性最高。
    """
    
    METRIC_NAME = "FCDTM"
    
    # 结果列名定义
    COLUMN_NAMES = [
        # 增量指标
        "OA_delta", "F1_delta", "precision_delta",
        # 相对增量指标
        "OA_delta_relative", "F1_delta_relative", "precision_delta_relative",
        # 源域指标 (使用 _s 后缀)
        "OA_s", "F1_s", "precision_s",
        # 目标域指标 (使用 _t 后缀)
        "OA_t", "F1_t", "precision_t",
        # FCDTM 核心度量：均值差异 × 权重差异
        "mean_dif_absolute_y0_y1_diff",
    ]
    
    # 索引说明（相对于COLUMN_NAMES）:
    # 0-2: 增量指标
    # 3-5: 相对增量指标
    # 6-8: 源域指标
    # 9-11: 目标域指标
    # 12: FCDTM核心度量
    METRIC_PLOT_INDICES = [12]  # mean_dif_absolute_y0_y1_diff
    ACCURACY_PLOT_INDICES = [0, 1]  # OA_delta, F1_delta
    
    def compute(
        self,
        model: torch.nn.Module,
        model_manager: ModelManager,
        source_loader,
        target_loader
    ) -> List[MetricResult]:
        """
        计算FCDTM度量
        
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
            # 模式1：处理所有目标域数据，计算单个FCDTM值
            print("提取目标域特征（全部）...")
            target_metrics, target_stats, _ = extractor.extract(
                target_loader,
                target_label_index=target_label,
                use_prediction_labels=use_pred,
                max_images=max_images,
                exclude_zero_features=exclude_zeros,
                single_batch=False
            )
            
            # 计算FCDTM
            result = self._compute_single_fcdtm(
                source_stats, target_stats,
                source_metrics, target_metrics,
                weight_diff
            )
            self.add_result(result)
            
        else:
            # 模式2：按批次处理目标域，每批次计算一个FCDTM值
            print(f"按批次处理目标域 (batch_size={self.config.batch_size})...")
            
            # 创建目标域迭代器
            target_iter = iter(target_loader)
            n_batches = len(target_loader)
            
            for batch_idx in tqdm(range(n_batches), desc="计算FCDTM度量"):
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
                    
                    # 计算FCDTM
                    result = self._compute_single_fcdtm(
                        source_stats, target_stats,
                        source_metrics, target_metrics,
                        weight_diff
                    )
                    self.add_result(result)
                    
                except StopIteration:
                    print(f"目标域数据已处理完毕，共 {batch_idx} 个批次")
                    break
        
        return self.results
    
    def _compute_single_fcdtm(
        self,
        source_stats,
        target_stats,
        source_metrics,
        target_metrics,
        weight_diff: dict
    ) -> MetricResult:
        """
        计算单个FCDTM结果
        
        参数:
            source_stats: 源域特征统计
            target_stats: 目标域特征统计
            source_metrics: 源域评估指标
            target_metrics: 目标域评估指标
            weight_diff: 权重差异字典
        
        返回:
            MetricResult对象
        """
        # 计算均值差异绝对值
        # mean_dif_absolute = mean_t - mean_s (注意顺序)
        # 关键：这里使用 mean_dif_absolute 而不是 mean_dif_absolute_abs
        # 因为 FCDTM-Test 中 mean_dif_absolute_y0_y1_diff = sum((mean_t - mean_s) * weight)
        # 没有对均值差异取绝对值
        mean_dif_absolute = target_stats.mean - source_stats.mean
        
        # FCDTM 核心度量：均值差异 × 权重差异 (y0_y1_diff)
        # 与 FCDTM-Test 中的 mean_dif_absolute_y0_y1_diff 计算方式一致
        weight = weight_diff['y0_y1_diff'].numpy()
        mean_dif_absolute_y0_y1_diff = float(np.sum(mean_dif_absolute * weight))
        
        # 计算相对变化
        oa_rel = (source_metrics.overall_accuracy - target_metrics.overall_accuracy) / (source_metrics.overall_accuracy + 1e-8)
        f1_rel = (source_metrics.f1_score - target_metrics.f1_score) / (source_metrics.f1_score + 1e-8)
        precision_rel = (source_metrics.precision - target_metrics.precision) / (source_metrics.precision + 1e-8)
        
        # 创建结果
        result = MetricResult(
            source_domain=self.config.source_dataset,
            target_domain=self.config.target_dataset,
            class_index=0,  # 由外部填充
            class_name="",   # 由外部填充
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
                # 源域指标
                "OA_s": source_metrics.overall_accuracy,
                "F1_s": source_metrics.f1_score,
                "precision_s": source_metrics.precision,
                # 目标域指标
                "OA_t": target_metrics.overall_accuracy,
                "F1_t": target_metrics.f1_score,
                "precision_t": target_metrics.precision,
                # FCDTM 核心度量
                "mean_dif_absolute_y0_y1_diff": mean_dif_absolute_y0_y1_diff,
            }
        )
        
        return result
