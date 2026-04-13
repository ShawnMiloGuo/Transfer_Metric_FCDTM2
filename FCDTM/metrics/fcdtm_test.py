#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCDTM-Test (Fréchet Class Difference Transfer Metric - Test) 度量实现

这是研发FCDTM算法过程中的终极测试模型。
包含了所有可能的组合方式，用于测试和对比不同配置的效果。

FCDTM是FD度量的改进版本，专注于mean_dif_absolute_y0_y1_diff组合方式。
该方式在实验中表现最佳，能更准确地预测迁移学习效果。

核心思想：
- 使用均值差异的绝对值 (|mean_t - mean_s|)
- 乘以类别权重差异 (y0_y1_diff)，衡量不同类别间的特征分布变化
"""

import torch
import numpy as np
from typing import List
from tqdm import tqdm

from .base import BaseMetric, MetricResult, calculate_frechet_distance
from feature_extractor import FDFeatureExtractor
from model import ModelManager


class FCDTMTestMetric(BaseMetric):
    """
    FCDTM-Test 度量计算器
    
    FCDTM-Test 是研发 FCDTM 算法过程中的终极测试模型。
    包含了所有可能的组合方式，用于测试和对比。
    
    最终确定的最优组合方式：
    - 均值差异绝对值 × 权重差异 (y0_y1_diff)
    
    详细的组合方式包括：
    1. 均值差异基础统计（4种）
    2. 均值差异 × 权重差异（16种）
    3. FD分数（5种）
    """
    
    METRIC_NAME = "FCDTM-Test"
    
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
        # 均值差异 - 基础统计
        "mean_dif_absolute_sum", "mean_dif_absolute_abs_sum",
        "mean_dif_relative_sum", "mean_dif_relative_abs_sum",
        # 均值差异 × 权重差异 (y0_y1_diff)
        "mean_dif_absolute_y0_y1_diff", "mean_dif_absolute_abs_y0_y1_diff",
        "mean_dif_relative_y0_y1_diff", "mean_dif_relative_abs_y0_y1_diff",
        # 均值差异 × 权重差异绝对值 (y0_y1_diff_abs)
        "mean_dif_absolute_y0_y1_diff_abs", "mean_dif_absolute_abs_y0_y1_diff_abs",
        "mean_dif_relative_y0_y1_diff_abs", "mean_dif_relative_abs_y0_y1_diff_abs",
        # 均值差异 × 归一化权重差异
        "mean_dif_absolute_y0_y1_diff_normalized", "mean_dif_absolute_abs_y0_y1_diff_normalized",
        "mean_dif_relative_y0_y1_diff_normalized", "mean_dif_relative_abs_y0_y1_diff_normalized",
        # 均值差异 × 归一化权重差异绝对值
        "mean_dif_absolute_y0_y1_diff_abs_normalized", "mean_dif_absolute_abs_y0_y1_diff_abs_normalized",
        "mean_dif_relative_y0_y1_diff_abs_normalized", "mean_dif_relative_abs_y0_y1_diff_abs_normalized",
        # FD 分数
        "FD_sum", "FD_y0_y1_diff", "FD_y0_y1_diff_abs",
        "FD_y0_y1_diff_normalized", "FD_y0_y1_diff_abs_normalized",
    ]
    
    # 索引说明（相对于COLUMN_NAMES）:
    # 0-2: 增量指标
    # 3-5: 相对增量指标
    # 6-8: 源域指标
    # 9-11: 目标域指标
    # 12-15: 均值差异基础统计
    # 16-31: 均值差异×权重差异
    # 32-36: FD分数
    METRIC_PLOT_INDICES = [32, 33, 34, 36]  # FD_sum, FD_y0_y1_diff, FD_y0_y1_diff_abs, FD_y0_y1_diff_abs_normalized
    ACCURACY_PLOT_INDICES = [0, 1]  # OA_delta, F1_delta
    
    def compute(
        self,
        model: torch.nn.Module,
        model_manager: ModelManager,
        source_loader,
        target_loader
    ) -> List[MetricResult]:
        """
        计算FCDTM-Test度量
        
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
            
            for batch_idx in tqdm(range(n_batches), desc="计算FCDTM-Test度量"):
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
        计算单个FCDTM-Test结果
        
        参数:
            source_stats: 源域特征统计
            target_stats: 目标域特征统计
            source_metrics: 源域评估指标
            target_metrics: 目标域评估指标
            weight_diff: 权重差异字典
        
        返回:
            MetricResult对象
        """
        # 计算均值差异（与原始代码一致）
        # mean_dif_absolute = mean_t - mean_s (注意顺序)
        mean_dif_absolute = target_stats.mean - source_stats.mean
        mean_dif_absolute_abs = np.abs(mean_dif_absolute)
        mean_dif_relative = (target_stats.mean - source_stats.mean) / (source_stats.mean + 1e-8)
        mean_dif_relative_abs = np.abs(mean_dif_relative)
        
        # 均值差异字典
        mean_dif_dict = {
            'mean_dif_absolute': mean_dif_absolute,
            'mean_dif_absolute_abs': mean_dif_absolute_abs,
            'mean_dif_relative': mean_dif_relative,
            'mean_dif_relative_abs': mean_dif_relative_abs,
        }
        
        # 权重差异键名列表（与原始代码一致）
        weight_diff_keys = ['y0_y1_diff', 'y0_y1_diff_abs', 
                           'y0_y1_diff_normalized', 'y0_y1_diff_abs_normalized']
        
        # 计算均值差异 × 权重差异的组合
        mean_dif_weighted = {}
        for mean_key, mean_val in mean_dif_dict.items():
            for weight_key in weight_diff_keys:
                weight = weight_diff[weight_key].numpy()
                combined_key = f"{mean_key}_{weight_key}"
                mean_dif_weighted[combined_key] = float(np.sum(mean_val * weight))
        
        # 计算 FD 分数
        fd_sum = calculate_frechet_distance(
            source_stats.mean, target_stats.mean,
            source_stats.covariance, target_stats.covariance
        )
        
        # 计算加权 FD
        fd_weighted = {}
        for weight_key in weight_diff_keys:
            weight = weight_diff[weight_key].numpy()
            fd_w = calculate_frechet_distance(
                source_stats.mean * weight,
                target_stats.mean * weight,
                source_stats.covariance,
                target_stats.covariance
            )
            fd_weighted[f"FD_{weight_key}"] = fd_w
        
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
                # 均值差异基础统计
                "mean_dif_absolute_sum": float(np.sum(mean_dif_absolute)),
                "mean_dif_absolute_abs_sum": float(np.sum(mean_dif_absolute_abs)),
                "mean_dif_relative_sum": float(np.sum(mean_dif_relative)),
                "mean_dif_relative_abs_sum": float(np.sum(mean_dif_relative_abs)),
                # 均值差异 × 权重差异
                **mean_dif_weighted,
                # FD 分数
                "FD_sum": fd_sum,
                **fd_weighted
            }
        )
        
        return result
