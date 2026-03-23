#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移学习度量指标计算模块

该模块实现了三种迁移学习度量指标的计算：
- FD (Fréchet Distance): 基于特征分布的Fréchet距离
- DS (Dispersion Score): 基于类别间特征分散度的度量
- GBC (Geometric Bayesian Classifier): 基于几何贝叶斯分类器的度量

作者: zt
创建日期: 2024-12-19
重构日期: 2024-xx-xx

主要功能:
1. 加载预训练的语义分割模型
2. 提取指定层的特征表示
3. 计算源域和目标域之间的迁移度量
4. 生成可视化分析结果
"""

import torch
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# 添加父目录到系统路径，以便导入自定义模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from component.utils import test_path_exist
from torchmetrics.functional.classification import (
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score
)
from tqdm import tqdm
from component.dataset import get_dataset_reader
from torch.utils import data
import csv
import matplotlib.pyplot as plt
import glob
from component.utils import save_log
import json
from scipy.linalg import sqrtm
import numpy as np
import argparse

from metric_gbc import get_gbc_score


# ============================================================================
# 常量定义
# ============================================================================

class MetricType(Enum):
    """迁移度量类型枚举"""
    FD = "FD"   # Fréchet Distance
    DS = "DS"   # Dispersion Score
    GBC = "GBC"  # Geometric Bayesian Classifier


class FeatureLayer(Enum):
    """特征提取层枚举"""
    DECODER_UP4 = "up4"      # 解码器第4层, 输出维度 [B, 64, 256, 256]
    DECODER_OUTC = "outc"    # 输出层, 输出维度 [B, num_classes, 256, 256]
    ENCODER_DOWN4 = "down4"  # 编码器第4层, 输出维度 [B, 512, 32, 32]


# 默认参数
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 0
DEFAULT_NUM_CLASSES = 2
DEFAULT_FEATURE_LAYER = FeatureLayer.DECODER_UP4.value
DEFAULT_MAX_IMAGES = 100  # 最大处理图像数量限制

# 类别名称映射
CLASS_NAMES = [
    "background",   # 0: 背景
    "Cropland",     # 1: 耕地
    "Forest",       # 2: 森林
    "Grassland",    # 3: 草地
    "Shrubland",    # 4: 灌木地
    "Wetland",      # 5: 湿地
    "Water",        # 6: 水体
    "Built-up",     # 7: 建筑用地
    "Bareland"      # 8: 裸地
]


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ClassificationMetrics:
    """分类评估指标数据类"""
    overall_accuracy: float
    f1_score: float
    mean_iou: float
    precision: float
    recall: float
    class_ious: Optional[List[float]] = None


@dataclass
class FeatureStatistics:
    """特征统计数据类"""
    mean: np.ndarray
    covariance: np.ndarray
    variance: np.ndarray
    features: np.ndarray
    num_samples: int


@dataclass
class TransferResult:
    """迁移度量结果数据类"""
    source_domain: str
    target_domain: str
    class_index: int
    class_name: str
    
    # 源域指标
    source_accuracy: float
    source_f1: float
    source_precision: float
    
    # 目标域指标
    target_accuracy: float
    target_f1: float
    target_precision: float
    
    # 迁移度量
    accuracy_delta: float
    f1_delta: float
    precision_delta: float
    
    # 额外的度量指标（根据不同方法存储不同内容）
    extra_metrics: Optional[Dict[str, float]] = None


# ============================================================================
# 全局变量（用于特征提取钩子）
# ============================================================================

# 存储钩子捕获的特征图
_feature_hook_output: Dict[str, torch.Tensor] = {}


# ============================================================================
# 核心评估函数
# ============================================================================

def calculate_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_background: bool = False
) -> ClassificationMetrics:
    """
    计算分类评估指标
    
    包括总体精度(OA)、F1分数、平均交并比(mIoU)、精确率和召回率。
    
    参数:
        predictions: 模型预测结果, 形状为 [H, W] 或 [B, H, W]
        labels: 真实标签, 形状与 predictions 相同
        num_classes: 类别数量
        ignore_background: 是否忽略背景类(索引为0)的计算
    
    返回:
        ClassificationMetrics: 包含各项评估指标的数据对象
    
    示例:
        >>> preds = torch.argmax(model_output, dim=1)
        >>> metrics = calculate_classification_metrics(preds, labels, num_classes=2)
        >>> print(f"OA: {metrics.overall_accuracy:.4f}, mIoU: {metrics.mean_iou:.4f}")
    """
    # 展平预测和标签
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    # 计算总体精度 (Overall Accuracy)
    overall_accuracy = (predictions_flat == labels_flat).float().mean().item()
    
    # 计算 mIoU
    mean_iou, class_ious = _calculate_mean_iou(
        predictions_flat, labels_flat, num_classes, ignore_background
    )
    
    # 设置多分类指标计算参数
    ignore_index = 0 if ignore_background else None
    average_method = "micro" if ignore_background else "macro"
    
    # 计算精确率
    precision = multiclass_precision(
        predictions_flat, labels_flat, num_classes, 
        average=average_method, ignore_index=ignore_index
    ).item()
    
    # 计算召回率
    recall = multiclass_recall(
        predictions_flat, labels_flat, num_classes, 
        average=average_method, ignore_index=ignore_index
    ).item()
    
    # 计算 F1 分数
    f1_score = multiclass_f1_score(
        predictions_flat, labels_flat, num_classes, 
        average=average_method, ignore_index=ignore_index
    ).item()
    
    return ClassificationMetrics(
        overall_accuracy=overall_accuracy,
        f1_score=f1_score,
        mean_iou=mean_iou,
        precision=precision,
        recall=recall,
        class_ious=class_ious
    )


def _calculate_mean_iou(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_background: bool = False
) -> Tuple[float, List[float]]:
    """
    计算平均交并比 (Mean Intersection over Union)
    
    通过构建混淆矩阵计算每个类别的 IoU，然后取平均。
    
    参数:
        predictions: 展平后的预测结果
        labels: 展平后的真实标签
        num_classes: 类别数量
        ignore_background: 是否在计算均值时忽略背景类
    
    返回:
        Tuple[float, List[float]]: (mIoU, 各类别IoU列表)
    """
    # 构建混淆矩阵
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), 
        device=predictions.device
    )
    
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = torch.sum(
                (labels == i) & (predictions == j)
            )
    
    # 计算每个类别的 IoU
    class_ious = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = (
            torch.sum(confusion_matrix[i, :]) + 
            torch.sum(confusion_matrix[:, i]) - 
            intersection
        )
        
        if union == 0:
            iou = 0.0  # 该类别在真实标签中不存在
        else:
            iou = (intersection / union).item()
        class_ious.append(iou)
    
    # 计算 mIoU
    if ignore_background:
        # 只计算前景类的平均
        mean_iou = np.mean(class_ious[1:])
    else:
        mean_iou = np.mean(class_ious)
    
    return mean_iou, class_ious


# ============================================================================
# 特征提取钩子函数
# ============================================================================

def _feature_extraction_hook(
    module: torch.nn.Module,
    input: torch.Tensor,
    output: torch.Tensor
) -> None:
    """
    前向传播钩子函数，用于捕获中间层的特征图
    
    该函数注册到模型的指定层，在前向传播时自动捕获输出特征。
    捕获的特征存储在全局变量 _feature_hook_output 中。
    
    参数:
        module: 被钩住的模块
        input: 模块的输入张量
        output: 模块的输出张量（即需要捕获的特征图）
    
    注意:
        这是内部函数，不应直接调用。
        使用 register_feature_hook() 来注册钩子。
    """
    global _feature_hook_output
    _feature_hook_output['feature'] = output


def register_feature_hook(
    model: torch.nn.Module,
    layer_name: str
) -> torch.utils.hooks.RemovableHandle:
    """
    在模型指定层注册特征提取钩子
    
    参数:
        model: PyTorch 模型
        layer_name: 要钩住的层名称（如 'up4', 'down4', 'outc'）
    
    返回:
        RemovableHandle: 钩子句柄，用于后续移除钩子
    
    示例:
        >>> handle = register_feature_hook(model, 'up4')
        >>> # ... 进行前向传播 ...
        >>> features = get_hooked_features()
        >>> handle.remove()  # 使用完毕后移除钩子
    """
    global _feature_hook_output
    _feature_hook_output = {}  # 重置特征存储
    
    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(_feature_extraction_hook)
    
    return handle


def get_hooked_features() -> torch.Tensor:
    """
    获取钩子捕获的特征图
    
    返回:
        torch.Tensor: 捕获的特征图
    
    异常:
        KeyError: 如果钩子尚未捕获任何特征
    """
    return _feature_hook_output['feature']


# ============================================================================
# 特征提取器类
# ============================================================================

class FeatureExtractor:
    """
    特征提取器基类
    
    提供从数据加载器中批量提取特征和计算评估指标的功能。
    支持按类别提取特征，可选择使用真实标签或预测标签。
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_classes: int = DEFAULT_NUM_CLASSES
    ):
        """
        初始化特征提取器
        
        参数:
            model: 预训练的语义分割模型
            device: 计算设备（CPU或GPU）
            num_classes: 分割任务的类别数量
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
    
    def extract_features_by_label(
        self,
        data_loader: torch.utils.data.DataLoader,
        target_label_index: Optional[int] = None,
        use_prediction_labels: bool = False,
        max_images: int = DEFAULT_MAX_IMAGES,
        exclude_zero_features: bool = False,
        single_batch: bool = False
    ) -> Tuple[ClassificationMetrics, FeatureStatistics, np.ndarray]:
        """
        按标签索引提取特征并计算评估指标
        
        参数:
            data_loader: 数据加载器
            target_label_index: 目标标签索引（None表示提取所有特征）
            use_prediction_labels: 是否使用预测标签而非真实标签
            max_images: 最大处理图像数量
            exclude_zero_features: 是否排除零值特征
            single_batch: 是否只处理单个批次
        
        返回:
            Tuple包含:
            - ClassificationMetrics: 分类评估指标的平均值
            - FeatureStatistics: 特征统计数据
            - np.ndarray: 原始特征数组
        """
        metrics_list = []
        extracted_features = []
        num_processed_images = 0
        
        data_iterator = iter(data_loader)
        
        def process_single_batch():
            nonlocal num_processed_images
            
            images, true_masks = next(data_iterator)
            num_processed_images += images.shape[0]
            
            # 将数据移至目标设备
            images = images.to(self.device, dtype=torch.float32)
            true_masks_device = true_masks.to(self.device, dtype=torch.long)
            
            # 模型推理
            with torch.no_grad():
                model_output = self.model(images)
            
            predictions = torch.argmax(model_output, dim=1)
            
            # 获取钩子捕获的特征
            features = get_hooked_features().cpu()
            
            # 按标签提取特征
            batch_features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
            
            if target_label_index is not None:
                # 使用指定的标签提取对应特征
                if use_prediction_labels:
                    label_mask = predictions.cpu()
                else:
                    label_mask = true_masks
                
                label_mask_expanded = label_mask.unsqueeze(1).expand_as(features).permute(0, 2, 3, 1)
                selected_features = batch_features[label_mask_expanded == target_label_index]
                selected_features = selected_features.reshape(-1, features.shape[1])
            else:
                # 提取所有特征
                selected_features = batch_features.reshape(-1, features.shape[1])
            
            extracted_features.extend(selected_features.tolist())
            
            # 计算当前批次的评估指标
            ignore_background = target_label_index is not None
            metrics = calculate_classification_metrics(
                predictions, true_masks_device, 
                self.num_classes, ignore_background
            )
            metrics_list.append([
                metrics.overall_accuracy, metrics.f1_score, 
                metrics.mean_iou, metrics.precision, metrics.recall
            ])
        
        # 执行批次处理
        if single_batch:
            process_single_batch()
        else:
            for _ in tqdm(range(len(data_loader)), desc="提取特征中"):
                process_single_batch()
                if num_processed_images >= max_images:
                    break
        
        # 转换特征为numpy数组
        features_array = np.array(extracted_features, dtype=np.float32)
        
        # 计算特征统计量
        if exclude_zero_features:
            # 排除零值特征后计算统计量
            mean, variance = self._calculate_nonzero_statistics(features_array)
        else:
            mean = np.mean(features_array, axis=0)
            variance = np.var(features_array, axis=0)
        
        covariance = np.cov(features_array, rowvar=False) if features_array.shape[0] > 1 else np.zeros((features_array.shape[1],))
        
        # 计算平均评估指标
        metrics_array = np.array(metrics_list)
        avg_metrics = ClassificationMetrics(
            overall_accuracy=np.mean(metrics_array[:, 0]),
            f1_score=np.mean(metrics_array[:, 1]),
            mean_iou=np.mean(metrics_array[:, 2]),
            precision=np.mean(metrics_array[:, 3]),
            recall=np.mean(metrics_array[:, 4])
        )
        
        feature_stats = FeatureStatistics(
            mean=mean,
            covariance=covariance,
            variance=variance,
            features=features_array,
            num_samples=features_array.shape[0]
        )
        
        return avg_metrics, feature_stats, features_array
    
    def _calculate_nonzero_statistics(
        self, 
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算非零特征的统计量
        
        参数:
            features: 特征数组 [N, C]
        
        返回:
            Tuple[np.ndarray, np.ndarray]: (均值, 方差)
        """
        num_features = features.shape[1]
        mean = np.zeros(num_features)
        variance = np.zeros(num_features)
        
        for i in range(num_features):
            nonzero_mask = features[:, i] != 0.0
            if np.any(nonzero_mask):
                mean[i] = np.mean(features[nonzero_mask, i])
                variance[i] = np.var(features[nonzero_mask, i])
        
        return mean, variance


class DispersionFeatureExtractor(FeatureExtractor):
    """
    分散度特征提取器
    
    用于计算 DS (Dispersion Score) 度量，分别提取前景和背景特征。
    """
    
    def extract_features_by_class(
        self,
        data_loader: torch.utils.data.DataLoader,
        use_prediction_labels: bool = False,
        max_images: int = DEFAULT_MAX_IMAGES,
        single_batch: bool = False
    ) -> Tuple[ClassificationMetrics, Dict[int, FeatureStatistics], Dict[int, List]]:
        """
        按类别分别提取特征
        
        参数:
            data_loader: 数据加载器
            use_prediction_labels: 是否使用预测标签
            max_images: 最大处理图像数量
            single_batch: 是否只处理单个批次
        
        返回:
            Tuple包含:
            - ClassificationMetrics: 平均评估指标
            - Dict[int, FeatureStatistics]: 各类别特征统计
            - Dict[int, List]: 各类别原始特征列表
        """
        metrics_list = []
        class_features = {i: [] for i in range(self.num_classes)}
        num_processed_images = 0
        
        data_iterator = iter(data_loader)
        
        def process_single_batch():
            nonlocal num_processed_images
            
            images, true_masks = next(data_iterator)
            num_processed_images += images.shape[0]
            
            images = images.to(self.device, dtype=torch.float32)
            true_masks_device = true_masks.to(self.device, dtype=torch.long)
            
            with torch.no_grad():
                model_output = self.model(images)
            
            predictions = torch.argmax(model_output, dim=1)
            features = get_hooked_features().cpu()
            
            batch_features = features.permute(0, 2, 3, 1)
            
            # 确定使用哪个标签
            if use_prediction_labels:
                labels = predictions.cpu()
            else:
                labels = true_masks
            
            labels_expanded = labels.unsqueeze(1).expand_as(features).permute(0, 2, 3, 1)
            
            # 按类别提取特征
            for class_idx in range(self.num_classes):
                class_mask = labels_expanded == class_idx
                class_feat = batch_features[class_mask].reshape(-1, features.shape[1])
                class_features[class_idx].extend(class_feat.tolist())
            
            metrics = calculate_classification_metrics(
                predictions, true_masks_device, self.num_classes
            )
            metrics_list.append([
                metrics.overall_accuracy, metrics.f1_score,
                metrics.mean_iou, metrics.precision, metrics.recall
            ])
        
        if single_batch:
            process_single_batch()
        else:
            for _ in tqdm(range(len(data_loader)), desc="提取类别特征中"):
                process_single_batch()
                if num_processed_images >= max_images:
                    break
        
        # 计算各类别特征统计
        class_stats = {}
        for class_idx in range(self.num_classes):
            features_array = np.array(class_features[class_idx], dtype=np.float32)
            if features_array.shape[0] > 0:
                class_stats[class_idx] = FeatureStatistics(
                    mean=np.mean(features_array, axis=0),
                    covariance=np.cov(features_array, rowvar=False) if features_array.shape[0] > 1 else np.zeros((features_array.shape[1],)),
                    variance=np.var(features_array, axis=0),
                    features=features_array,
                    num_samples=features_array.shape[0]
                )
        
        # 计算平均指标
        metrics_array = np.array(metrics_list)
        avg_metrics = ClassificationMetrics(
            overall_accuracy=np.mean(metrics_array[:, 0]),
            f1_score=np.mean(metrics_array[:, 1]),
            mean_iou=np.mean(metrics_array[:, 2]),
            precision=np.mean(metrics_array[:, 3]),
            recall=np.mean(metrics_array[:, 4])
        )
        
        return avg_metrics, class_stats, class_features


class GBCFeatureExtractor(FeatureExtractor):
    """
    GBC 特征提取器
    
    用于计算 GBC (Geometric Bayesian Classifier) 度量。
    提取所有特征及其对应的标签。
    """
    
    def extract_features_with_labels(
        self,
        data_loader: torch.utils.data.DataLoader,
        use_prediction_labels: bool = False,
        max_images: int = DEFAULT_MAX_IMAGES,
        single_batch: bool = False
    ) -> Tuple[ClassificationMetrics, List, List]:
        """
        提取特征及其对应的标签
        
        参数:
            data_loader: 数据加载器
            use_prediction_labels: 是否使用预测标签
            max_images: 最大处理图像数量
            single_batch: 是否只处理单个批次
        
        返回:
            Tuple包含:
            - ClassificationMetrics: 平均评估指标
            - List: 特征列表
            - List: 标签列表
        """
        metrics_list = []
        all_features = []
        all_labels = []
        num_processed_images = 0
        
        data_iterator = iter(data_loader)
        
        def process_single_batch():
            nonlocal num_processed_images
            
            images, true_masks = next(data_iterator)
            num_processed_images += images.shape[0]
            
            images = images.to(self.device, dtype=torch.float32)
            true_masks_device = true_masks.to(self.device, dtype=torch.long)
            
            with torch.no_grad():
                model_output = self.model(images)
            
            predictions = torch.argmax(model_output, dim=1)
            features = get_hooked_features().cpu()
            
            batch_features = features.permute(0, 2, 3, 1)
            all_features.extend(batch_features.reshape(-1, features.shape[1]).tolist())
            
            if use_prediction_labels:
                labels = predictions.cpu()
            else:
                labels = true_masks
            
            all_labels.extend(labels.reshape(-1).tolist())
            
            metrics = calculate_classification_metrics(
                predictions, true_masks_device, self.num_classes
            )
            metrics_list.append([
                metrics.overall_accuracy, metrics.f1_score,
                metrics.mean_iou, metrics.precision, metrics.recall
            ])
        
        if single_batch:
            process_single_batch()
        else:
            for _ in tqdm(range(len(data_loader)), desc="提取特征与标签中"):
                process_single_batch()
                if num_processed_images >= max_images:
                    break
        
        metrics_array = np.array(metrics_list)
        avg_metrics = ClassificationMetrics(
            overall_accuracy=np.mean(metrics_array[:, 0]),
            f1_score=np.mean(metrics_array[:, 1]),
            mean_iou=np.mean(metrics_array[:, 2]),
            precision=np.mean(metrics_array[:, 3]),
            recall=np.mean(metrics_array[:, 4])
        )
        
        return avg_metrics, all_features, all_labels


# ============================================================================
# 距离度量计算函数
# ============================================================================

def calculate_frechet_distance(
    source_mean: np.ndarray,
    target_mean: np.ndarray,
    source_cov: np.ndarray,
    target_cov: np.ndarray
) -> float:
    """
    计算 Fréchet Distance (FD)
    
    FD 用于衡量两个高斯分布之间的距离，常用于评估生成模型质量。
    公式: FD = ||mu_s - mu_t||² + Tr(Σ_s + Σ_t - 2*sqrt(Σ_s * Σ_t))
    
    参数:
        source_mean: 源域特征均值
        target_mean: 目标域特征均值
        source_cov: 源域特征协方差矩阵
        target_cov: 目标域特征协方差矩阵
    
    返回:
        float: Fréchet Distance 值
    
    示例:
        >>> fd = calculate_frechet_distance(mu1, mu2, sigma1, sigma2)
        >>> print(f"Fréchet Distance: {fd:.4f}")
    """
    # 计算均值差的L2范数平方
    mean_diff_squared = np.sum((source_mean - target_mean) ** 2)
    
    # 计算协方差矩阵乘积的平方根
    cov_product_sqrt = sqrtm(source_cov @ target_cov)
    
    # 处理复数结果
    if np.iscomplexobj(cov_product_sqrt):
        cov_product_sqrt = cov_product_sqrt.real
    
    # 计算迹项
    trace_term = np.trace(source_cov + target_cov - 2 * cov_product_sqrt)
    
    # 确保非负
    trace_term = abs(trace_term)
    
    return float(mean_diff_squared + trace_term)


def calculate_fid_from_features(
    source_features: np.ndarray,
    target_features: np.ndarray
) -> float:
    """
    从原始特征计算 Fréchet Inception Distance (FID)
    
    参数:
        source_features: 源域特征数组 [N, D]
        target_features: 目标域特征数组 [M, D]
    
    返回:
        float: FID 值
    """
    source_mean = np.mean(source_features, axis=0)
    source_cov = np.cov(source_features, rowvar=False)
    
    target_mean = np.mean(target_features, axis=0)
    target_cov = np.cov(target_features, rowvar=False)
    
    return calculate_frechet_distance(
        source_mean, target_mean, source_cov, target_cov
    )


def calculate_dispersion_score(
    overall_mean: np.ndarray,
    class_means: Dict[int, np.ndarray],
    class_samples: Dict[int, int],
    feature_weights: Optional[Dict[int, np.ndarray]] = None
) -> Tuple[float, float]:
    """
    计算分散度分数 (Dispersion Score)
    
    衡量各类别特征均值与全局均值之间的加权平均距离。
    
    参数:
        overall_mean: 所有特征的全局均值
        class_means: 各类别特征均值的字典
        class_samples: 各类别样本数量的字典
        feature_weights: 特征权重字典（可选）
    
    返回:
        Tuple[float, float]: (原始分数, 对数分数)
    """
    num_classes = len(class_means)
    weighted_sum = 0.0
    
    for class_idx in range(num_classes):
        mean_diff = overall_mean - class_means[class_idx]
        
        if feature_weights is not None and class_idx in feature_weights:
            mean_diff = mean_diff * feature_weights[class_idx]
        
        weighted_sum += class_samples[class_idx] * np.linalg.norm(mean_diff, ord=2)
    
    score = weighted_sum / (num_classes - 1)
    log_score = np.log(score) if score > 0 else float('-inf')
    
    return score, log_score


# ============================================================================
# 模型最后一层权重分析
# ============================================================================

def extract_last_layer_weight_difference(
    model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """
    提取模型最后一层的权重差异
    
    对于二分类任务，计算两个类别权重向量之间的差异。
    这个差异可以用于加权特征距离计算。
    
    参数:
        model: 语义分割模型（假设最后一层为 'outc'）
    
    返回:
        Dict[str, torch.Tensor]: 包含不同形式权重差异的字典
            - 'raw_difference': 原始差异 (w0 - w1)
            - 'absolute_difference': 绝对差异 |w0 - w1|
            - 'normalized_difference': 归一化差异
            - 'normalized_absolute': 归一化绝对差异
    """
    # 获取最后一层权重和偏置
    last_layer_weight = model.outc.weight.detach().cpu()  # [num_classes, C, 1, 1]
    last_layer_bias = model.outc.bias.detach().cpu()      # [num_classes]
    
    # 调整权重形状
    last_layer_weight = last_layer_weight.squeeze()  # [num_classes, C]
    
    print(f"最后一层权重形状: {last_layer_weight.shape}")
    print(f"最后一层偏置: {last_layer_bias}")
    
    # 计算类别间的权重差异
    # 假设二分类: y0 - y1 = (w0 - w1) * x + (b0 - b1)
    temp_scale = 10.0  # 缩放因子
    
    raw_diff = (last_layer_weight[0] - last_layer_weight[1]) * temp_scale + \
               (last_layer_bias[0] - last_layer_bias[1])
    
    abs_diff = torch.abs(raw_diff)
    normalized_diff = raw_diff / torch.max(abs_diff)
    normalized_abs = abs_diff / torch.max(abs_diff)
    
    return {
        'raw_difference': raw_diff,
        'absolute_difference': abs_diff,
        'normalized_difference': normalized_diff,
        'normalized_absolute': normalized_abs
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_scatter_correlation(
    results: List[List],
    x_column: int,
    y_columns: List[int],
    x_label: str,
    y_labels: List[str],
    output_path: str,
    figure_name: str
) -> None:
    """
    绘制散点图展示度量指标之间的相关性
    
    参数:
        results: 结果数据列表
        x_column: x轴数据列索引
        y_columns: y轴数据列索引列表
        x_label: x轴标签
        y_labels: y轴标签列表
        output_path: 输出目录
        figure_name: 图片文件名
    """
    plt.figure(figsize=(10, 6))
    
    x_values = [row[x_column] for row in results]
    
    # 计算点大小
    point_size = max(2, min(1000.0 / len(x_values), 20))
    
    for i, y_col in enumerate(y_columns):
        y_values = [row[y_col] for row in results]
        plt.scatter(x_values, y_values, label=y_labels[i], s=point_size, alpha=0.7)
    
    plt.xlabel(x_label)
    plt.ylabel("Metric Value")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    test_path_exist(output_path)
    plt.savefig(os.path.join(output_path, figure_name), bbox_inches='tight', dpi=150)
    plt.close()


def plot_results_by_class(
    results: List[List],
    x_column: int,
    y_column: int,
    x_label: str,
    y_label: str,
    output_path: str,
    figure_name: str
) -> None:
    """
    按类别绘制结果散点图
    
    每个类别使用不同颜色，并添加图例。
    """
    plt.figure(figsize=(12, 8))
    
    point_size = max(2, min(1000.0 / len(results), 20))
    
    for row_idx, row in enumerate(results):
        label_text = f"{row[0]}-{row[1]}_cls-{row[2]}-{row[3]}"
        plt.scatter(row[x_column], row[y_column], label=label_text, s=point_size)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()
    
    test_path_exist(output_path)
    plt.savefig(os.path.join(output_path, figure_name), bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# 迁移度量计算主函数
# ============================================================================

def compute_fd_transfer_metric(
    model: torch.nn.Module,
    model_device: torch.device,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    target_label_index: Optional[int],
    use_prediction_labels: bool,
    exclude_zero_features: bool,
    process_all_target: bool,
    max_images: int
) -> Tuple[List[str], List[List]]:
    """
    计算 FD (Fréchet Distance) 迁移度量
    
    参数:
        model: 预训练模型
        model_device: 计算设备
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        target_label_index: 目标标签索引
        use_prediction_labels: 是否使用预测标签
        exclude_zero_features: 是否排除零值特征
        process_all_target: 是否处理所有目标域数据
        max_images: 最大图像数量
    
    返回:
        Tuple[List[str], List[List]]: (列名列表, 结果数据列表)
    """
    extractor = FeatureExtractor(model, model_device)
    
    # 提取源域特征
    source_metrics, source_stats, source_features = extractor.extract_features_by_label(
        iter(source_loader),
        target_label_index=target_label_index,
        use_prediction_labels=use_prediction_labels,
        exclude_zero_features=exclude_zero_features,
        max_images=max_images,
        single_batch=False
    )
    
    # 获取权重差异
    weight_diff_dict = extract_last_layer_weight_difference(model)
    
    results = []
    target_iterator = iter(target_loader)
    
    # 处理目标域
    if process_all_target:
        target_metrics, target_stats, target_features = extractor.extract_features_by_label(
            target_iterator,
            target_label_index=target_label_index,
            use_prediction_labels=use_prediction_labels,
            exclude_zero_features=exclude_zero_features,
            max_images=max_images,
            single_batch=False
        )
    
    for batch_idx in tqdm(range(len(target_loader)), desc="计算FD度量"):
        if not process_all_target:
            target_metrics, target_stats, target_features = extractor.extract_features_by_label(
                iter(target_loader),
                target_label_index=target_label_index,
                use_prediction_labels=use_prediction_labels,
                exclude_zero_features=exclude_zero_features,
                max_images=max_images,
                single_batch=True
            )
        
        # 计算均值差异
        mean_diff = target_stats.mean - source_stats.mean
        mean_diff_absolute = np.abs(mean_diff)
        mean_diff_relative = mean_diff / (source_stats.mean + 1e-8)
        mean_diff_relative_abs = np.abs(mean_diff_relative)
        
        # 计算 FD
        fd_score = calculate_frechet_distance(
            source_stats.mean, target_stats.mean,
            source_stats.covariance, target_stats.covariance
        )
        
        # 计算加权 FD
        fd_weighted_scores = {}
        for weight_key, weight_values in weight_diff_dict.items():
            weight_array = weight_values.numpy()
            fd_weighted = calculate_frechet_distance(
                source_stats.mean * weight_array,
                target_stats.mean * weight_array,
                source_stats.covariance,
                target_stats.covariance
            )
            fd_weighted_scores[f"FD_{weight_key}"] = fd_weighted
        
        # 计算相对变化
        oa_delta_relative = (source_metrics.overall_accuracy - target_metrics.overall_accuracy) / source_metrics.overall_accuracy
        f1_delta_relative = (source_metrics.f1_score - target_metrics.f1_score) / source_metrics.f1_score
        
        results.append([
            source_metrics.overall_accuracy - target_metrics.overall_accuracy,
            source_metrics.f1_score - target_metrics.f1_score,
            source_metrics.precision - target_metrics.precision,
            oa_delta_relative, f1_delta_relative,
            source_metrics.overall_accuracy, source_metrics.f1_score, source_metrics.precision,
            target_metrics.overall_accuracy, target_metrics.f1_score, target_metrics.precision,
            np.sum(mean_diff), np.sum(mean_diff_absolute),
            np.sum(mean_diff_relative), np.sum(mean_diff_relative_abs),
            fd_score
        ] + list(fd_weighted_scores.values()))
        
        if process_all_target:
            break
    
    column_names = [
        "OA_delta", "F1_delta", "precision_delta",
        "OA_delta_relative", "F1_delta_relative",
        "OA_source", "F1_source", "precision_source",
        "OA_target", "F1_target", "precision_target",
        "mean_diff_sum", "mean_diff_abs_sum",
        "mean_diff_relative_sum", "mean_diff_relative_abs_sum",
        "FD_score"
    ] + list(fd_weighted_scores.keys())
    
    return column_names, results


def compute_ds_transfer_metric(
    model: torch.nn.Module,
    model_device: torch.device,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    use_prediction_labels: bool,
    process_all_target: bool,
    max_images: int
) -> Tuple[List[str], List[List]]:
    """
    计算 DS (Dispersion Score) 迁移度量
    
    参数:
        model: 预训练模型
        model_device: 计算设备
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        use_prediction_labels: 是否使用预测标签
        process_all_target: 是否处理所有目标域数据
        max_images: 最大图像数量
    
    返回:
        Tuple[List[str], List[List]]: (列名列表, 结果数据列表)
    """
    extractor = DispersionFeatureExtractor(model, model_device)
    
    # 提取源域特征
    source_metrics, source_class_stats, _ = extractor.extract_features_by_class(
        iter(source_loader),
        use_prediction_labels=False,
        max_images=max_images,
        single_batch=False
    )
    
    # 获取权重差异用于加权
    weight_diff_dict = extract_last_layer_weight_difference(model)
    weight_abs_normalized = weight_diff_dict['normalized_absolute']
    weight_dict_normalized = {i: weight_abs_normalized.numpy() for i in range(DEFAULT_NUM_CLASSES)}
    
    results = []
    
    for batch_idx in tqdm(range(len(target_loader)), desc="计算DS度量"):
        target_metrics, target_class_stats, _ = extractor.extract_features_by_class(
            iter(target_loader),
            use_prediction_labels=use_prediction_labels,
            max_images=max_images,
            single_batch=not process_all_target
        )
        
        # 计算目标域全局均值
        total_samples = sum(stats.num_samples for stats in target_class_stats.values())
        overall_mean = sum(
            target_class_stats[i].mean * target_class_stats[i].num_samples 
            for i in target_class_stats.keys()
        ) / total_samples
        
        # 计算样本数量
        class_samples = {i: target_class_stats[i].num_samples for i in target_class_stats.keys()}
        
        # 计算分散度分数
        dispersion_score, log_dispersion = calculate_dispersion_score(
            overall_mean,
            {i: target_class_stats[i].mean for i in target_class_stats.keys()},
            class_samples
        )
        
        weighted_dispersion, weighted_log_dispersion = calculate_dispersion_score(
            overall_mean,
            {i: target_class_stats[i].mean for i in target_class_stats.keys()},
            class_samples,
            weight_dict_normalized
        )
        
        results.append([
            source_metrics.overall_accuracy, source_metrics.f1_score, 
            source_metrics.mean_iou, source_metrics.precision, source_metrics.recall,
            target_metrics.overall_accuracy, target_metrics.f1_score,
            target_metrics.mean_iou, target_metrics.precision, target_metrics.recall,
            source_metrics.overall_accuracy - target_metrics.overall_accuracy,
            source_metrics.f1_score - target_metrics.f1_score,
            source_metrics.mean_iou - target_metrics.mean_iou,
            source_metrics.precision - target_metrics.precision,
            source_metrics.recall - target_metrics.recall,
            dispersion_score, log_dispersion,
            weighted_dispersion, weighted_log_dispersion
        ])
        
        if process_all_target:
            break
    
    column_names = [
        "OA_source", "F1_source", "mIoU_source", "precision_source", "recall_source",
        "OA_target", "F1_target", "mIoU_target", "precision_target", "recall_target",
        "OA_delta", "F1_delta", "mIoU_delta", "precision_delta", "recall_delta",
        "dispersion_score", "log_dispersion_score",
        "weighted_dispersion_score", "weighted_log_dispersion_score"
    ]
    
    return column_names, results


def compute_gbc_transfer_metric(
    model: torch.nn.Module,
    model_device: torch.device,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    use_prediction_labels: bool,
    process_all_target: bool,
    max_images: int
) -> Tuple[List[str], List[List]]:
    """
    计算 GBC (Geometric Bayesian Classifier) 迁移度量
    
    参数:
        model: 预训练模型
        model_device: 计算设备
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        use_prediction_labels: 是否使用预测标签
        process_all_target: 是否处理所有目标域数据
        max_images: 最大图像数量
    
    返回:
        Tuple[List[str], List[List]]: (列名列表, 结果数据列表)
    """
    extractor = GBCFeatureExtractor(model, model_device)
    
    # 提取源域特征
    source_metrics, _, _ = extractor.extract_features_with_labels(
        iter(source_loader),
        use_prediction_labels=False,
        max_images=max_images,
        single_batch=False
    )
    
    results = []
    
    for batch_idx in tqdm(range(len(target_loader)), desc="计算GBC度量"):
        target_metrics, target_features, target_labels = extractor.extract_features_with_labels(
            iter(target_loader),
            use_prediction_labels=use_prediction_labels,
            max_images=max_images,
            single_batch=not process_all_target
        )
        
        # 计算 GBC 分数
        diagonal_gbc = get_gbc_score(target_features, target_labels, 'diagonal')
        spherical_gbc = get_gbc_score(target_features, target_labels, 'spherical')
        
        results.append([
            source_metrics.overall_accuracy, source_metrics.f1_score,
            source_metrics.mean_iou, source_metrics.precision, source_metrics.recall,
            target_metrics.overall_accuracy, target_metrics.f1_score,
            target_metrics.mean_iou, target_metrics.precision, target_metrics.recall,
            source_metrics.overall_accuracy - target_metrics.overall_accuracy,
            source_metrics.f1_score - target_metrics.f1_score,
            source_metrics.mean_iou - target_metrics.mean_iou,
            source_metrics.precision - target_metrics.precision,
            source_metrics.recall - target_metrics.recall,
            diagonal_gbc, spherical_gbc
        ])
        
        if process_all_target:
            break
    
    column_names = [
        "OA_source", "F1_source", "mIoU_source", "precision_source", "recall_source",
        "OA_target", "F1_target", "mIoU_target", "precision_target", "recall_target",
        "OA_delta", "F1_delta", "mIoU_delta", "precision_delta", "recall_delta",
        "diagonal_GBC", "spherical_GBC"
    ]
    
    return column_names, results


# ============================================================================
# 主程序入口
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='迁移学习度量指标计算工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型参数
    parser.add_argument(
        '--model_path_prefix',
        type=str,
        default='/home/Shanxin.Guo/ZhangtuosCode/2_model_pth',
        help='模型文件路径前缀'
    )
    
    # 数据路径
    parser.add_argument(
        '--data_path_source',
        type=str,
        default='/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset/dwq_sentinel2/train_val',
        help='源域数据集路径'
    )
    parser.add_argument(
        '--data_path_target',
        type=str,
        default='/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset/xj_sentinel2/train_val',
        help='目标域数据集路径'
    )
    
    # 输出路径
    parser.add_argument(
        '--result_path',
        type=str,
        default='/home/Shanxin.Guo/ZhangtuosCode/code/Transfer_Metric_FCDTM/result',
        help='结果输出路径'
    )
    parser.add_argument(
        '--log_name',
        type=str,
        default='transfer_metric.log',
        help='日志文件名'
    )
    
    # 计算参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='批次大小'
    )
    parser.add_argument(
        '--label_1_percent',
        type=float,
        default=0.2,
        help='前景像素比例阈值'
    )
    parser.add_argument(
        '--target_domain_all',
        type=int,
        default=1,
        help='是否处理所有目标域数据 (1: 是, 0: 按批次处理)'
    )
    parser.add_argument(
        '--no_feature0',
        type=int,
        default=0,
        help='是否排除零值特征 (1: 是, 0: 否)'
    )
    parser.add_argument(
        '--feature_layer_name',
        type=str,
        default=DEFAULT_FEATURE_LAYER,
        choices=['up4', 'outc', 'down4'],
        help='特征提取层名称'
    )
    parser.add_argument(
        '--only_label_1',
        type=int,
        default=0,
        help='是否只提取前景类特征 (1: 是, 0: 提取所有)'
    )
    parser.add_argument(
        '--by_pred',
        type=int,
        default=0,
        help='是否使用预测标签而非真实标签 (1: 是, 0: 否)'
    )
    parser.add_argument(
        '--dataset_is_train',
        type=int,
        default=1,
        help='使用训练集还是验证集 (1: 训练集, 0: 验证集)'
    )
    
    # 任务参数
    parser.add_argument(
        '--task_transfer',
        type=str,
        default='dwq_s2_xj_s2',
        choices=['dwq_s2_xj_s2', 'dwq_s2_dwq_l8', 'dwq_l8_xj_l8', 'xj_s2_xj_l8'],
        help='迁移任务名称'
    )
    parser.add_argument(
        '--transfer_metric_name',
        type=str,
        default='FD',
        choices=['FD', 'DS', 'GBC'],
        help='迁移度量类型'
    )
    
    return parser.parse_args()


def get_task_configuration(
    task_name: str
) -> Tuple[List[int], List[str]]:
    """
    根据任务名称获取配置
    
    参数:
        task_name: 任务名称
    
    返回:
        Tuple[List[int], List[str]]: (类别索引列表, 数据集名称列表)
    """
    # 各任务对应的类别索引
    task_class_indices = {
        'dwq_s2_xj_s2': [1, 2, 3, 6, 7, 8],
        'dwq_s2_dwq_l8': [1, 2, 6, 7, 8],
        'dwq_l8_xj_l8': [1, 6, 7, 8],
        'xj_s2_xj_l8': [1, 3, 5, 6, 7, 8]
    }
    
    # 各任务对应的数据集
    task_datasets = {
        'dwq_s2_xj_s2': ['dwq_sentinel2', 'xj_sentinel2'],
        'dwq_s2_dwq_l8': ['dwq_sentinel2', 'dwq_landsat8'],
        'dwq_l8_xj_l8': ['dwq_landsat8', 'xj_landsat8'],
        'xj_s2_xj_l8': ['xj_sentinel2', 'xj_landsat8']
    }
    
    return task_class_indices[task_name], task_datasets[task_name]


def main():
    """主函数"""
    args = parse_arguments()
    
    # 获取任务配置
    class_indices, dataset_names = get_task_configuration(args.task_transfer)
    
    # 构建数据路径
    data_path_source = args.data_path_source.replace(
        'dwq_sentinel2', dataset_names[0]
    )
    data_path_target = args.data_path_target.replace(
        'xj_sentinel2', dataset_names[1]
    )
    
    # 保存日志
    save_log(
        result_path=args.result_path,
        log_name=args.log_name,
        args=args
    )
    
    # 确保输出目录存在
    test_path_exist(args.result_path)
    
    print(f"任务: {args.task_transfer}")
    print(f"度量类型: {args.transfer_metric_name}")
    print(f"源域: {dataset_names[0]}, 目标域: {dataset_names[1]}")
    print(f"类别索引: {class_indices}")
    
    # 加载模型
    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_chdir = os.path.dirname(args.model_path_prefix)
    os.chdir(model_chdir)
    
    # 准备数据加载器
    dataset_reader = get_dataset_reader('rgbn')
    
    # ... 后续处理逻辑 ...


if __name__ == "__main__":
    main()
