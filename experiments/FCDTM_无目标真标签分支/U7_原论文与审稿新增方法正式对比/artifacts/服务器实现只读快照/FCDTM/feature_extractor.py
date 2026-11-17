#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取模块

提供特征提取、分类指标计算等功能。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from torchmetrics.functional.classification import (
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score
)

from model import get_hooked_features


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ClassificationMetrics:
    """分类评估指标"""
    overall_accuracy: float
    f1_score: float
    mean_iou: float
    precision: float
    recall: float
    class_ious: Optional[List[float]] = None


@dataclass
class FeatureStatistics:
    """特征统计数据"""
    mean: np.ndarray
    covariance: np.ndarray
    variance: np.ndarray
    features: np.ndarray
    num_samples: int


# ============================================================================
# 分类指标计算
# ============================================================================

def calculate_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
    ignore_background: bool = False
) -> ClassificationMetrics:
    """
    计算分类评估指标
    
    参数:
        predictions: 预测结果 [H, W] 或 [B, H, W]
        labels: 真实标签
        num_classes: 类别数量
        ignore_background: 是否忽略背景类
    
    返回:
        ClassificationMetrics对象
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # 总体精度
    overall_accuracy = (predictions == labels).float().mean().item()
    
    # mIoU
    mean_iou, class_ious = _calculate_mean_iou(
        predictions, labels, num_classes, ignore_background
    )
    
    # 设置参数
    ignore_index = 0 if ignore_background else None
    average_method = "micro" if ignore_background else "macro"
    
    # 精确率
    precision = multiclass_precision(
        predictions, labels, num_classes,
        average=average_method, ignore_index=ignore_index
    ).item()
    
    # 召回率
    recall = multiclass_recall(
        predictions, labels, num_classes,
        average=average_method, ignore_index=ignore_index
    ).item()
    
    # F1分数
    f1_score = multiclass_f1_score(
        predictions, labels, num_classes,
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
    计算平均交并比
    """
    # 构建混淆矩阵
    confusion_matrix = torch.zeros((num_classes, num_classes), device=predictions.device)
    
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = torch.sum((labels == i) & (predictions == j))
    
    # 计算各类别IoU
    class_ious = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = torch.sum(confusion_matrix[i, :]) + torch.sum(confusion_matrix[:, i]) - intersection
        
        if union == 0:
            iou = 0.0
        else:
            iou = (intersection / union).item()
        class_ious.append(iou)
    
    # 计算mIoU
    if ignore_background:
        mean_iou = np.mean(class_ious[1:])
    else:
        mean_iou = np.mean(class_ious)
    
    return mean_iou, class_ious


# ============================================================================
# 特征提取器基类
# ============================================================================

class BaseFeatureExtractor:
    """
    特征提取器基类
    """
    
    def __init__(self, model, device, num_classes: int = 2):
        """
        初始化
        
        参数:
            model: 预训练模型
            device: 计算设备
            num_classes: 类别数量
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
    
    def _inference(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行推理
        
        返回:
            (predictions, features)
        """
        images = images.to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            output = self.model(images)
        
        predictions = torch.argmax(output, dim=1)
        features = get_hooked_features().cpu()
        
        return predictions, features
    
    def _extract_batch_features(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        target_label_index: Optional[int] = None,
        use_prediction_labels: bool = False
    ) -> torch.Tensor:
        """
        提取批次特征
        
        参数:
            features: 特征图 [B, C, H, W]
            predictions: 预测结果
            labels: 真实标签
            target_label_index: 目标标签索引
            use_prediction_labels: 是否使用预测标签
        
        返回:
            提取的特征 [N, C]
        """
        batch_features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        if target_label_index is not None:
            if use_prediction_labels:
                mask = predictions.cpu()
            else:
                mask = labels
            
            mask_expanded = mask.unsqueeze(1).expand_as(features).permute(0, 2, 3, 1)
            selected = batch_features[mask_expanded == target_label_index]
            return selected.reshape(-1, features.shape[1])
        else:
            return batch_features.reshape(-1, features.shape[1])


# ============================================================================
# FD特征提取器
# ============================================================================

class FDFeatureExtractor(BaseFeatureExtractor):
    """
    FD度量特征提取器
    
    提取特征并计算统计量。
    """
    
    def extract(
        self,
        data_loader,
        target_label_index: Optional[int] = None,
        use_prediction_labels: bool = False,
        max_images: int = 100,
        exclude_zero_features: bool = False,
        single_batch: bool = False
    ) -> Tuple[ClassificationMetrics, FeatureStatistics, np.ndarray]:
        """
        提取特征并计算统计量
        
        参数:
            data_loader: 数据加载器
            target_label_index: 目标标签索引（None=所有特征）
            use_prediction_labels: 是否使用预测标签
            max_images: 最大图像数量
            exclude_zero_features: 是否排除零值特征
            single_batch: 是否只处理单个批次
        
        返回:
            (评估指标, 特征统计, 原始特征数组)
        """
        metrics_list = []
        all_features = []
        num_processed = 0
        
        data_iter = iter(data_loader)
        
        def process_batch():
            nonlocal num_processed
            
            images, labels = next(data_iter)
            num_processed += images.shape[0]
            
            labels_device = labels.to(self.device, dtype=torch.long)
            predictions, features = self._inference(images)
            
            # 提取特征
            selected = self._extract_batch_features(
                features, predictions, labels,
                target_label_index, use_prediction_labels
            )
            all_features.extend(selected.tolist())
            
            # 计算指标
            ignore_bg = target_label_index is not None
            metrics = calculate_metrics(predictions, labels_device, self.num_classes, ignore_bg)
            metrics_list.append([metrics.overall_accuracy, metrics.f1_score, 
                               metrics.mean_iou, metrics.precision, metrics.recall])
        
        # 处理批次
        if single_batch:
            process_batch()
        else:
            for _ in tqdm(range(len(data_loader)), desc="提取特征"):
                process_batch()
                if num_processed >= max_images:
                    break
        
        # 转换为数组
        features_array = np.array(all_features, dtype=np.float32)
        
        # 计算统计量
        if exclude_zero_features:
            mean, variance = self._calc_nonzero_stats(features_array)
        else:
            mean = np.mean(features_array, axis=0)
            variance = np.var(features_array, axis=0)
        
        cov = np.cov(features_array, rowvar=False) if features_array.shape[0] > 1 else np.zeros((features_array.shape[1],))
        
        # 平均指标
        metrics_array = np.array(metrics_list)
        avg_metrics = ClassificationMetrics(
            overall_accuracy=float(np.mean(metrics_array[:, 0])),
            f1_score=float(np.mean(metrics_array[:, 1])),
            mean_iou=float(np.mean(metrics_array[:, 2])),
            precision=float(np.mean(metrics_array[:, 3])),
            recall=float(np.mean(metrics_array[:, 4]))
        )
        
        stats = FeatureStatistics(
            mean=mean,
            covariance=cov,
            variance=variance,
            features=features_array,
            num_samples=features_array.shape[0]
        )
        
        return avg_metrics, stats, features_array
    
    def _calc_nonzero_stats(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算非零特征统计量"""
        n = features.shape[1]
        mean = np.zeros(n)
        variance = np.zeros(n)
        
        for i in range(n):
            mask = features[:, i] != 0.0
            if np.any(mask):
                mean[i] = np.mean(features[mask, i])
                variance[i] = np.var(features[mask, i])
        
        return mean, variance


# ============================================================================
# DS特征提取器
# ============================================================================

class DSFeatureExtractor(BaseFeatureExtractor):
    """
    DS度量特征提取器
    
    按类别分别提取特征。
    """
    
    def extract_by_class(
        self,
        data_loader,
        use_prediction_labels: bool = False,
        max_images: int = 100,
        single_batch: bool = False
    ) -> Tuple[ClassificationMetrics, Dict[int, FeatureStatistics], Dict[int, List]]:
        """
        按类别提取特征
        
        返回:
            (评估指标, 各类别统计, 各类别特征列表)
        """
        metrics_list = []
        class_features = {i: [] for i in range(self.num_classes)}
        num_processed = 0
        
        data_iter = iter(data_loader)
        
        def process_batch():
            nonlocal num_processed
            
            images, labels = next(data_iter)
            num_processed += images.shape[0]
            
            labels_device = labels.to(self.device, dtype=torch.long)
            predictions, features = self._inference(images)
            
            batch_features = features.permute(0, 2, 3, 1)
            
            # 确定标签来源
            if use_prediction_labels:
                mask_labels = predictions.cpu()
            else:
                mask_labels = labels
            
            mask_expanded = mask_labels.unsqueeze(1).expand_as(features).permute(0, 2, 3, 1)
            
            # 按类别提取
            for c in range(self.num_classes):
                class_mask = mask_expanded == c
                class_feat = batch_features[class_mask].reshape(-1, features.shape[1])
                class_features[c].extend(class_feat.tolist())
            
            metrics = calculate_metrics(predictions, labels_device, self.num_classes)
            metrics_list.append([metrics.overall_accuracy, metrics.f1_score,
                               metrics.mean_iou, metrics.precision, metrics.recall])
        
        if single_batch:
            process_batch()
        else:
            for _ in tqdm(range(len(data_loader)), desc="提取特征"):
                process_batch()
                if num_processed >= max_images:
                    break
        
        # 计算统计量
        class_stats = {}
        for c in range(self.num_classes):
            arr = np.array(class_features[c], dtype=np.float32)
            if arr.shape[0] > 0:
                class_stats[c] = FeatureStatistics(
                    mean=np.mean(arr, axis=0),
                    covariance=np.cov(arr, rowvar=False) if arr.shape[0] > 1 else np.zeros((arr.shape[1],)),
                    variance=np.var(arr, axis=0),
                    features=arr,
                    num_samples=arr.shape[0]
                )
        
        # 平均指标
        metrics_array = np.array(metrics_list)
        avg_metrics = ClassificationMetrics(
            overall_accuracy=float(np.mean(metrics_array[:, 0])),
            f1_score=float(np.mean(metrics_array[:, 1])),
            mean_iou=float(np.mean(metrics_array[:, 2])),
            precision=float(np.mean(metrics_array[:, 3])),
            recall=float(np.mean(metrics_array[:, 4]))
        )
        
        return avg_metrics, class_stats, class_features


# ============================================================================
# GBC特征提取器
# ============================================================================

class GBCFeatureExtractor(BaseFeatureExtractor):
    """
    GBC度量特征提取器
    
    提取特征和对应标签。
    """
    
    def extract_with_labels(
        self,
        data_loader,
        use_prediction_labels: bool = False,
        max_images: int = 100,
        single_batch: bool = False
    ) -> Tuple[ClassificationMetrics, List, List]:
        """
        提取特征和标签
        
        返回:
            (评估指标, 特征列表, 标签列表)
        """
        metrics_list = []
        all_features = []
        all_labels = []
        num_processed = 0
        
        data_iter = iter(data_loader)
        
        def process_batch():
            nonlocal num_processed
            
            images, labels = next(data_iter)
            num_processed += images.shape[0]
            
            labels_device = labels.to(self.device, dtype=torch.long)
            predictions, features = self._inference(images)
            
            batch_features = features.permute(0, 2, 3, 1)
            all_features.extend(batch_features.reshape(-1, features.shape[1]).tolist())
            
            if use_prediction_labels:
                batch_labels = predictions.cpu()
            else:
                batch_labels = labels
            
            all_labels.extend(batch_labels.reshape(-1).tolist())
            
            metrics = calculate_metrics(predictions, labels_device, self.num_classes)
            metrics_list.append([metrics.overall_accuracy, metrics.f1_score,
                               metrics.mean_iou, metrics.precision, metrics.recall])
        
        if single_batch:
            process_batch()
        else:
            for _ in tqdm(range(len(data_loader)), desc="提取特征"):
                process_batch()
                if num_processed >= max_images:
                    break
        
        metrics_array = np.array(metrics_list)
        avg_metrics = ClassificationMetrics(
            overall_accuracy=float(np.mean(metrics_array[:, 0])),
            f1_score=float(np.mean(metrics_array[:, 1])),
            mean_iou=float(np.mean(metrics_array[:, 2])),
            precision=float(np.mean(metrics_array[:, 3])),
            recall=float(np.mean(metrics_array[:, 4]))
        )
        
        return avg_metrics, all_features, all_labels


# ============================================================================
# OTCE特征提取器
# ============================================================================

class OTCEFeatureExtractor(BaseFeatureExtractor):
    """
    OTCE度量特征提取器
    
    提取特征和对应标签，用于最优传输计算。
    """
    
    def extract_with_labels(
        self,
        data_loader,
        use_prediction_labels: bool = False,
        max_images: int = 100,
        single_batch: bool = False
    ) -> Tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
        """
        提取特征和标签（返回numpy数组格式）
        
        返回:
            (评估指标, 特征数组, 标签数组)
        """
        metrics_list = []
        all_features = []
        all_labels = []
        num_processed = 0
        
        data_iter = iter(data_loader)
        
        def process_batch():
            nonlocal num_processed
            
            images, labels = next(data_iter)
            num_processed += images.shape[0]
            
            labels_device = labels.to(self.device, dtype=torch.long)
            predictions, features = self._inference(images)
            
            batch_features = features.permute(0, 2, 3, 1)
            all_features.append(batch_features.reshape(-1, features.shape[1]).cpu().numpy())
            
            if use_prediction_labels:
                batch_labels = predictions.cpu()
            else:
                batch_labels = labels
            
            all_labels.append(batch_labels.reshape(-1).cpu().numpy())
            
            metrics = calculate_metrics(predictions, labels_device, self.num_classes)
            metrics_list.append([metrics.overall_accuracy, metrics.f1_score,
                               metrics.mean_iou, metrics.precision, metrics.recall])
        
        if single_batch:
            process_batch()
        else:
            for _ in tqdm(range(len(data_loader)), desc="提取特征"):
                process_batch()
                if num_processed >= max_images:
                    break
        
        metrics_array = np.array(metrics_list)
        avg_metrics = ClassificationMetrics(
            overall_accuracy=float(np.mean(metrics_array[:, 0])),
            f1_score=float(np.mean(metrics_array[:, 1])),
            mean_iou=float(np.mean(metrics_array[:, 2])),
            precision=float(np.mean(metrics_array[:, 3])),
            recall=float(np.mean(metrics_array[:, 4]))
        )
        
        # 合并所有特征和标签
        features_arr = np.concatenate(all_features, axis=0).astype(np.float32)
        labels_arr = np.concatenate(all_labels, axis=0).astype(np.int64)
        
        return avg_metrics, features_arr, labels_arr


# ============================================================================
# LogME特征提取器
# ============================================================================

class LogMEFeatureExtractor(BaseFeatureExtractor):
    """
    LogME度量特征提取器
    
    提取特征和对应标签，用于最大证据计算。
    """
    
    def extract_with_labels(
        self,
        data_loader,
        use_prediction_labels: bool = False,
        max_images: int = 100,
        single_batch: bool = False
    ) -> Tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
        """
        提取特征和标签（返回numpy数组格式）
        
        返回:
            (评估指标, 特征数组, 标签数组)
        """
        metrics_list = []
        all_features = []
        all_labels = []
        num_processed = 0
        
        data_iter = iter(data_loader)
        
        def process_batch():
            nonlocal num_processed
            
            images, labels = next(data_iter)
            num_processed += images.shape[0]
            
            labels_device = labels.to(self.device, dtype=torch.long)
            predictions, features = self._inference(images)
            
            batch_features = features.permute(0, 2, 3, 1)
            all_features.append(batch_features.reshape(-1, features.shape[1]).cpu().numpy())
            
            if use_prediction_labels:
                batch_labels = predictions.cpu()
            else:
                batch_labels = labels
            
            all_labels.append(batch_labels.reshape(-1).cpu().numpy())
            
            metrics = calculate_metrics(predictions, labels_device, self.num_classes)
            metrics_list.append([metrics.overall_accuracy, metrics.f1_score,
                               metrics.mean_iou, metrics.precision, metrics.recall])
        
        if single_batch:
            process_batch()
        else:
            for _ in tqdm(range(len(data_loader)), desc="提取特征"):
                process_batch()
                if num_processed >= max_images:
                    break
        
        metrics_array = np.array(metrics_list)
        avg_metrics = ClassificationMetrics(
            overall_accuracy=float(np.mean(metrics_array[:, 0])),
            f1_score=float(np.mean(metrics_array[:, 1])),
            mean_iou=float(np.mean(metrics_array[:, 2])),
            precision=float(np.mean(metrics_array[:, 3])),
            recall=float(np.mean(metrics_array[:, 4]))
        )
        
        # 合并所有特征和标签
        features_arr = np.concatenate(all_features, axis=0).astype(np.float32)
        labels_arr = np.concatenate(all_labels, axis=0).astype(np.int64)
        
        return avg_metrics, features_arr, labels_arr


# ============================================================================
# 工厂函数
# ============================================================================

def get_feature_extractor(metric_type: str, model, device, num_classes: int = 2):
    """
    获取对应的特征提取器
    
    参数:
        metric_type: 度量类型 ("FD", "DS", "GBC", "OTCE", "LogME")
        model: 模型
        device: 设备
        num_classes: 类别数
    
    返回:
        特征提取器实例
    """
    extractors = {
        "FD": FDFeatureExtractor,
        "DS": DSFeatureExtractor,
        "GBC": GBCFeatureExtractor,
        "OTCE": OTCEFeatureExtractor,
        "LogME": LogMEFeatureExtractor,
    }
    
    if metric_type not in extractors:
        raise ValueError(f"未知的度量类型: {metric_type}")
    
    return extractors[metric_type](model, device, num_classes)
