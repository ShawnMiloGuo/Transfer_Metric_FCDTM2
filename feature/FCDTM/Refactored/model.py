#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型管理模块

提供模型加载、特征提取钩子、权重分析等功能。

使用方法:
    from model import ModelManager
    
    manager = ModelManager(device="cuda")
    model = manager.load_model(model_path)
    manager.register_hook(model, "up4")
    
    # 前向传播后获取特征
    output = model(images)
    features = manager.get_features()
    
    # 获取最后一层权重差异
    weight_diff = manager.get_last_layer_weight_diff(model)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from pathlib import Path
import glob


# ============================================================================
# 全局变量（用于特征提取钩子）
# ============================================================================

_hooked_features: Dict[str, torch.Tensor] = {}


# ============================================================================
# 特征提取钩子
# ============================================================================

def _forward_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
    """
    前向传播钩子函数
    
    捕获指定层的输出特征图。
    """
    global _hooked_features
    _hooked_features['output'] = output


def register_feature_hook(model: nn.Module, layer_name: str) -> torch.utils.hooks.RemovableHandle:
    """
    在模型指定层注册特征提取钩子
    
    参数:
        model: PyTorch模型
        layer_name: 层名称（如 'up4', 'down4', 'outc'）
    
    返回:
        钩子句柄，用于后续移除
    
    示例:
        >>> handle = register_feature_hook(model, 'up4')
        >>> # 前向传播...
        >>> features = get_hooked_features()
        >>> handle.remove()
    """
    global _hooked_features
    _hooked_features = {}
    
    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(_forward_hook)
    
    return handle


def get_hooked_features() -> torch.Tensor:
    """
    获取钩子捕获的特征图
    
    返回:
        捕获的特征张量
    
    异常:
        KeyError: 如果钩子尚未捕获任何特征
    """
    if 'output' not in _hooked_features:
        raise KeyError("尚未捕获任何特征，请先进行前向传播")
    return _hooked_features['output']


def clear_hooked_features() -> None:
    """清除捕获的特征"""
    global _hooked_features
    _hooked_features = {}


# ============================================================================
# 模型管理器
# ============================================================================

class ModelManager:
    """
    模型管理器
    
    提供统一的模型加载、钩子管理和权重分析接口。
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化模型管理器
        
        参数:
            device: 计算设备，None则自动选择
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._current_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    
    def load_model(self, model_path: str) -> nn.Module:
        """
        加载模型
        
        参数:
            model_path: 模型文件路径（支持glob模式）
        
        返回:
            加载的模型（已设置为评估模式）
        
        异常:
            FileNotFoundError: 未找到模型文件
        """
        # 支持glob模式
        if '*' in model_path:
            files = glob.glob(model_path)
            if not files:
                raise FileNotFoundError(f"未找到匹配的模型文件: {model_path}")
            model_path = files[0]
            print(f"加载模型: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        return model
    
    def register_hook(self, model: nn.Module, layer_name: str) -> torch.utils.hooks.RemovableHandle:
        """
        注册特征提取钩子
        
        参数:
            model: 模型
            layer_name: 层名称
        
        返回:
            钩子句柄
        """
        # 先移除旧钩子
        self.remove_hook()
        
        self._current_hook_handle = register_feature_hook(model, layer_name)
        return self._current_hook_handle
    
    def remove_hook(self) -> None:
        """移除当前钩子"""
        if self._current_hook_handle is not None:
            self._current_hook_handle.remove()
            self._current_hook_handle = None
        clear_hooked_features()
    
    def get_features(self) -> torch.Tensor:
        """获取捕获的特征"""
        return get_hooked_features()
    
    @staticmethod
    def get_last_layer_weight_diff(model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        获取最后一层权重差异
        
        对于二分类任务，计算两个类别权重向量之间的差异，
        用于加权特征距离计算。
        
        参数:
            model: 语义分割模型（假设最后一层为 'outc'）
        
        返回:
            包含不同形式权重差异的字典:
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
        
        # 计算类别间权重差异
        # y0 - y1 = (w0 - w1) * x + (b0 - b1)
        scale_factor = 10.0
        
        raw_diff = (last_layer_weight[0] - last_layer_weight[1]) * scale_factor + \
                   (last_layer_bias[0] - last_layer_bias[1])
        
        abs_diff = torch.abs(raw_diff)
        normalized_diff = raw_diff / (torch.max(abs_diff) + 1e-8)
        normalized_abs = abs_diff / (torch.max(abs_diff) + 1e-8)
        
        return {
            'raw_difference': raw_diff,
            'absolute_difference': abs_diff,
            'normalized_difference': normalized_diff,
            'normalized_absolute': normalized_abs
        }
    
    def inference(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """
        执行推理
        
        参数:
            model: 模型
            images: 输入图像
        
        返回:
            预测结果
        """
        images = images.to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(images)
        
        predictions = torch.argmax(output, dim=1)
        return predictions, output


# ============================================================================
# 数据加载工具
# ============================================================================

def create_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    创建数据加载器
    
    参数:
        dataset: 数据集对象
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作线程数
    
    返回:
        DataLoader实例
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == "__main__":
    # 测试
    manager = ModelManager()
    print(f"设备: {manager.device}")
