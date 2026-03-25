#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块

提供结果可视化功能，包括散点图、相关性分析等。
"""

import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
from tqdm import tqdm


class Visualizer:
    """
    可视化工具类
    
    提供多种可视化方法用于展示度量结果。
    """
    
    def __init__(self, output_dir: str, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化
        
        参数:
            output_dir: 输出目录
            figsize: 图像大小
        """
        self.output_dir = output_dir
        self.figsize = figsize
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_scatter(
        self,
        x_values: List[float],
        y_values: List[float],
        x_label: str,
        y_label: str,
        title: str = "",
        label: str = "",
        output_name: str = "scatter.png",
        point_size: Optional[float] = None
    ) -> str:
        """
        绘制散点图
        
        参数:
            x_values: x轴数据
            y_values: y轴数据
            x_label: x轴标签
            y_label: y轴标签
            title: 图标题
            label: 图例标签
            output_name: 输出文件名
            point_size: 点大小
        
        返回:
            输出文件路径
        """
        plt.figure(figsize=self.figsize)
        
        # 自动计算点大小
        if point_size is None:
            point_size = max(2, min(1000.0 / len(x_values), 20))
        
        plt.scatter(x_values, y_values, label=label, s=point_size, alpha=0.7)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        if label:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_name)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return output_path
    
    def plot_multi_scatter(
        self,
        data_series: List[Dict],
        x_label: str,
        y_label: str,
        output_name: str = "multi_scatter.png"
    ) -> str:
        """
        绘制多系列散点图
        
        参数:
            data_series: 数据系列列表，每个元素包含 'x', 'y', 'label'
            x_label: x轴标签
            y_label: y轴标签
            output_name: 输出文件名
        
        返回:
            输出文件路径
        """
        plt.figure(figsize=self.figsize)
        
        # 计算点大小
        total_points = sum(len(s['x']) for s in data_series)
        point_size = max(2, min(1000.0 / total_points, 20))
        
        for series in data_series:
            plt.scatter(
                series['x'], series['y'],
                label=series.get('label', ''),
                s=point_size,
                alpha=0.7
            )
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_name)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return output_path
    
    def plot_results_by_class(
        self,
        results: Dict[str, List[List]],
        column_names: List[str],
        metric_col: int,
        accuracy_col: int,
        output_prefix: str = ""
    ) -> List[str]:
        """
        按类别绘制结果散点图
        
        参数:
            results: 结果字典 {key: rows}
            column_names: 列名
            metric_col: 度量列索引
            accuracy_col: 精度列索引
            output_prefix: 输出文件名前缀
        
        返回:
            输出文件路径列表
        """
        output_paths = []
        
        # 收集所有数据系列
        data_series = []
        for key, rows in results.items():
            if not rows:
                continue
            
            x_values = [row[metric_col] for row in rows]
            y_values = [row[accuracy_col] for row in rows]
            
            data_series.append({
                'x': x_values,
                'y': y_values,
                'label': key
            })
        
        if not data_series:
            return output_paths
        
        # 绘制
        metric_name = column_names[metric_col] if metric_col < len(column_names) else f"col_{metric_col}"
        accuracy_name = column_names[accuracy_col] if accuracy_col < len(column_names) else f"col_{accuracy_col}"
        
        output_name = f"{output_prefix}scatter_{metric_name}_{accuracy_name}.png"
        
        output_path = self.plot_multi_scatter(
            data_series,
            metric_name,
            accuracy_name,
            output_name
        )
        
        output_paths.append(output_path)
        
        return output_paths


def generate_visualization(
    results: Dict[str, List[List]],
    column_names: List[str],
    metric_indices: List[int],
    accuracy_indices: List[int],
    output_dir: str
) -> List[str]:
    """
    生成可视化图表
    
    参数:
        results: 结果字典
        column_names: 列名
        metric_indices: 度量列索引
        accuracy_indices: 精度列索引
        output_dir: 输出目录
    
    返回:
        输出文件路径列表
    """
    visualizer = Visualizer(output_dir)
    output_paths = []
    
    for metric_idx in tqdm(metric_indices, desc="生成图表"):
        for accuracy_idx in accuracy_indices:
            paths = visualizer.plot_results_by_class(
                results,
                column_names,
                metric_idx,
                accuracy_idx
            )
            output_paths.extend(paths)
    
    return output_paths


def plot_correlation_heatmap(
    results: List[List],
    column_names: List[str],
    output_dir: str,
    output_name: str = "correlation_heatmap.png"
) -> str:
    """
    绘制相关性热力图
    
    参数:
        results: 结果行列表
        column_names: 列名
        output_dir: 输出目录
        output_name: 输出文件名
    
    返回:
        输出文件路径
    """
    import pandas as pd
    import seaborn as sns
    
    # 转换为DataFrame
    df = pd.DataFrame(results, columns=column_names)
    
    # 只选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return ""
    
    # 计算相关矩阵
    corr = df[numeric_cols].corr()
    
    # 绘制热力图
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


if __name__ == "__main__":
    # 测试
    viz = Visualizer("./test_output")
    
    # 生成测试数据
    x = [np.random.randn() for _ in range(100)]
    y = [xi + np.random.randn() * 0.5 for xi in x]
    
    path = viz.plot_scatter(x, y, "Metric", "Accuracy", label="test")
    print(f"图表已保存: {path}")
