#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块

提供相关性分析和可视化功能，包括:
- 散点图绘制（带回归线）
- 热力图绘制
- 相关性矩阵计算
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr, linregress

from .config import PostprocessConfig


@dataclass
class CorrelationResult:
    """相关性计算结果"""
    x_col: str
    y_col: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_samples: int


class CorrelationVisualizer:
    """
    相关性可视化和计算器
    
    提供完整的相关性分析和可视化功能。
    """
    
    def __init__(self, config: PostprocessConfig):
        """
        初始化可视化器
        
        参数:
            config: 后处理配置
        """
        self.config = config
        self._setup_style()
    
    def _setup_style(self):
        """设置绘图样式"""
        # 设置字体
        plt.rcParams["font.family"] = self.config.font_family
        plt.rcParams["font.size"] = self.config.font_size
        
        # 设置样式
        sns.set_style("whitegrid")
        
        # 解决中文显示问题
        plt.rcParams["axes.unicode_minus"] = False
    
    def calculate_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson"
    ) -> Tuple[float, float]:
        """
        计算相关系数
        
        参数:
            x: x数据
            y: y数据
            method: 计算方法 ('pearson' 或 'spearman')
        
        返回:
            (相关系数, p值)
        """
        # 移除NaN值
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return np.nan, np.nan
        
        if method == "pearson":
            r, p = pearsonr(x_clean, y_clean)
        elif method == "spearman":
            r, p = spearmanr(x_clean, y_clean)
        else:
            raise ValueError(f"未知的相关方法: {method}")
        
        return r, p
    
    def calculate_correlation_matrix(
        self,
        df: pd.DataFrame,
        x_cols: List[str],
        y_cols: List[str],
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        计算相关性矩阵
        
        参数:
            df: 数据框
            x_cols: x列名列表
            y_cols: y列名列表
            method: 计算方法
        
        返回:
            相关性矩阵DataFrame
        """
        results = []
        
        for x_col in x_cols:
            row = {}
            for y_col in y_cols:
                if x_col in df.columns and y_col in df.columns:
                    r, _ = self.calculate_correlation(
                        df[x_col].values, df[y_col].values, method
                    )
                    row[y_col] = r
                else:
                    row[y_col] = np.nan
            results.append(row)
        
        corr_df = pd.DataFrame(results, index=x_cols)
        return corr_df
    
    def calculate_full_correlation(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ) -> CorrelationResult:
        """
        计算完整的相关性统计
        
        参数:
            df: 数据框
            x_col: x列名
            y_col: y列名
        
        返回:
            CorrelationResult对象
        """
        x = df[x_col].values
        y = df[y_col].values
        
        # 移除NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        pearson_r, pearson_p = self.calculate_correlation(x, y, "pearson")
        spearman_r, spearman_p = self.calculate_correlation(x, y, "spearman")
        
        return CorrelationResult(
            x_col=x_col,
            y_col=y_col,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            n_samples=len(x_clean)
        )
    
    def draw_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        figsize: Tuple[int, int] = (8, 6),
        show_regression: bool = True,
        show_stats: bool = True,
        color: str = "#1f77b4",
        alpha: float = 0.6,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制散点图（带回归线）
        
        参数:
            df: 数据框
            x_col: x列名
            y_col: y列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            figsize: 图表大小
            show_regression: 是否显示回归线
            show_stats: 是否显示统计信息
            color: 点颜色
            alpha: 透明度
            save_path: 保存路径
        
        返回:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 提取数据
        x = df[x_col].values
        y = df[y_col].values
        
        # 移除NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        # 绘制散点
        ax.scatter(x_clean, y_clean, color=color, alpha=alpha, s=30, edgecolors='white', linewidth=0.5)
        
        # 绘制回归线
        if show_regression and len(x_clean) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
            line_x = np.array([x_clean.min(), x_clean.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, color='red', linewidth=2, linestyle='--', label='回归线')
        
        # 显示统计信息
        if show_stats:
            corr_result = self.calculate_full_correlation(df, x_col, y_col)
            stats_text = (
                f"Pearson r = {corr_result.pearson_r:.3f} (p = {corr_result.pearson_p:.3e})\n"
                f"Spearman ρ = {corr_result.spearman_r:.3f} (p = {corr_result.spearman_p:.3e})\n"
                f"n = {corr_result.n_samples}"
            )
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 设置标签
        ax.set_xlabel(xlabel or x_col, fontsize=12)
        ax.set_ylabel(ylabel or y_col, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        if show_regression:
            ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"保存图片: {save_path}")
        
        return fig
    
    def draw_heatmap(
        self,
        corr_df: pd.DataFrame,
        title: str = "",
        figsize: Tuple[int, int] = (10, 8),
        annot: bool = True,
        fmt: str = ".2f",
        vmin: float = -1,
        vmax: float = 1,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制相关性热力图
        
        参数:
            corr_df: 相关性矩阵DataFrame
            title: 图表标题
            figsize: 图表大小
            annot: 是否显示数值
            fmt: 数值格式
            vmin: 最小值
            vmax: 最大值
            save_path: 保存路径
        
        返回:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        sns.heatmap(
            corr_df,
            annot=annot,
            fmt=fmt,
            cmap=self.config.colormap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"保存图片: {save_path}")
        
        return fig
    
    def draw_scatter_all_metrics(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        accuracy_col: str = "F1_delta",
        ncols: int = 3,
        figsize_per_plot: Tuple[float, float] = (5, 4),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制所有度量指标与精度指标的散点图
        
        参数:
            df: 数据框
            metric_cols: 度量列名列表
            accuracy_col: 精度列名
            ncols: 每行子图数量
            figsize_per_plot: 每个子图的大小
            save_path: 保存路径
        
        返回:
            matplotlib Figure对象
        """
        n_metrics = len(metric_cols)
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
        )
        
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, metric_col in enumerate(metric_cols):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            if metric_col not in df.columns:
                ax.text(0.5, 0.5, f"列不存在:\n{metric_col}", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            x = df[metric_col].values
            y = df[accuracy_col].values
            
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            # 绘制散点
            ax.scatter(x_clean, y_clean, alpha=0.6, s=20)
            
            # 绘制回归线
            if len(x_clean) > 2:
                slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
                line_x = np.array([x_clean.min(), x_clean.max()])
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, color='red', linewidth=1.5, linestyle='--')
                
                # 显示相关系数
                ax.text(0.05, 0.95, f"r = {r_value:.3f}", transform=ax.transAxes,
                       fontsize=9, verticalalignment='top')
            
            ax.set_xlabel(metric_col, fontsize=9)
            ax.set_ylabel(accuracy_col, fontsize=9)
            ax.tick_params(labelsize=8)
        
        # 隐藏空白子图
        for idx in range(n_metrics, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"保存图片: {save_path}")
        
        return fig
    
    def save_correlation_to_csv(
        self,
        corr_df: pd.DataFrame,
        save_path: str
    ):
        """
        保存相关性矩阵到CSV
        
        参数:
            corr_df: 相关性矩阵DataFrame
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        corr_df.to_csv(save_path)
        print(f"保存相关性矩阵: {save_path}")


def analyze_and_visualize(
    df: pd.DataFrame,
    metric_cols: List[str],
    accuracy_cols: List[str],
    output_dir: str,
    method: str = "pearson",
    prefix: str = ""
) -> pd.DataFrame:
    """
    便捷函数：执行完整的相关性分析
    
    参数:
        df: 数据框
        metric_cols: 度量列名列表
        accuracy_cols: 精度列名列表
        output_dir: 输出目录
        method: 相关方法
        prefix: 文件名前缀
    
    返回:
        相关性矩阵DataFrame
    """
    config = PostprocessConfig(output_dir=output_dir)
    visualizer = CorrelationVisualizer(config)
    
    # 过滤存在的列
    metric_cols = [c for c in metric_cols if c in df.columns]
    accuracy_cols = [c for c in accuracy_cols if c in df.columns]
    
    # 计算相关性矩阵
    corr_df = visualizer.calculate_correlation_matrix(df, metric_cols, accuracy_cols, method)
    
    # 保存CSV
    csv_path = os.path.join(output_dir, "csv", f"{prefix}correlation_{method}.csv")
    visualizer.save_correlation_to_csv(corr_df, csv_path)
    
    # 绘制热力图
    fig_path = os.path.join(output_dir, "fig", f"{prefix}heatmap_{method}.{config.figure_format}")
    visualizer.draw_heatmap(corr_df, title=f"相关性矩阵 ({method})", save_path=fig_path)
    
    # 绘制散点图
    for acc_col in accuracy_cols:
        scatter_path = os.path.join(output_dir, "fig", f"{prefix}scatter_{acc_col}_{method}.{config.figure_format}")
        visualizer.draw_scatter_all_metrics(df, metric_cols, acc_col, save_path=scatter_path)
    
    plt.close('all')
    
    return corr_df


if __name__ == "__main__":
    # 测试可视化
    np.random.seed(42)
    test_df = pd.DataFrame({
        "metric1": np.random.randn(100),
        "metric2": np.random.randn(100),
        "F1_delta": np.random.randn(100) * 0.1 + 0.5,
    })
    
    config = PostprocessConfig(output_dir="./test_output")
    visualizer = CorrelationVisualizer(config)
    
    # 绘制散点图
    fig = visualizer.draw_scatter(test_df, "metric1", "F1_delta", title="测试散点图")
    plt.show()
