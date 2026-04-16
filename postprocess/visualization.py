#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块

提供相关性分析和可视化功能，包括:
- 散点图绘制（带回归线，按类别着色）
- 热力图绘制（度量指标 vs 类别+迁移方向）
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

from .config import PostprocessConfig, CLASS_NAMES


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
        plt.rcParams["font.family"] = self.config.font_family
        plt.rcParams["font.size"] = self.config.font_size
        sns.set_style("whitegrid")
        plt.rcParams["axes.unicode_minus"] = False
    
    def calculate_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson"
    ) -> Tuple[float, float]:
        """计算相关系数"""
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
        """计算相关性矩阵"""
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
    
    def calculate_correlation_for_heatmap(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        accuracy_col: str,
        class_col: str = "class_index",
        task_col: str = "_task_name",
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        计算用于热力图的相关性矩阵
        
        热力图结构：
        - 行（index）：度量指标（如 FD_sum, mean_dif_absolute_sum 等）
        - 列（columns）：类别+迁移方向（如 "Cropland_dwq_s2_xj_s2"）
        - 值：该度量指标在该类别+迁移方向下，与精度指标的相关系数
        
        参数:
            df: 数据框（可能包含多个任务的数据）
            metric_cols: 度量列名列表
            accuracy_col: 精度指标列名（单个，如 "F1_delta"）
            class_col: 类别列名
            task_col: 任务名称列名（需要提前添加到数据中）
            method: 相关方法
        
        返回:
            相关性矩阵DataFrame
        """
        # 过滤存在的列
        metric_cols = [c for c in metric_cols if c in df.columns]
        
        if accuracy_col not in df.columns:
            print(f"警告: 精度列 {accuracy_col} 不存在")
            return pd.DataFrame()
        
        # 获取所有唯一的（任务，类别）组合
        if task_col in df.columns:
            tasks = df[task_col].unique()
        else:
            tasks = ["unknown"]
        
        classes = df[class_col].unique()
        classes = sorted([c for c in classes if pd.notna(c)])
        
        # 构建列标签列表：类别_任务名
        col_labels = []
        for cls in classes:
            cls_name = CLASS_NAMES.get(int(cls), f"class_{int(cls)}")
            # 从数据中获取该类别的 class_name
            cls_df_sample = df[df[class_col] == cls]
            if "class_name" in cls_df_sample.columns:
                cls_name = cls_df_sample["class_name"].iloc[0]
            
            for task in tasks:
                if pd.isna(task):
                    continue
                col_label = f"{cls_name}_{task}"
                col_labels.append((cls, task, col_label))
        
        # 计算每个（度量指标，类别_任务）的相关系数
        results = {}
        
        for metric_col in metric_cols:
            row = {}
            for cls, task, col_label in col_labels:
                # 筛选数据
                mask = (df[class_col] == cls)
                if task_col in df.columns:
                    mask = mask & (df[task_col] == task)
                
                subset = df[mask]
                
                if len(subset) < 3:
                    row[col_label] = np.nan
                    continue
                
                # 计算相关系数
                x = subset[metric_col].values
                y = subset[accuracy_col].values
                
                r, _ = self.calculate_correlation(x, y, method)
                row[col_label] = r
            
            results[metric_col] = row
        
        corr_df = pd.DataFrame(results).T
        return corr_df
    
    def draw_scatter_one_metric_by_class(
        self,
        df: pd.DataFrame,
        metric_col: str,
        accuracy_col: str,
        class_col: str = "class_index",
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        figsize: Tuple[int, int] = (8, 6),
        show_regression: bool = False,
        legend: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制单个度量指标的散点图（不同迁移过程+类别用不同颜色）
        
        这是一个指标一张散点图，不同迁移过程+类别用不同颜色区分。
        标签格式为 "SOURCE->TARGET ClassName"（如 "DWQ_S2->XJ_S2 Cropland"）
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取所有唯一的（迁移过程，类别）组合
        groups = []
        if 'source' in df.columns and 'target' in df.columns:
            # 按 source->target 和 class 分组
            # 去重获取唯一的组合
            unique_combos = df[['source', 'target', class_col, 'class_name']].drop_duplicates()
            
            for _, row in unique_combos.iterrows():
                source = str(row['source']).upper()
                target = str(row['target']).upper()
                
                # 获取类别索引和名称
                cls = row[class_col]
                if pd.isna(cls):
                    continue
                    
                cls = int(cls) if not isinstance(cls, str) else cls
                cls_name = str(row.get('class_name', ''))
                if cls_name == 'nan' or cls_name == '':
                    cls_name = CLASS_NAMES.get(int(cls) if isinstance(cls, int) else cls, f'class_{cls}')
                
                groups.append({
                    'source': source,
                    'target': target,
                    'class_index': cls,
                    'label': f"{source}->{target} {cls_name}"
                })
        else:
            # 只有类别分组
            classes = df[class_col].unique()
            classes = sorted([c for c in classes if pd.notna(c)])
            for cls in classes:
                cls = int(cls) if not isinstance(cls, str) else cls
                cls_name = CLASS_NAMES.get(cls, f"class_{cls}")
                if "class_name" in df.columns:
                    cls_name_val = df[df[class_col] == cls]["class_name"].iloc[0] if len(df[df[class_col] == cls]) > 0 else cls_name
                    if pd.notna(cls_name_val):
                        cls_name = str(cls_name_val)
                groups.append({
                    'source': '',
                    'target': '',
                    'class_index': cls,
                    'label': cls_name
                })
        
        # 颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        
        for idx, group in enumerate(groups):
            if 'source' in df.columns:
                # 将 source 和 target 列转换为字符串类型，避免 .str 访问器报错
                source_col = df['source'].astype(str).str.upper()
                target_col = df['target'].astype(str).str.upper()
                # 按 source->target 和 class 筛选
                mask = (source_col == group['source']) & \
                       (target_col == group['target']) & \
                       (df[class_col] == group['class_index'])
            else:
                mask = df[class_col] == group['class_index']
            
            group_df = df[mask]
            x = group_df[metric_col].values
            y = group_df[accuracy_col].values
            
            # 移除NaN
            mask_valid = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask_valid]
            y_clean = y[mask_valid]
            
            if len(x_clean) == 0:
                continue
            
            # 绘制散点
            ax.scatter(x_clean, y_clean, color=colors[idx], alpha=1, 
                      s=10, label=group['label'])
            
            # 绘制回归线
            if show_regression and len(x_clean) > 2:
                slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
                line_x = np.array([x_clean.min(), x_clean.max()])
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, color=colors[idx], linewidth=1.5, 
                       linestyle='--', alpha=0.7)
        
        # 设置标签
        ax.set_xlabel(xlabel or metric_col, fontsize=12)
        ax.set_ylabel(ylabel or accuracy_col, fontsize=12)
        ax.set_title(title or f"{metric_col} vs {accuracy_col}", fontsize=14)
        
        if legend and len(groups) > 0:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"保存图片: {save_path}")
        
        return fig
    
    def draw_all_scatter_by_class(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        accuracy_col: str,
        class_col: str = "class_index",
        output_dir: str = "./fig",
        prefix: str = "",
        metric_name_map: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        为每个度量指标绘制一张散点图（不同迁移过程+类别用不同颜色）
        
        参数:
            df: 数据框
            metric_cols: 度量列名列表
            accuracy_col: 精度指标列名
            class_col: 类别列名
            output_dir: 输出目录
            prefix: 文件名前缀
            metric_name_map: 度量名称映射字典，用于将度量列名映射为简短的显示名称
                           例如: {"mean_dif_absolute_y0_y1_diff": "FCDTM"}
        """
        saved_paths = []
        
        for metric_col in metric_cols:
            if metric_col not in df.columns:
                continue
            
            # 获取度量显示名称
            display_name = metric_name_map.get(metric_col, metric_col) if metric_name_map else metric_col
            
            save_path = os.path.join(
                output_dir, 
                f"{prefix}scatter_{metric_col}_vs_{accuracy_col}.{self.config.figure_format}"
            )
            
            self.draw_scatter_one_metric_by_class(
                df, metric_col, accuracy_col,
                class_col=class_col,
                title=f"{display_name} vs {accuracy_col}",
                xlabel=display_name,
                save_path=save_path
            )
            plt.close()
            saved_paths.append(save_path)
        
        return saved_paths
    
    def draw_heatmap_metrics_vs_class_task(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        accuracy_col: str,
        class_col: str = "class_index",
        task_col: str = "_task_name",
        method: str = "pearson",
        title: str = "",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        绘制热力图
        
        热力图结构：
        - 行（纵坐标）：度量指标（如 FD_sum, mean_dif_absolute_sum 等）
        - 列（横坐标）：类别+迁移方向（如 "Cropland_dwq_s2_xj_s2"）
        - 值：该度量指标在该类别+迁移方向下，与精度指标的相关系数
        
        参数:
            df: 数据框（可能包含多个任务的数据）
            metric_cols: 度量列名列表
            accuracy_col: 精度指标列名（单个）
            class_col: 类别列名
            task_col: 任务名称列名
            method: 相关方法
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        
        返回:
            (Figure, 相关性矩阵DataFrame)
        """
        # 计算相关性矩阵
        corr_df = self.calculate_correlation_for_heatmap(
            df, metric_cols, accuracy_col,
            class_col=class_col,
            task_col=task_col,
            method=method
        )
        
        if corr_df.empty:
            print("警告: 相关性矩阵为空")
            return None, corr_df
        
        # 自动调整图表大小
        if figsize is None:
            n_rows = len(corr_df)
            n_cols = len(corr_df.columns)
            figsize = (max(12, n_cols * 1.5), max(10, n_rows * 0.5))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            cmap=self.config.colormap,
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            ax=ax,
            annot_kws={"size": 8}
        )
        
        # 设置标题和标签
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f"Correlation Heatmap ({accuracy_col}, {method})", fontsize=14)
        
        ax.set_xlabel("Class_Task", fontsize=12)
        ax.set_ylabel("Metric", fontsize=12)
        
        # 调整标签
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"保存图片: {save_path}")
        
        return fig, corr_df
    
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
        """绘制热力图（通用方法）"""
        fig, ax = plt.subplots(figsize=figsize)
        
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
        """保存相关性矩阵到CSV"""
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
    """
    config = PostprocessConfig(output_dir=output_dir)
    visualizer = CorrelationVisualizer(config)
    
    # 过滤存在的列
    metric_cols = [c for c in metric_cols if c in df.columns]
    accuracy_cols = [c for c in accuracy_cols if c in df.columns]
    
    results = {}
    
    # 为每个精度指标绘制热力图
    for acc_col in accuracy_cols:
        # 绘制热力图
        heatmap_path = os.path.join(output_dir, "fig", f"{prefix}heatmap_{acc_col}_{method}.{config.figure_format}")
        fig, corr_df = visualizer.draw_heatmap_metrics_vs_class_task(
            df, metric_cols, acc_col,
            method=method,
            save_path=heatmap_path
        )
        if fig:
            plt.close(fig)
        
        # 保存CSV
        csv_path = os.path.join(output_dir, "csv", f"{prefix}correlation_{acc_col}_{method}.csv")
        visualizer.save_correlation_to_csv(corr_df, csv_path)
        
        # 绘制散点图
        visualizer.draw_all_scatter_by_class(
            df, metric_cols, acc_col,
            output_dir=os.path.join(output_dir, "fig"),
            prefix=f"{prefix}{acc_col}_"
        )
        
        results[acc_col] = corr_df
    
    return results


if __name__ == "__main__":
    # 测试可视化
    np.random.seed(42)
    
    # 创建测试数据
    test_df = pd.DataFrame({
        "metric1": np.random.randn(200),
        "metric2": np.random.randn(200),
        "F1_delta": np.random.randn(200) * 0.1 + 0.5,
        "class_index": np.tile([1, 2, 3, 6], 50),
        "_task_name": np.tile(["task1", "task2"], 100),
    })
    
    config = PostprocessConfig(output_dir="./test_output")
    visualizer = CorrelationVisualizer(config)
    
    # 测试热力图
    fig, corr_df = visualizer.draw_heatmap_metrics_vs_class_task(
        test_df, ["metric1", "metric2"], "F1_delta"
    )
    plt.show()
