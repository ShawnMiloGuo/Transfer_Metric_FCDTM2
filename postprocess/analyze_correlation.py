#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
相关性分析主程序

读取合并后的CSV文件，计算相关性并生成可视化图表。
支持按类别分析：
- 热力图：行=度量指标，列=类别+迁移方向，值=相关系数
- 散点图：每指标一张图，不同类别用不同颜色
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from postprocess.config import (
    PostprocessConfig, 
    METRIC_COLUMNS, 
    METRIC_SCORE_COLUMNS,
    ACCURACY_COLUMNS, 
    CLASS_NAMES,
    TASK_CONFIGS
)
from postprocess.loader import load_all_csv_files, merge_all_data
from postprocess.visualization import CorrelationVisualizer


def analyze_by_class(
    df: pd.DataFrame,
    metric_cols: List[str],
    accuracy_col: str,
    visualizer: CorrelationVisualizer,
    output_dir: str,
    task_col: str = "_task_name",
    method: str = "pearson"
) -> pd.DataFrame:
    """
    按类别分析相关性
    
    生成：
    1. 热力图：行=度量指标，列=类别+迁移方向，值=相关系数
    2. 散点图：每个度量指标一张图，不同类别用不同颜色
    
    参数:
        df: 数据框
        metric_cols: 度量列名列表
        accuracy_col: 精度指标列名（单个）
        visualizer: 可视化器
        output_dir: 输出目录
        task_col: 任务名称列名
        method: 相关方法
    
    返回:
        相关性矩阵DataFrame
    """
    # 过滤存在的列
    metric_cols = [c for c in metric_cols if c in df.columns]
    
    if accuracy_col not in df.columns:
        print(f"警告: 精度列 {accuracy_col} 不存在，跳过")
        return pd.DataFrame()
    
    # 1. 绘制热力图
    print(f"\n生成 {accuracy_col} 的热力图...")
    heatmap_path = os.path.join(output_dir, "fig", f"heatmap_{accuracy_col}_{method}.png")
    fig, corr_df = visualizer.draw_heatmap_metrics_vs_class_task(
        df, metric_cols, accuracy_col,
        task_col=task_col,
        method=method,
        title=f"Metric vs {accuracy_col} Correlation (by Class and Task)",
        save_path=heatmap_path
    )
    
    if fig:
        plt.close(fig)
    
    # 保存相关性矩阵到CSV
    if not corr_df.empty:
        csv_path = os.path.join(output_dir, "csv", f"correlation_{accuracy_col}_{method}.csv")
        visualizer.save_correlation_to_csv(corr_df, csv_path)
    
    # 2. 绘制散点图
    print(f"生成 {accuracy_col} 的散点图...")
    scatter_dir = os.path.join(output_dir, "fig")
    saved_paths = visualizer.draw_all_scatter_by_class(
        df, metric_cols, accuracy_col,
        class_col="class_index",
        output_dir=scatter_dir,
        prefix=f"{accuracy_col}_"
    )
    print(f"生成了 {len(saved_paths)} 张散点图")
    
    return corr_df


def run_analysis(
    result_root: str,
    output_dir: str,
    metric_types: List[str],
    batch_sizes: List[int],
    correlation_methods: List[str] = None,
    accuracy_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    运行完整的分析流程
    
    支持两种目录结构:
    1. 层级结构: result_root/metric_type/task_name/*.csv
    2. 扁平结构: result_root/*.csv (从文件名推断metric_type和task_name)
    
    参数:
        result_root: 结果文件根目录
        output_dir: 输出目录
        metric_types: 度量类型列表（如 ['FD', 'DS', 'GBC']）
        batch_sizes: 批次大小列表
        correlation_methods: 相关方法列表（如 ['pearson', 'spearman']）
        accuracy_cols: 精度列名列表（None则使用默认）
    
    返回:
        各精度指标的相关性矩阵字典
    """
    # 默认值
    if correlation_methods is None:
        correlation_methods = ["pearson"]
    if accuracy_cols is None:
        accuracy_cols = ACCURACY_COLUMNS
    
    # 创建配置
    config = PostprocessConfig(
        result_root=result_root,
        output_dir=output_dir,
        metric_types=metric_types,
        batch_sizes=batch_sizes,
    )
    
    all_results = {}
    
    # 检查目录结构：扁平结构还是层级结构
    csv_files_in_root = list(Path(result_root).glob("*.csv"))
    
    if csv_files_in_root:
        # 扁平结构：直接在 result_root 下有 CSV 文件
        print(f"\n检测到扁平目录结构: {result_root}")
        print(f"找到 {len(csv_files_in_root)} 个 CSV 文件")
        
        # 加载所有数据
        dfs = load_all_csv_files(result_root)
        if not dfs:
            print("错误: 未找到任何CSV文件")
            return {}
        
        # 合并数据
        df = merge_all_data(dfs)
        print(f"合并后数据: {len(df)} 行, {len(df.columns)} 列")
        
        # 显示任务名称
        print(f"任务名称: {df['_task_name'].unique()}")
        
        # 创建输出目录
        os.makedirs(os.path.join(output_dir, "fig"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)
        
        # 获取所有可用的度量列（合并所有请求的度量类型）
        all_metric_cols = []
        for metric_type in metric_types:
            metric_cols = METRIC_SCORE_COLUMNS.get(metric_type, [])
            all_metric_cols.extend([c for c in metric_cols if c in df.columns])
        
        # 显示可用列
        available_acc = [c for c in accuracy_cols if c in df.columns]
        print(f"可用度量列 ({len(all_metric_cols)}): {all_metric_cols[:5]}...")
        print(f"可用精度列: {available_acc}")
        
        # 创建可视化器
        visualizer = CorrelationVisualizer(config)
        
        # 为每个相关方法执行分析
        for method in correlation_methods:
            print(f"\n使用相关方法: {method}")
            
            # 为每个精度指标执行分析
            for acc_col in available_acc:
                print(f"\n{'-'*50}")
                print(f"分析精度指标: {acc_col}")
                print(f"{'-'*50}")
                
                corr_df = analyze_by_class(
                    df, all_metric_cols, acc_col,
                    visualizer, output_dir,
                    task_col="_task_name",
                    method=method
                )
                
                key = f"all_{acc_col}_{method}"
                all_results[key] = corr_df
    
    else:
        # 层级结构：result_root/metric_type/task_name/*.csv
        print(f"\n检测到层级目录结构: {result_root}")
        
        # 按度量类型处理
        for metric_type in metric_types:
            print(f"\n{'='*60}")
            print(f"处理度量类型: {metric_type}")
            print(f"{'='*60}")
            
            # 获取该度量类型的列名
            metric_cols = METRIC_SCORE_COLUMNS.get(metric_type, [])
            if not metric_cols:
                print(f"警告: 未知度量类型 {metric_type}，跳过")
                continue
            
            # 按任务处理
            for task_name in config.task_names:
                print(f"\n处理任务: {task_name}")
                
                # 构建数据目录路径
                data_dir = os.path.join(result_root, metric_type, task_name)
                if not os.path.exists(data_dir):
                    print(f"警告: 数据目录不存在: {data_dir}，跳过")
                    continue
                
                # 创建输出目录
                task_output_dir = os.path.join(output_dir, metric_type, task_name)
                os.makedirs(os.path.join(task_output_dir, "fig"), exist_ok=True)
                os.makedirs(os.path.join(task_output_dir, "csv"), exist_ok=True)
                
                # 加载数据
                print(f"从 {data_dir} 加载数据...")
                dfs = load_all_csv_files(data_dir)
                if not dfs:
                    print(f"警告: 未找到CSV文件: {data_dir}")
                    continue
                
                # 合并数据
                df = merge_all_data(dfs)
                print(f"合并后数据: {len(df)} 行, {len(df.columns)} 列")
                
                # 显示任务名称
                print(f"任务名称: {df['_task_name'].unique()}")
                
                # 显示可用列
                available_metrics = [c for c in metric_cols if c in df.columns]
                available_acc = [c for c in accuracy_cols if c in df.columns]
                print(f"可用度量列 ({len(available_metrics)}): {available_metrics[:5]}...")
                print(f"可用精度列: {available_acc}")
                
                # 创建可视化器
                visualizer = CorrelationVisualizer(config)
                
                # 为每个相关方法执行分析
                for method in correlation_methods:
                    print(f"\n使用相关方法: {method}")
                    
                    # 为每个精度指标执行分析
                    for acc_col in available_acc:
                        print(f"\n{'-'*50}")
                        print(f"分析精度指标: {acc_col}")
                        print(f"{'-'*50}")
                        
                        corr_df = analyze_by_class(
                            df, available_metrics, acc_col,
                            visualizer, task_output_dir,
                            task_col="_task_name",
                            method=method
                        )
                        
                        key = f"{metric_type}_{task_name}_{acc_col}_{method}"
                        all_results[key] = corr_df
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")
    print(f"结果保存在: {output_dir}")
    print(f"  - 热力图: {os.path.join(output_dir, '*/fig/heatmap_*.png')}")
    print(f"  - 相关性矩阵: {os.path.join(output_dir, '*/csv/correlation_*.csv')}")
    print(f"  - 散点图: {os.path.join(output_dir, '*/fig/*_scatter_*.png')}")
    
    return all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="迁移学习度量相关性分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析FD度量，批次大小为1
  python analyze_correlation.py --result_root ./result --metric_types FD --batch_sizes 1
  
  # 分析多种度量类型
  python analyze_correlation.py --result_root ./result --metric_types FD DS GBC --batch_sizes 1 4
        """
    )
    
    parser.add_argument(
        "--result_root",
        type=str,
        default="./results",
        help="结果文件根目录 (默认: ./results)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis",
        help="输出目录 (默认: ./analysis)"
    )
    
    parser.add_argument(
        "--metric_types",
        type=str,
        nargs="+",
        default=["FD", "DS", "GBC"],
        help="要分析的度量类型 (默认: FD DS GBC)"
    )
    
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 4],
        help="批次大小 (默认: 1 4)"
    )
    
    parser.add_argument(
        "--correlation_methods",
        type=str,
        nargs="+",
        default=["pearson"],
        choices=["pearson", "spearman"],
        help="相关性计算方法 (默认: pearson)"
    )
    
    args = parser.parse_args()
    
    # 运行分析
    run_analysis(
        result_root=args.result_root,
        output_dir=args.output_dir,
        metric_types=args.metric_types,
        batch_sizes=args.batch_sizes,
        correlation_methods=args.correlation_methods,
    )


if __name__ == "__main__":
    main()
