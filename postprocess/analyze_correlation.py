#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移度量相关性分析主程序

分析迁移度量指标与精度下降之间的相关性，生成:
1. 相关性矩阵CSV文件
2. 散点图（带回归线）
3. 热力图

使用方法:
    python analyze_correlation.py --result_root ./results --output_dir ./analysis
    python analyze_correlation.py --metric_types FD DS --task_names dwq_s2_xj_s2
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postprocess.config import PostprocessConfig, METRIC_SCORE_COLUMNS, print_config
from postprocess.loader import ResultLoader, LoadedData
from postprocess.visualization import CorrelationVisualizer


def analyze_single_dataset(
    data: LoadedData,
    visualizer: CorrelationVisualizer,
    config: PostprocessConfig,
    accuracy_cols: List[str]
) -> Dict[str, str]:
    """
    分析单个数据集
    
    参数:
        data: 加载的数据
        visualizer: 可视化器
        config: 配置
        accuracy_cols: 精度列名列表
    
    返回:
        生成的文件路径字典
    """
    results = {}
    df = data.df
    
    # 获取度量分数列
    metric_cols = config.get_metric_score_columns(data.metric_type)
    metric_cols = [c for c in metric_cols if c in df.columns]
    
    if not metric_cols:
        print(f"警告: 未找到度量分数列 {data.metric_type}")
        return results
    
    # 过滤精度列
    acc_cols = [c for c in accuracy_cols if c in df.columns]
    
    if not acc_cols:
        print(f"警告: 未找到精度列")
        return results
    
    # 构建输出子目录
    output_subdir = f"{data.metric_type}/{data.task_name}/batch{data.batch_size}"
    
    # 对每种相关性方法进行分析
    for method in config.correlation_methods:
        # 计算相关性矩阵
        corr_df = visualizer.calculate_correlation_matrix(
            df, metric_cols, acc_cols, method
        )
        
        # 保存CSV
        csv_dir = config.get_output_path(output_subdir, "csv")
        csv_path = os.path.join(csv_dir, f"correlation_{method}.csv")
        visualizer.save_correlation_to_csv(corr_df, csv_path)
        results[f"csv_{method}"] = csv_path
        
        # 绘制热力图
        fig_dir = config.get_output_path(output_subdir, "fig")
        heatmap_path = os.path.join(fig_dir, f"heatmap_{method}.{config.figure_format}")
        visualizer.draw_heatmap(
            corr_df,
            title=f"{data.metric_type} - {data.task_name} ({method})",
            save_path=heatmap_path
        )
        results[f"heatmap_{method}"] = heatmap_path
        
        # 绘制散点图（每个精度指标）
        for acc_col in acc_cols:
            scatter_path = os.path.join(
                fig_dir, 
                f"scatter_{acc_col}_{method}.{config.figure_format}"
            )
            visualizer.draw_scatter_all_metrics(
                df, metric_cols, acc_col,
                save_path=scatter_path
            )
            results[f"scatter_{acc_col}_{method}"] = scatter_path
    
    return results


def analyze_cross_domain(
    loader: ResultLoader,
    visualizer: CorrelationVisualizer,
    config: PostprocessConfig,
    accuracy_cols: List[str]
):
    """
    跨域分析（原始代码的主要功能）
    
    对每个度量类型，合并所有任务的数据进行分析。
    """
    for metric_type in config.metric_types:
        print(f"\n{'='*60}")
        print(f"分析度量类型: {metric_type}")
        print('='*60)
        
        # 合并该度量类型的所有任务数据
        df = loader.merge_by_metric(metric_type)
        
        if df is None or df.empty:
            print(f"警告: 无数据 {metric_type}")
            continue
        
        # 获取度量分数列
        metric_cols = config.get_metric_score_columns(metric_type)
        metric_cols = [c for c in metric_cols if c in df.columns]
        
        acc_cols = [c for c in accuracy_cols if c in df.columns]
        
        if not metric_cols or not acc_cols:
            print(f"警告: 缺少必要的列")
            continue
        
        # 构建输出子目录
        output_subdir = f"{metric_type}/cross_domain"
        
        for method in config.correlation_methods:
            # 计算相关性矩阵
            corr_df = visualizer.calculate_correlation_matrix(
                df, metric_cols, acc_cols, method
            )
            
            # 保存CSV
            csv_path = config.get_output_path(output_subdir, "csv", f"correlation_{method}.csv")
            visualizer.save_correlation_to_csv(corr_df, csv_path)
            
            # 绘制热力图
            heatmap_path = config.get_output_path(
                output_subdir, "fig", f"heatmap_{method}.{config.figure_format}"
            )
            visualizer.draw_heatmap(
                corr_df,
                title=f"{metric_type} - Cross Domain ({method})",
                save_path=heatmap_path
            )
            
            # 绘制散点图
            for acc_col in acc_cols:
                scatter_path = config.get_output_path(
                    output_subdir, "fig", 
                    f"scatter_{acc_col}_{method}.{config.figure_format}"
                )
                visualizer.draw_scatter_all_metrics(
                    df, metric_cols, acc_col,
                    save_path=scatter_path
                )


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="迁移度量相关性分析工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--result_root", type=str, default="./results",
                       help="结果文件根目录")
    parser.add_argument("--output_dir", type=str, default="./analysis",
                       help="输出目录")
    parser.add_argument("--metric_types", type=str, nargs="+",
                       default=["FD", "DS", "GBC"],
                       help="要分析的度量类型")
    parser.add_argument("--task_names", type=str, nargs="+",
                       default=None,
                       help="要分析的任务名称（默认全部）")
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                       default=[1, 4],
                       help="要分析的批次大小")
    parser.add_argument("--correlation_methods", type=str, nargs="+",
                       default=["pearson", "spearman"],
                       help="相关性计算方法")
    parser.add_argument("--accuracy_cols", type=str, nargs="+",
                       default=["F1_delta"],
                       help="精度指标列名")
    parser.add_argument("--cross_domain", action="store_true",
                       help="执行跨域分析")
    
    args = parser.parse_args()
    
    # 获取默认任务名称
    if args.task_names is None:
        from postprocess.config import TASK_CONFIGS
        args.task_names = list(TASK_CONFIGS.keys())
    
    # 创建配置
    config = PostprocessConfig(
        result_root=args.result_root,
        output_dir=args.output_dir,
        metric_types=args.metric_types,
        task_names=args.task_names,
        batch_sizes=args.batch_sizes,
        correlation_methods=args.correlation_methods,
        accuracy_columns=args.accuracy_cols,
    )
    
    print_config(config)
    
    # 创建加载器和可视化器
    loader = ResultLoader(config)
    visualizer = CorrelationVisualizer(config)
    
    # 加载数据
    print("\n正在加载结果数据...")
    loader.load_all()
    
    if not loader.loaded_data:
        print("错误: 未找到任何结果数据")
        return 1
    
    # 分析每个数据集
    print("\n正在分析数据...")
    for key, data in loader.loaded_data.items():
        print(f"\n处理: {key}")
        print(f"  数据形状: {data.df.shape}")
        
        analyze_single_dataset(
            data, visualizer, config, args.accuracy_cols
        )
    
    # 跨域分析
    if args.cross_domain:
        print("\n执行跨域分析...")
        analyze_cross_domain(loader, visualizer, config, args.accuracy_cols)
    
    # 保存分析摘要
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "result_root": config.result_root,
            "output_dir": config.output_dir,
            "metric_types": config.metric_types,
            "task_names": config.task_names,
            "batch_sizes": config.batch_sizes,
            "correlation_methods": config.correlation_methods,
        },
        "loaded_datasets": list(loader.loaded_data.keys()),
    }
    
    summary_path = config.get_output_path("analysis_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析完成! 输出目录: {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
