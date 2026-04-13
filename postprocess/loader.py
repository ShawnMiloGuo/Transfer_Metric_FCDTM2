#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载模块

提供结果文件加载功能，支持:
- CSV 格式结果文件
- 多任务、多度量类型批量加载
- 数据合并和预处理
"""

import os
import glob
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

from .config import PostprocessConfig, TASK_CONFIGS


@dataclass
class LoadedData:
    """加载的数据结构"""
    metric_type: str
    task_name: str
    batch_size: int
    df: pd.DataFrame
    source: str
    target: str
    file_path: str


class ResultLoader:
    """
    结果数据加载器
    
    支持从重构后的目录结构加载CSV结果文件。
    
    目录结构:
        results/
            FD/
                dwq_s2_xj_s2/
                    summary.csv
                    class_*.csv
            DS/
            ...
    """
    
    # 基础信息列
    BASE_COLUMNS = ["source", "target", "class_index", "class_name"]
    
    def __init__(self, config: PostprocessConfig):
        """
        初始化加载器
        
        参数:
            config: 后处理配置
        """
        self.config = config
        self.loaded_data: Dict[str, LoadedData] = {}
    
    def load_single(
        self,
        metric_type: str,
        task_name: str,
        batch_size: int = 1,
        file_pattern: str = "*.csv"
    ) -> Optional[LoadedData]:
        """
        加载单个任务的结果数据
        
        参数:
            metric_type: 度量类型
            task_name: 任务名称
            batch_size: 批次大小
            file_pattern: 文件匹配模式
        
        返回:
            LoadedData对象，如果加载失败返回None
        """
        # 构建结果目录路径
        result_dir = self.config.get_result_path(metric_type, task_name)
        
        # 查找CSV文件
        csv_files = glob.glob(os.path.join(result_dir, file_pattern))
        
        if not csv_files:
            print(f"警告: 未找到结果文件: {result_dir}/{file_pattern}")
            return None
        
        # 加载所有CSV文件并合并
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df["_source_file"] = os.path.basename(csv_file)
                dfs.append(df)
            except Exception as e:
                print(f"警告: 读取文件失败 {csv_file}: {e}")
        
        if not dfs:
            return None
        
        # 合并数据
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # 获取任务信息
        task_info = self.config.get_task_info(task_name)
        
        return LoadedData(
            metric_type=metric_type,
            task_name=task_name,
            batch_size=batch_size,
            df=combined_df,
            source=task_info.get("source", ""),
            target=task_info.get("target", ""),
            file_path=result_dir
        )
    
    def load_all(self) -> Dict[str, LoadedData]:
        """
        批量加载所有配置的结果数据
        
        返回:
            字典，键为 "metric_task_batch" 格式
        """
        self.loaded_data = {}
        
        total = len(self.config.metric_types) * len(self.config.task_names) * len(self.config.batch_sizes)
        
        with tqdm(total=total, desc="加载数据") as pbar:
            for metric_type in self.config.metric_types:
                for task_name in self.config.task_names:
                    for batch_size in self.config.batch_sizes:
                        key = f"{metric_type}_{task_name}_batch{batch_size}"
                        
                        data = self.load_single(metric_type, task_name, batch_size)
                        if data is not None:
                            self.loaded_data[key] = data
                        
                        pbar.update(1)
        
        print(f"成功加载 {len(self.loaded_data)} 个数据集")
        return self.loaded_data
    
    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        获取指定键的数据框
        
        参数:
            key: 数据键
        
        返回:
            DataFrame或None
        """
        if key in self.loaded_data:
            return self.loaded_data[key].df
        return None
    
    def get_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        获取所有数据框
        
        返回:
            字典，键为数据键，值为DataFrame
        """
        return {k: v.df for k, v in self.loaded_data.items()}
    
    def merge_by_metric(self, metric_type: str) -> Optional[pd.DataFrame]:
        """
        按度量类型合并数据
        
        参数:
            metric_type: 度量类型
        
        返回:
            合并后的DataFrame
        """
        dfs = []
        for key, data in self.loaded_data.items():
            if data.metric_type == metric_type:
                df = data.df.copy()
                df["_task_name"] = data.task_name
                df["_batch_size"] = data.batch_size
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return None
    
    def merge_all(self) -> pd.DataFrame:
        """
        合并所有数据
        
        返回:
            合并后的DataFrame
        """
        dfs = []
        for key, data in self.loaded_data.items():
            df = data.df.copy()
            df["_metric_type"] = data.metric_type
            df["_task_name"] = data.task_name
            df["_batch_size"] = data.batch_size
            dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    @staticmethod
    def get_numeric_columns(df: pd.DataFrame, exclude_patterns: List[str] = None) -> List[str]:
        """
        获取数值列名
        
        参数:
            df: 数据框
            exclude_patterns: 排除的模式列表
        
        返回:
            数值列名列表
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        if exclude_patterns:
            import re
            pattern = "|".join(exclude_patterns)
            numeric_cols = [col for col in numeric_cols 
                          if not re.search(pattern, col, re.IGNORECASE)]
        
        return numeric_cols
    
    @staticmethod
    def filter_by_class(
        df: pd.DataFrame,
        class_index: Optional[int] = None,
        class_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        按类别筛选数据
        
        参数:
            df: 数据框
            class_index: 类别索引
            class_name: 类别名称
        
        返回:
            筛选后的DataFrame
        """
        filtered = df.copy()
        
        if class_index is not None and "class_index" in filtered.columns:
            filtered = filtered[filtered["class_index"] == class_index]
        
        if class_name is not None and "class_name" in filtered.columns:
            filtered = filtered[filtered["class_name"] == class_name]
        
        return filtered
    
    @staticmethod
    def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据摘要统计
        
        参数:
            df: 数据框
        
        返回:
            统计信息字典
        """
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
            "missing_values": df.isnull().sum().sum(),
            "column_types": df.dtypes.value_counts().to_dict(),
        }


def load_results_from_directory(
    result_dir: str,
    metric_types: List[str] = None,
    task_names: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    便捷函数：从目录加载结果
    
    参数:
        result_dir: 结果目录
        metric_types: 度量类型列表
        task_names: 任务名称列表
    
    返回:
        字典，键为 "metric_task" 格式
    """
    if metric_types is None:
        metric_types = ["FD", "DS", "GBC", "OTCE", "LogME"]
    
    if task_names is None:
        task_names = list(TASK_CONFIGS.keys())
    
    config = PostprocessConfig(
        result_root=result_dir,
        metric_types=metric_types,
        task_names=task_names
    )
    
    loader = ResultLoader(config)
    loader.load_all()
    
    return loader.get_all_dataframes()


def load_csv_with_task_name(
    file_path: str,
    task_name: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    从单个CSV文件加载数据，并从文件名提取任务名称
    
    支持的文件名格式:
    - result_FD_dwq_s2_xj_s2.csv
    - result_dwq_sentinel2-xj_sentinel2_batch4.csv
    
    参数:
        file_path: CSV文件路径
        task_name: 可选的任务名称，如果不提供则从文件名提取
    
    返回:
        (DataFrame, 任务名称)
    """
    df = pd.read_csv(file_path)
    
    # 从文件名提取任务名称
    if task_name is None:
        filename = os.path.basename(file_path)
        # 尝试从文件名提取任务信息
        # 格式: result_XXX_taskname.csv 或 result_taskname_batchX.csv
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split("_")
        
        if len(parts) >= 3:
            # 尝试识别任务名称
            # 例如: result_FD_dwq_s2_xj_s2 -> 任务名: dwq_s2_xj_s2
            # 例如: result_dwq_sentinel2-xj_sentinel2_batch4 -> 任务名: dwq_sentinel2-xj_sentinel2
            
            # 跳过 "result" 和度量类型前缀
            start_idx = 1
            if parts[1] in ["FD", "DS", "GBC", "OTCE", "LogME"]:
                start_idx = 2
            
            # 提取任务名称部分（排除最后的 batchX）
            task_parts = []
            for i in range(start_idx, len(parts)):
                if parts[i].startswith("batch"):
                    break
                task_parts.append(parts[i])
            
            task_name = "_".join(task_parts) if task_parts else "unknown"
        else:
            task_name = "unknown"
    
    # 添加任务名称列
    df["_task_name"] = task_name
    
    return df, task_name


def load_all_csv_files(
    data_dir: str,
    file_pattern: str = "*.csv"
) -> Dict[str, pd.DataFrame]:
    """
    加载目录下所有CSV文件
    
    参数:
        data_dir: 数据目录
        file_pattern: 文件匹配模式
    
    返回:
        字典，键为任务名称，值为DataFrame
    """
    csv_files = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not csv_files:
        print(f"警告: 未找到CSV文件: {data_dir}/{file_pattern}")
        return {}
    
    results = {}
    for csv_file in csv_files:
        try:
            df, task_name = load_csv_with_task_name(csv_file)
            results[task_name] = df
            print(f"加载: {csv_file} -> 任务: {task_name}, 行数: {len(df)}")
        except Exception as e:
            print(f"警告: 读取文件失败 {csv_file}: {e}")
    
    return results


def merge_all_data(
    dfs: Dict[str, pd.DataFrame],
    task_col: str = "_task_name"
) -> pd.DataFrame:
    """
    合并多个DataFrame
    
    参数:
        dfs: DataFrame字典
        task_col: 任务列名
    
    返回:
        合并后的DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    all_dfs = []
    for task_name, df in dfs.items():
        df = df.copy()
        if task_col not in df.columns:
            df[task_col] = task_name
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    # 测试加载
    config = PostprocessConfig(result_root="./results")
    loader = ResultLoader(config)
    loader.load_all()
    
    for key, data in loader.loaded_data.items():
        print(f"\n{key}:")
        print(f"  形状: {data.df.shape}")
        print(f"  列: {list(data.df.columns[:10])}...")
