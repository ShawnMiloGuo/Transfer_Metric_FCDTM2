#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移度量计算主入口

支持单独运行FD、DS、GBC度量计算。

使用方法:
    # 运行FD度量
    python main.py --metric_type FD --task_name dwq_s2_xj_s2
    
    # 运行DS度量
    python main.py --metric_type DS --task_name dwq_s2_xj_s2
    
    # 运行GBC度量
    python main.py --metric_type GBC --task_name dwq_s2_xj_s2
    
    # 自定义配置
    python main.py --metric_type FD --batch_size 4 --max_images 200
"""

import os
import json
import csv
import glob
from typing import Dict, List
from tqdm import tqdm

import torch

from config import Config, print_config_summary, CLASS_NAMES
from model import ModelManager, create_dataloader
from metrics import get_metric
from visualization import generate_visualization


class TransferMetricRunner:
    """
    迁移度量计算运行器
    
    提供统一的接口运行不同类型的度量计算。
    """
    
    def __init__(self, config: Config):
        """
        初始化
        
        参数:
            config: 配置对象
        """
        self.config = config
        self.model_manager = ModelManager()
        
        # 初始化度量计算器
        self.metric_class = get_metric(config.metric_type)
        
        # 存储结果
        self.results: Dict[str, List[List]] = {}
        self.column_names: List[str] = []
    
    def run(self) -> Dict[str, List[List]]:
        """
        执行度量计算
        
        返回:
            结果字典
        """
        print_config_summary(self.config)
        
        # 确保输出目录存在
        result_path = self.config.get_result_path()
        os.makedirs(result_path, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(result_path, "config.json")
        self.config.save(config_path)
        print(f"配置已保存: {config_path}")
        
        # 获取数据集读取器
        from component.dataset import get_dataset_reader
        dataset_reader = get_dataset_reader('rgbn')
        
        # 获取类别索引
        class_indices = self.config.class_indices
        
        # 数据集索引（双向迁移）
        dataset_indices = [0, 1]
        
        # ========== 遍历数据集组合 ==========
        for dataset_idx in dataset_indices:
            # 确定源域和目标域
            if dataset_idx == 0:
                source_domain = self.config.source_dataset
                target_domain = self.config.target_dataset
            else:
                source_domain = self.config.target_dataset
                target_domain = self.config.source_dataset
            
            source_path = self.config.get_data_path(source_domain)
            target_path = self.config.get_data_path(target_domain)
            
            # ========== 遍历各类别 ==========
            for class_idx in class_indices:
                class_name = self.config.get_class_name(class_idx)
                
                print(f"\n{'='*60}")
                print(f"处理: {source_domain} -> {target_domain}, 类别: {class_idx} ({class_name})")
                print(f"{'='*60}")
                
                result_key = f"{source_domain}-{target_domain}_cls_{class_idx}"
                self.results[result_key] = []
                
                # 查找模型文件
                model_pattern = self.config.get_model_path(class_idx)
                model_files = glob.glob(model_pattern)
                
                if not model_files:
                    print(f"警告: 未找到模型文件: {model_pattern}")
                    continue
                
                # 替换模型路径中的数据集名称
                actual_model_pattern = os.path.join(
                    self.config.model_root,
                    f"train_{source_domain}_cls_{class_idx}",
                    "unet_*_best_val.pth"
                )
                model_files = glob.glob(actual_model_pattern)
                
                if not model_files:
                    print(f"警告: 未找到模型文件: {actual_model_pattern}")
                    continue
                
                print(f"加载模型: {model_files[0]}")
                
                # 加载模型
                model = self.model_manager.load_model(model_files[0])
                
                # 注册钩子
                hook_handle = self.model_manager.register_hook(model, self.config.feature_layer)
                
                # 创建数据加载器
                source_dataset = dataset_reader(
                    root_dir=source_path,
                    is_train=self.config.use_train_set,
                    transform=None,
                    binary_class_index=class_idx,
                    label_1_percent=self.config.foreground_ratio_threshold
                )
                source_loader = create_dataloader(
                    source_dataset,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers
                )
                
                target_dataset = dataset_reader(
                    root_dir=target_path,
                    is_train=self.config.use_train_set,
                    transform=None,
                    binary_class_index=class_idx,
                    label_1_percent=self.config.foreground_ratio_threshold
                )
                target_loader = create_dataloader(
                    target_dataset,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers
                )
                
                # 创建度量计算器并计算
                metric = self.metric_class(self.config)
                
                try:
                    metric_results = metric.compute(
                        model,
                        self.model_manager,
                        source_loader,
                        target_loader
                    )
                    
                    # 更新结果中的class信息
                    for r in metric_results:
                        r.class_index = class_idx
                        r.class_name = class_name
                    
                    # 转换为行数据
                    self.column_names = metric.get_column_names()
                    rows = [r.to_list(self.column_names) for r in metric_results]
                    self.results[result_key] = rows
                    
                    print(f"完成: 获得 {len(rows)} 条结果记录")
                    
                except Exception as e:
                    print(f"计算错误: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    # 移除钩子
                    self.model_manager.remove_hook()
        
        # ========== 保存结果 ==========
        self._save_results(result_path)
        
        # ========== 生成可视化 ==========
        if self.config.save_figures:
            self._generate_visualization(result_path)
        
        return self.results
    
    def _save_results(self, output_dir: str):
        """保存结果到文件"""
        print(f"\n{'='*60}")
        print("保存结果...")
        print(f"{'='*60}")
        
        # 保存列名
        column_path = os.path.join(output_dir, "column_names.json")
        with open(column_path, "w", encoding="utf-8") as f:
            json.dump(self.column_names, f, ensure_ascii=False, indent=2)
        print(f"列名已保存: {column_path}")
        
        # 保存CSV
        csv_path = os.path.join(output_dir, f"result_{self.config.metric_type}_{self.config.task_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.column_names)
            for key, rows in self.results.items():
                writer.writerows(rows)
        print(f"结果已保存: {csv_path}")
        
        # 保存JSON
        json_path = os.path.join(output_dir, f"result_{self.config.metric_type}_{self.config.task_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {json_path}")
    
    def _generate_visualization(self, output_dir: str):
        """生成可视化图表"""
        print(f"\n{'='*60}")
        print("生成可视化图表...")
        print(f"{'='*60}")
        
        fig_dir = os.path.join(output_dir, "fig")
        
        # 获取绘图索引
        metric = self.metric_class(self.config)
        metric_indices, accuracy_indices = metric.get_plot_indices()
        
        if not metric_indices or not accuracy_indices:
            print("未定义绘图索引，跳过可视化")
            return
        
        # 生成图表
        output_paths = generate_visualization(
            self.results,
            self.column_names,
            metric_indices,
            accuracy_indices,
            fig_dir
        )
        
        print(f"已生成 {len(output_paths)} 张图表")


def main():
    """主函数"""
    # 解析配置
    config = Config.from_args()
    
    # 创建运行器并执行
    runner = TransferMetricRunner(config)
    results = runner.run()
    
    print(f"\n{'='*60}")
    print("所有任务执行完成!")
    print(f"结果已保存到: {config.get_result_path()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
