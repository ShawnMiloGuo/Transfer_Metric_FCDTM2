#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块

集中管理所有参数、路径和任务配置，方便实验管理和扩展。

使用方法:
    from config import Config
    
    # 使用默认配置
    config = Config()
    
    # 从命令行参数加载
    config = Config.from_args()
    
    # 自定义配置
    config = Config(
        metric_type="FD",
        batch_size=4,
        model_root="/path/to/models"
    )
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json


# ============================================================================
# 项目根目录定位
# ============================================================================

def find_project_root() -> str:
    """
    智能定位项目根目录
    
    按优先级尝试:
    1. 环境变量 PROJECT_ROOT
    2. VSCode 工作区环境变量
    3. 向上查找项目标志文件
    4. 当前工作目录
    """
    # 方式1: 自定义环境变量
    if 'PROJECT_ROOT' in os.environ:
        project_root = os.environ['PROJECT_ROOT']
        if os.path.isdir(project_root):
            return project_root
    
    # 方式2: VSCode 工作区路径
    if 'VSCODE_WORKSPACE_PATH' in os.environ:
        workspace = os.environ['VSCODE_WORKSPACE_PATH']
        if os.path.isdir(workspace):
            return workspace
    
    # 方式3: 向上查找项目标志文件
    project_markers = ['.git', 'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    while current_dir != os.path.dirname(current_dir):
        for marker in project_markers:
            if os.path.exists(os.path.join(current_dir, marker)):
                return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # 方式4: 当前工作目录
    return os.getcwd()


PROJECT_ROOT = find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 枚举定义
# ============================================================================

class MetricType(Enum):
    """度量类型枚举"""
    FD = "FD"       # Fréchet Distance
    DS = "DS"       # Dispersion Score
    GBC = "GBC"     # Geometric Bayesian Classifier
    OTCE = "OTCE"   # Optimal Transport for Conditional Estimation
    LogME = "LogME" # Log Maximum Evidence


class FeatureLayer(Enum):
    """特征提取层枚举"""
    DECODER_UP4 = "up4"       # 解码器第4层
    DECODER_OUTC = "outc"     # 输出层
    ENCODER_DOWN4 = "down4"   # 编码器第4层


# ============================================================================
# 任务配置
# ============================================================================

@dataclass
class TaskConfig:
    """单个迁移任务配置"""
    name: str
    source_dataset: str
    target_dataset: str
    class_indices: List[int]
    description: str = ""


# 预定义的任务配置
TASK_CONFIGS: Dict[str, TaskConfig] = {
    "dwq_s2_xj_s2": TaskConfig(
        name="dwq_s2_xj_s2",
        source_dataset="dwq_sentinel2",
        target_dataset="xj_sentinel2",
        class_indices=[1, 2, 3, 6, 7, 8],
        description="大湾区 Sentinel2 -> 新疆 Sentinel2 (跨区域)"
    ),
    "dwq_l8_xj_l8": TaskConfig(
        name="dwq_l8_xj_l8",
        source_dataset="dwq_landsat8",
        target_dataset="xj_landsat8",
        class_indices=[1, 6, 7, 8],
        description="大湾区 Landsat8 -> 新疆 Landsat8 (跨区域)"
    ),
    "dwq_s2_dwq_l8": TaskConfig(
        name="dwq_s2_dwq_l8",
        source_dataset="dwq_sentinel2",
        target_dataset="dwq_landsat8",
        class_indices=[1, 2, 6, 7, 8],
        description="大湾区 Sentinel2 -> 大湾区 Landsat8 (跨传感器)"
    ),
    "xj_s2_xj_l8": TaskConfig(
        name="xj_s2_xj_l8",
        source_dataset="xj_sentinel2",
        target_dataset="xj_landsat8",
        class_indices=[1, 3, 5, 6, 7, 8],
        description="新疆 Sentinel2 -> 新疆 Landsat8 (跨传感器)"
    ),
}

# 数据集名称到路径模板的映射
DATASET_PATH_TEMPLATE = {
    "dwq_sentinel2": "dwq_sentinel2/train_val",
    "dwq_landsat8": "dwq_landsat8/train_val",
    "xj_sentinel2": "xj_sentinel2/train_val",
    "xj_landsat8": "xj_landsat8/train_val",
}

# 类别名称
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
# 主配置类
# ============================================================================

@dataclass
class Config:
    """
    主配置类
    
    集中管理所有实验参数，支持:
    - 默认值设置
    - 命令行参数解析
    - 配置文件加载
    - 参数验证
    """
    # ========== 路径配置 ==========
    model_root: str = "/home/Shanxin.Guo/ZhangtuosCode/2_model_pth"
    data_root: str = "/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset"
    result_root: str = "/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/result"
    
    # ========== 计算参数 ==========
    metric_type: str = "FD"
    task_name: str = "dwq_s2_xj_s2"
    batch_size: int = 1
    max_images: int = 100
    num_workers: int = 0
    
    # ========== 特征提取参数 ==========
    feature_layer: str = "up4"
    only_foreground: bool = False  # 只提取前景类特征
    exclude_zero_features: bool = False
    use_prediction_labels: bool = False
    
    # ========== 目标域处理 ==========
    process_all_target: bool = True
    
    # ========== 数据集参数 ==========
    use_train_set: bool = True
    foreground_ratio_threshold: float = 0.2
    
    # ========== 输出参数 ==========
    log_name: str = "transfer_metric.log"
    save_figures: bool = True
    
    # ========== 内部属性 ==========
    _task_config: Optional[TaskConfig] = field(default=None, repr=False)
    
    def __post_init__(self):
        """初始化后处理"""
        self._validate()
        self._task_config = TASK_CONFIGS.get(self.task_name)
    
    def _validate(self):
        """验证配置参数"""
        # 验证度量类型
        valid_metrics = [m.value for m in MetricType]
        if self.metric_type not in valid_metrics:
            raise ValueError(f"无效的度量类型: {self.metric_type}, 可选: {valid_metrics}")
        
        # 验证任务名称
        if self.task_name not in TASK_CONFIGS:
            raise ValueError(f"未知的任务: {self.task_name}, 可选: {list(TASK_CONFIGS.keys())}")
        
        # 验证特征层
        valid_layers = [l.value for l in FeatureLayer]
        if self.feature_layer not in valid_layers:
            raise ValueError(f"无效的特征层: {self.feature_layer}, 可选: {valid_layers}")
    
    @property
    def task_config(self) -> TaskConfig:
        """获取当前任务配置"""
        if self._task_config is None:
            self._task_config = TASK_CONFIGS[self.task_name]
        return self._task_config
    
    @property
    def class_indices(self) -> List[int]:
        """获取类别索引列表"""
        return self.task_config.class_indices
    
    @property
    def source_dataset(self) -> str:
        """获取源域数据集名称"""
        return self.task_config.source_dataset
    
    @property
    def target_dataset(self) -> str:
        """获取目标域数据集名称"""
        return self.task_config.target_dataset
    
    def get_model_path(self, class_index: int) -> str:
        """
        获取指定类别的模型路径
        
        参数:
            class_index: 类别索引
        
        返回:
            模型文件路径模板
        """
        return os.path.join(
            self.model_root,
            f"train_{self.source_dataset}_cls_{class_index}",
            "unet_*_best_val.pth"
        )
    
    def get_data_path(self, dataset_name: str) -> str:
        """
        获取数据集路径
        
        参数:
            dataset_name: 数据集名称
        
        返回:
            数据集完整路径
        """
        path_suffix = DATASET_PATH_TEMPLATE.get(dataset_name, f"{dataset_name}/train_val")
        return os.path.join(self.data_root, path_suffix)
    
    def get_result_path(self, sub_dir: str = "") -> str:
        """
        获取结果输出路径
        
        参数:
            sub_dir: 子目录名称
        
        返回:
            结果输出路径
        """
        metric_dir = f"{MetricType(self.metric_type).value}"
        path = os.path.join(self.result_root, metric_dir, self.task_name)
        if sub_dir:
            path = os.path.join(path, sub_dir)
        return path
    
    def get_class_name(self, class_index: int) -> str:
        """获取类别名称"""
        if 0 <= class_index < len(CLASS_NAMES):
            return CLASS_NAMES[class_index]
        return f"class_{class_index}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_root": self.model_root,
            "data_root": self.data_root,
            "result_root": self.result_root,
            "metric_type": self.metric_type,
            "task_name": self.task_name,
            "batch_size": self.batch_size,
            "max_images": self.max_images,
            "feature_layer": self.feature_layer,
            "only_foreground": self.only_foreground,
            "exclude_zero_features": self.exclude_zero_features,
            "use_prediction_labels": self.use_prediction_labels,
            "process_all_target": self.process_all_target,
            "use_train_set": self.use_train_set,
            "foreground_ratio_threshold": self.foreground_ratio_threshold,
        }
    
    def save(self, path: str):
        """保存配置到JSON文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_args(cls, args: Optional[List[str]] = None) -> "Config":
        """
        从命令行参数创建配置
        
        参数:
            args: 命令行参数列表，None表示使用sys.argv
        
        返回:
            Config实例
        """
        parser = argparse.ArgumentParser(
            description="迁移学习度量指标计算工具",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # 路径参数
        parser.add_argument("--model_root", type=str, default=cls.model_root,
                          help="模型文件根目录")
        parser.add_argument("--data_root", type=str, default=cls.data_root,
                          help="数据集根目录")
        parser.add_argument("--result_root", type=str, default=cls.result_root,
                          help="结果输出根目录")
        
        # 计算参数
        parser.add_argument("--metric_type", type=str, default=cls.metric_type,
                          choices=[m.value for m in MetricType],
                          help="度量类型")
        parser.add_argument("--task_name", type=str, default=cls.task_name,
                          choices=list(TASK_CONFIGS.keys()),
                          help="迁移任务名称")
        parser.add_argument("--batch_size", type=int, default=cls.batch_size,
                          help="批次大小")
        parser.add_argument("--max_images", type=int, default=cls.max_images,
                          help="最大处理图像数量")
        parser.add_argument("--num_workers", type=int, default=cls.num_workers,
                          help="数据加载线程数")
        
        # 特征提取参数
        parser.add_argument("--feature_layer", type=str, default=cls.feature_layer,
                          choices=[l.value for l in FeatureLayer],
                          help="特征提取层")
        parser.add_argument("--only_foreground", action="store_true",
                          help="只提取前景类特征")
        parser.add_argument("--exclude_zero_features", action="store_true",
                          help="排除零值特征")
        parser.add_argument("--use_prediction_labels", action="store_true",
                          help="使用预测标签而非真实标签")
        
        # 目标域处理
        # 默认 process_all_target=True（处理所有目标域，计算单个度量值）
        # 使用 --batch_target 可切换为按批次处理（每批次计算一个度量值）
        parser.add_argument("--batch_target", action="store_true",
                          help="按批次处理目标域（每批次计算一个度量值，而非汇总计算单个值）")
        
        # 数据集参数
        parser.add_argument("--use_val_set", action="store_true",
                          help="使用验证集而非训练集")
        parser.add_argument("--foreground_ratio", type=float, 
                          default=cls.foreground_ratio_threshold,
                          help="前景像素比例阈值")
        
        # 输出参数
        parser.add_argument("--log_name", type=str, default=cls.log_name,
                          help="日志文件名")
        parser.add_argument("--no_figures", action="store_true",
                          help="不生成可视化图表")
        
        parsed = parser.parse_args(args)
        
        return cls(
            model_root=parsed.model_root,
            data_root=parsed.data_root,
            result_root=parsed.result_root,
            metric_type=parsed.metric_type,
            task_name=parsed.task_name,
            batch_size=parsed.batch_size,
            max_images=parsed.max_images,
            num_workers=parsed.num_workers,
            feature_layer=parsed.feature_layer,
            only_foreground=parsed.only_foreground,
            exclude_zero_features=parsed.exclude_zero_features,
            use_prediction_labels=parsed.use_prediction_labels,
            process_all_target=not parsed.batch_target,
            use_train_set=not parsed.use_val_set,
            foreground_ratio_threshold=parsed.foreground_ratio,
            log_name=parsed.log_name,
            save_figures=not parsed.no_figures,
        )
    
    @classmethod
    def from_file(cls, path: str) -> "Config":
        """从JSON文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# 便捷函数
# ============================================================================

def get_available_metrics() -> List[str]:
    """获取所有可用的度量类型"""
    return [m.value for m in MetricType]


def get_available_tasks() -> List[str]:
    """获取所有可用的任务名称"""
    return list(TASK_CONFIGS.keys())


def print_config_summary(config: Config):
    """打印配置摘要"""
    print("=" * 60)
    print("配置摘要")
    print("=" * 60)
    print(f"度量类型: {config.metric_type}")
    print(f"任务名称: {config.task_name}")
    print(f"任务描述: {config.task_config.description}")
    print(f"源域数据集: {config.source_dataset}")
    print(f"目标域数据集: {config.target_dataset}")
    print(f"类别索引: {config.class_indices}")
    print(f"批次大小: {config.batch_size}")
    print(f"特征层: {config.feature_layer}")
    print(f"最大图像数: {config.max_images}")
    print("=" * 60)


if __name__ == "__main__":
    # 测试配置
    config = Config.from_args()
    print_config_summary(config)
