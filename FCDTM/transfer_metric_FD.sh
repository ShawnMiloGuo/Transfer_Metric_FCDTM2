#!/bin/bash
# ============================================================================
# 迁移学习度量指标计算脚本
# ============================================================================
#
# 功能说明:
#   该脚本用于批量运行迁移度量计算，支持三种度量方法：
#   - FD (Fréchet Distance): 基于特征分布的Fréchet距离
#   - DS (Dispersion Score): 基于类别间特征分散度的度量  
#   - GBC (Geometric Bayesian Classifier): 基于几何贝叶斯分类器的度量
#
# 使用方法:
#   bash transfer_metric_FD.sh
#
# 参数说明:
#   - transfer_metric_name: 度量类型 (FD/DS/GBC)
#   - batch_size: 批次大小 (1/4)
#   - by_pred: 标签类型 (0=真实标签, 1=预测标签)
#   - only_label_1: 特征提取范围 (0=所有类别, 1=仅前景类)
#   - target_domain_all: 目标域处理方式 (0=按批次, 1=全部)
#   - task_transfer: 迁移任务名称
#
# 作者: ShanxinGuo
# 创建日期: 2026-3-23
# 重构日期: 2024-xx-xx
# ============================================================================

# ============================================================================
# 初始化配置
# ============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 日志文件名
LOG_FILE="20260323_transfer_metric_FD_DS.log"

# 创建日志目录
mkdir -p ./result
# ============================================================================
# 可配置参数
# ============================================================================

# 最大处理图像数量
MAX_IMAGES=100

# 特征提取层名称 (可选: up4, outc, down4)
FEATURE_LAYER="up4"

# 是否排除零值特征 (0: 否, 1: 是)
EXCLUDE_ZERO_FEATURES=0

# 数据集类型 (1: 训练集, 0: 验证集)
DATASET_TYPE=1

# ============================================================================
# 迁移任务定义
# ============================================================================

# 迁移任务列表
# - dwq_s2_xj_s2: 大湾区 Sentinel2 -> 新疆 Sentinel2 (跨区域)
# - dwq_l8_xj_l8: 大湾区 Landsat8 -> 新疆 Landsat8 (跨区域)
# - dwq_s2_dwq_l8: 大湾区 Sentinel2 -> 大湾区 Landsat8 (跨传感器)
# - xj_s2_xj_l8: 新疆 Sentinel2 -> 新疆 Landsat8 (跨传感器)
TASK_LIST=("dwq_s2_xj_s2" "dwq_l8_xj_l8" "dwq_s2_dwq_l8" "xj_s2_xj_l8")

# ============================================================================
# 辅助函数
# ============================================================================

# 打印带时间戳的日志
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

# 构建结果输出路径
build_result_path() {
    local metric_name=$1
    local task_order=$2
    local task_name=$3
    local order=$4
    local label_type=$5
    local batch_type=$6
    local label_source=$7
    
    echo "${metric_name}_${task_order}_${task_name}/${order}-${metric_name}_${label_type}all-${batch_type}_${MAX_IMAGES}img_${label_source}"
}

# ============================================================================
# 主程序
# ============================================================================

log_info "开始执行迁移度量计算..."
log_info "脚本目录: $SCRIPT_DIR"
log_info "日志文件: $LOG_FILE"

# 任务计数器
task_order=0
total_order=0

# ============================================================================
# 遍历所有度量方法
# ============================================================================

for metric_name in "FD" "DS" "GBC"; do
    
    log_info "=========================================="
    log_info "开始计算 ${metric_name} 度量"
    log_info "=========================================="
    
    # 设置结果输出根目录
    case $metric_name in
        "FD")
            RESULT_BASE_DIR="/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/result/1_FD/"
            ;;
        "DS")
            RESULT_BASE_DIR="/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/result/2_DS/"
            ;;
        "GBC")
            RESULT_BASE_DIR="/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/result/3_GBC/"
            ;;
    esac
    
    # ----------------------------------------------------------------
    # 遍历标签类型 (真实标签 vs 预测标签)
    # ----------------------------------------------------------------
    for use_prediction in 0 1; do
        
        label_source="by_pred${use_prediction}"
        
        if [ $use_prediction -eq 0 ]; then
            log_info "使用真实标签进行度量计算"
        else
            log_info "使用预测标签进行度量计算"
        fi
        
        # ----------------------------------------------------------------
        # 遍历批次大小
        # ----------------------------------------------------------------
        for batch_size in 1 4; do
            
            log_info "--- 批次大小: ${batch_size} ---"
            
            # ----------------------------------------------------------------
            # 遍历特征提取范围
            # ----------------------------------------------------------------
            # FD方法支持两种模式，GBC和DS默认使用所有类别
            if [ $metric_name == "FD" ]; then
                label_range_list="0 1"
            else
                label_range_list="0"
            fi
            
            for only_label_1 in $label_range_list; do
                
                # 设置标签范围描述
                if [ $only_label_1 -eq 0 ]; then
                    label_type="-"      # 同时计算前景和背景
                else
                    label_type="label1" # 只计算前景类
                fi
                
                # ----------------------------------------------------------------
                # 遍历目标域处理方式
                # ----------------------------------------------------------------
                for process_all_target in 0 1; do
                    
                    # 跳过无效组合: batch_size=4 且处理全部目标域
                    if [ $batch_size -eq 4 ] && [ $process_all_target -eq 1 ]; then
                        continue
                    fi
                    
                    # 设置处理方式描述
                    if [ $process_all_target -eq 1 ]; then
                        batch_type="all"  # 目标域所有数据
                    else
                        batch_type="batch${batch_size}"  # 按批次处理
                    fi
                    
                    ((total_order++))
                    
                    # ----------------------------------------------------------------
                    # 遍历迁移任务
                    # ----------------------------------------------------------------
                    task_order=0
                    for task_name in "${TASK_LIST[@]}"; do
                        
                        ((task_order++))
                        
                        # 构建结果路径
                        result_path="${RESULT_BASE_DIR}${task_order}_${task_name}/${total_order}-${metric_name}_${label_type}all-${batch_type}_${MAX_IMAGES}img_${label_source}"
                        log_name="transfer_metric_${metric_name}_${task_name}.log"
                        
                        log_info "任务: ${task_name}, 结果路径: ${result_path}"
                        
                        # ----------------------------------------------------------------
                        # 执行 Python 计算脚本
                        # 说明: 
                        #   - stdout (print输出) 重定向到日志文件
                        #   - stderr (tqdm进度条) 输出到终端显示
                        # ----------------------------------------------------------------
                        python -u -W ignore transfer_metric_FD_DS.py \
                            --batch_size $batch_size \
                            --target_domain_all $process_all_target \
                            --no_feature0 $EXCLUDE_ZERO_FEATURES \
                            --feature_layer_name $FEATURE_LAYER \
                            --only_label_1 $only_label_1 \
                            --result_path "$result_path" \
                            --log_name "$log_name" \
                            --task_transfer "$task_name" \
                            --transfer_metric_name "$metric_name" \
                            --by_pred $use_prediction \
                            --dataset_is_train $DATASET_TYPE \
                            >> ./result/$LOG_FILE
                        
                        # 检查执行状态
                        if [ $? -eq 0 ]; then
                            log_info "任务 ${task_name} 完成"
                        else
                            log_error "任务 ${task_name} 执行失败"
                        fi
                        
                    done  # 遍历迁移任务
                    
                done  # 遍历目标域处理方式
                
            done  # 遍历特征提取范围
            
        done  # 遍历批次大小
        
    done  # 遍历标签类型
    
    log_info "${metric_name} 度量计算完成"
    
done  # 遍历度量方法

# ============================================================================
# 完成
# ============================================================================

log_info "=========================================="
log_info "所有任务执行完成"
log_info "总执行次数: ${total_order}"
log_info "日志文件: ./result/${LOG_FILE}"
log_info "=========================================="
