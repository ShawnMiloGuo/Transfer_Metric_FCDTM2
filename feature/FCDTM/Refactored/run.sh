#!/bin/bash
# ============================================================================
# 迁移学习度量计算脚本
# ============================================================================
#
# 功能说明:
#   该脚本用于运行迁移度量计算，支持单独运行FD、DS、GBC、OTCE、LogME五种度量方法。
#
# 使用方法:
#   # 运行所有度量
#   bash run.sh --all
#
#   # 单独运行FD度量
#   bash run.sh --metric FD
#
#   # 单独运行DS度量
#   bash run.sh --metric DS
#
#   # 单独运行GBC度量
#   bash run.sh --metric GBC
#
#   # 单独运行OTCE度量
#   bash run.sh --metric OTCE
#
#   # 单独运行LogME度量
#   bash run.sh --metric LogME
#
#   # 指定任务
#   bash run.sh --metric FD --task dwq_s2_xj_s2
#
# 参数说明:
#   --metric       : 度量类型 (FD/DS/GBC/OTCE/LogME)
#   --task         : 迁移任务名称
#   --batch SIZE   : 批次大小（用于控制每多少张图片计算一个度量值）
#   --batch_target : 按批次处理目标域（每 batch_size 张图片计算一个度量值）
#   --all          : 运行所有度量
#   --help         : 显示帮助信息
#
# 作者: ShanxinGuo
# 创建日期: 2026-03-23
# ============================================================================

set -e  # 遇错即停

# ============================================================================
# 配置参数（在此集中修改）
# ============================================================================

# 路径配置
MODEL_ROOT="/home/Shanxin.Guo/ZhangtuosCode/2_model_pth"
DATA_ROOT="/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset"
RESULT_ROOT="/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/result"

# 计算参数
BATCH_SIZE=1
MAX_IMAGES=100
FEATURE_LAYER="up4"

# 特征提取参数
ONLY_FOREGROUND=true
EXCLUDE_ZERO=false
USE_PREDICTION=false

# 目标域处理
PROCESS_ALL_TARGET=true

# 数据集参数
USE_TRAIN_SET=true
FOREGROUND_RATIO=0.2

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# ============================================================================
# 辅助函数
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

show_help() {
    echo "用法: bash run.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --metric TYPE     运行指定类型的度量 (FD/DS/GBC/OTCE/LogME)"
    echo "  --task NAME       指定迁移任务名称"
    echo "  --batch SIZE      批次大小 (默认: $BATCH_SIZE)"
    echo "  --batch_target    按批次处理目标域（每 batch_size 张图片计算一个度量值）"
    echo "  --all             运行所有度量类型"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 计算单个FD值（汇总所有目标域数据）"
    echo "  bash run.sh --metric FD --task dwq_s2_xj_s2"
    echo ""
    echo "  # 每4张图片计算一个FD值"
    echo "  bash run.sh --metric FD --task dwq_s2_xj_s2 --batch 4 --batch_target"
    echo ""
    echo "  # 运行所有度量"
    echo "  bash run.sh --all"
}

# 构建Python命令参数
build_args() {
    local metric=$1
    local task=$2
    
    args="--metric_type $metric"
    args="$args --task_name $task"
    args="$args --model_root $MODEL_ROOT"
    args="$args --data_root $DATA_ROOT"
    args="$args --result_root $RESULT_ROOT"
    args="$args --batch_size $BATCH_SIZE"
    args="$args --max_images $MAX_IMAGES"
    args="$args --feature_layer $FEATURE_LAYER"
    args="$args --foreground_ratio $FOREGROUND_RATIO"
    
    if [ "$ONLY_FOREGROUND" = true ]; then
        args="$args --only_foreground"
    fi
    
    if [ "$EXCLUDE_ZERO" = true ]; then
        args="$args --exclude_zero_features"
    fi
    
    if [ "$USE_PREDICTION" = true ]; then
        args="$args --use_prediction_labels"
    fi
    
    if [ "$USE_TRAIN_SET" = false ]; then
        args="$args --use_val_set"
    fi
    
    if [ "$PROCESS_ALL_TARGET" = false ]; then
        args="$args --batch_target"
    fi
    
    echo "$args"
}

# 运行单个度量
run_metric() {
    local metric=$1
    local task=$2
    
    log_info "开始计算 ${metric} 度量 (${task})"
    
    args=$(build_args "$metric" "$task")
    log_file="$LOG_DIR/${metric}_${task}_$(date '+%Y%m%d_%H%M%S').log"
    
    python -u main.py $args 2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "${metric} 度量计算完成"
    else
        log_error "${metric} 度量计算失败"
        return 1
    fi
}

# ============================================================================
# 解析命令行参数
# ============================================================================

METRIC_TYPE=""
TASK_NAME="dwq_s2_xj_s2"
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --metric)
            METRIC_TYPE="$2"
            shift 2
            ;;
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --batch_target)
            PROCESS_ALL_TARGET=false
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# 执行度量计算
# ============================================================================

log_info "迁移度量计算脚本启动"
log_info "脚本目录: $SCRIPT_DIR"
log_info "模型目录: $MODEL_ROOT"
log_info "数据目录: $DATA_ROOT"
log_info "结果目录: $RESULT_ROOT"

if [ "$RUN_ALL" = true ]; then
    # 运行所有度量
    for metric in "FD" "DS" "GBC" "OTCE" "LogME"; do
        run_metric "$metric" "$TASK_NAME"
    done
elif [ -n "$METRIC_TYPE" ]; then
    # 运行指定度量
    run_metric "$METRIC_TYPE" "$TASK_NAME"
else
    log_error "请指定 --metric 或 --all 参数"
    show_help
    exit 1
fi

log_info "所有任务执行完成"
