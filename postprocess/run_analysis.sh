#!/bin/bash
# =============================================================================
# 迁移度量相关性分析脚本
# 
# 功能: 分析迁移度量指标与精度下降之间的相关性
# 输入: 重构后的结果文件 (CSV格式)
# 输出: 相关性矩阵CSV、散点图、热力图
#
# 使用方法:
#   ./run_analysis.sh                           # 默认配置运行
#   ./run_analysis.sh --cross-domain            # 包含跨域分析
#   ./run_analysis.sh --metric_types FD DS      # 只分析FD和DS
# =============================================================================

# 设置错误处理
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/analysis}"
METRIC_TYPES="${METRIC_TYPES:-FD DS GBC}"
BATCH_SIZES="${BATCH_SIZES:-1 4}"
CROSS_DOMAIN=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --result_root)
            RESULT_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --metric_types)
            METRIC_TYPES="$2"
            shift 2
            ;;
        --batch_sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --cross_domain|--cross-domain)
            CROSS_DOMAIN="--cross_domain"
            shift
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --result_root DIR    结果文件根目录 (默认: $PROJECT_ROOT/results)"
            echo "  --output_dir DIR     输出目录 (默认: $PROJECT_ROOT/analysis)"
            echo "  --metric_types LIST  要分析的度量类型 (默认: FD DS GBC)"
            echo "  --batch_sizes LIST   批次大小 (默认: 1 4)"
            echo "  --cross-domain       执行跨域分析"
            echo "  -h, --help           显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "迁移度量相关性分析"
echo "============================================================"
echo "结果目录: $RESULT_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "度量类型: $METRIC_TYPES"
echo "批次大小: $BATCH_SIZES"
echo "============================================================"

# 检查结果目录是否存在
if [ ! -d "$RESULT_ROOT" ]; then
    echo "错误: 结果目录不存在: $RESULT_ROOT"
    echo "请先运行度量计算程序生成结果文件"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行分析程序
cd "$PROJECT_ROOT"

python postprocess/analyze_correlation.py \
    --result_root "$RESULT_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --metric_types $METRIC_TYPES \
    --batch_sizes $BATCH_SIZES \
    $CROSS_DOMAIN

echo ""
echo "============================================================"
echo "分析完成!"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"
