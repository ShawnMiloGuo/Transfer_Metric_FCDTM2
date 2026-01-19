#!/usr/bin/env bash
set -euo pipefail

# 运行示例：
# bash run_compare.sh \
#   --src_x /path/src_features.npy \
#   --src_y /path/src_labels.npy \
#   --tar_x /path/tar_features.npy \
#   --tar_y /path/tar_labels.npy \
#   --out result/compare_dwq_s2_to_xj_s2.csv

mkdir -p result
python3 compare_metrics.py "$@"
