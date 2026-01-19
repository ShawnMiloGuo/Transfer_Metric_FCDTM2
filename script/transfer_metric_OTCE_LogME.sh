#!/usr/bin/env bash
# One-click batch runner for OTCE / F-OTCE / LogME
# Usage example:
#   bash script/transfer_metric_OTCE_LogME.sh \
#     --model_path "/abs/path/unet_epoch29.pth" \
#     --model_chdir "/abs/path/Transfer_Metric_FCDTM" \
#     --data_path_source "/abs/path/dataset_src" \
#     --data_path_target "/abs/path/dataset_tar" \
#     --result_path "/abs/path/results/20260116_otce_logme" \
#     --classes "1 2 3 4 5 6 7 8" \
#     --feature_layer_name up4 \
#     --batch_size 1 \
#     --label_1_percent 0.2 \
#     --dataset_is_train 1 \
#     --by_pred false
#
# Notes:
# - Keeps the SAME data calling style as FCDTM scripts (loader + forward hook).
# - Produces one CSV per direction (src->tar, tar->src): otce_fotce_logme.csv
# - You can run multiple times with different result_path to organize outputs.

set -euo pipefail

# ---------- default params ----------
MODEL_PATH=""
MODEL_CHDIR=""
DATA_PATH_SOURCE=""
DATA_PATH_TARGET=""
RESULT_PATH=""
CLASSES="1 2 3 4 5 6 7 8"
FEATURE_LAYER_NAME="up4"
BATCH_SIZE=1
LABEL_1_PERCENT=0.2
DATASET_IS_TRAIN=1
BY_PRED=false

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --model_chdir) MODEL_CHDIR="$2"; shift 2 ;;
    --data_path_source) DATA_PATH_SOURCE="$2"; shift 2 ;;
    --data_path_target) DATA_PATH_TARGET="$2"; shift 2 ;;
    --result_path) RESULT_PATH="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --feature_layer_name) FEATURE_LAYER_NAME="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --label_1_percent) LABEL_1_PERCENT="$2"; shift 2 ;;
    --dataset_is_train) DATASET_IS_TRAIN="$2"; shift 2 ;;
    --by_pred) BY_PRED="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---------- validations ----------
if [[ -z "$MODEL_PATH" || -z "$MODEL_CHDIR" || -z "$DATA_PATH_SOURCE" || -z "$DATA_PATH_TARGET" || -z "$RESULT_PATH" ]]; then
  echo "[ERROR] Required: --model_path --model_chdir --data_path_source --data_path_target --result_path";
  exit 1
fi

mkdir -p "$RESULT_PATH"

run_once() {
  local SRC_PATH="$1"; shift
  local TAR_PATH="$1"; shift
  local OUT_DIR="$1"; shift

  mkdir -p "$OUT_DIR"

  echo "\n=== Running batches: SRC=$SRC_PATH -> TAR=$TAR_PATH ==="
  for C in $CLASSES; do
    echo "[Class $C]"
    python3 feature/transfer_metric_OTCE_LogME.py \
      --model_path "$MODEL_PATH" \
      --model_chdir "$MODEL_CHDIR" \
      --data_path_source "$SRC_PATH" \
      --data_path_target "$TAR_PATH" \
      --result_path "$OUT_DIR/cls_${C}" \
      --batch_size "$BATCH_SIZE" \
      --binary_class_index "$C" \
      --label_1_percent "$LABEL_1_PERCENT" \
      --feature_layer_name "$FEATURE_LAYER_NAME" \
      --dataset_is_train "$DATASET_IS_TRAIN" \
      $( [[ "$BY_PRED" == "true" ]] && echo "--by_pred" )
  done
}

# ---------- run: src -> tar ----------
run_once "$DATA_PATH_SOURCE" "$DATA_PATH_TARGET" "$RESULT_PATH/src_to_tar"

# ---------- run: tar -> src ----------
run_once "$DATA_PATH_TARGET" "$DATA_PATH_SOURCE" "$RESULT_PATH/tar_to_src"

echo "\nAll done. Results at: $RESULT_PATH"
