# 新增：与 FCDTM 相同数据调用方式的 OTCE / F-OTCE / LogME
# 说明：不修改原有源码，通过复用 transfer_metric_FD_DS.py 中的数据/特征提取流程，
#       按相同的 dataloader + hook 方式提取源/目标域特征与标签，并计算三种指标。

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils import data

# 复用工程内依赖
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from component.utils import test_path_exist, save_log
from component.dataset import get_dataset_reader

# 直接复用 FD/DS 中已经实现的特征与标签提取函数
from transfer_metric_FD_DS import (
    hook,
    feature_label_and_index_all_batch,
)

# 复用我们新增的对比实现（已放在 transfer_metric_eval 子目录中）
from transfer_metric_eval.Comparison_OTCE_FOTCE_LogME.metric_f_otce import compute_f_otce
from feature.metric_otce import compute_coupling, compute_CE
from transfer_metric_eval.Comparison_OTCE_FOTCE_LogME.metric_logme import compute_logme


def all_index_simple(predictions, labels, num_classes=2):
    """与 FD/DS 中一致的指标计算形式保持最小化一致（这里只返回 OA）。"""
    predictions = predictions.flatten()
    labels = labels.flatten()
    OA = (predictions == labels).float().mean().item()
    return OA


def register_feature_hook(model, feature_layer_name='up4'):
    layer = getattr(model, feature_layer_name)
    handle = layer.register_forward_hook(hook)
    return handle


def compute_otce_from_features(src_x: np.ndarray, src_y: np.ndarray, tar_x: np.ndarray, tar_y: np.ndarray):
    P, W = compute_coupling(src_x, tar_x)
    CE = compute_CE(P, src_y, tar_y)
    score = CE  # 若需其它组合形式(如 W+CE 或加权)可在此替换
    return {"OTCE_score": float(score), "OTCE_W": float(W), "OTCE_CE": float(CE)}


def run_otce_fotce_logme(
    model_path,
    model_chdir,
    data_path_source,
    data_path_target,
    result_path,
    batch_size: int = 1,
    binary_class_index: int = 1,
    label_1_percent: float = 0.2,
    feature_layer_name: str = 'up4',
    by_pred: bool = False,
    dataset_is_train: int = 1,
    log_name: str = 'otce_fotce_logme.log',
):
    # 环境 && 模型
    test_path_exist(result_path)
    os.chdir(model_chdir)
    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=model_device)
    model.eval()

    # 注册 hook，一致的数据调用方式
    handle = register_feature_hook(model, feature_layer_name)

    # 数据集与 DataLoader，与 FCDTM 保持一致的构造方式
    dataset_name = 'rgbn'
    num_workers = 0
    dataset_reader = get_dataset_reader(dataset_name)

    val_dataset_source = dataset_reader(root_dir=data_path_source, is_train=dataset_is_train, transform=None,
                                       binary_class_index=binary_class_index, label_1_percent=label_1_percent)
    val_loader_source = data.DataLoader(val_dataset_source, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    val_dataset_target = dataset_reader(root_dir=data_path_target, is_train=dataset_is_train, transform=None,
                                       binary_class_index=binary_class_index, label_1_percent=label_1_percent)
    val_loader_target = data.DataLoader(val_dataset_target, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    it_src = iter(val_loader_source)
    it_tar = iter(val_loader_target)

    # 从源域获取整批特征与标签（one_batch=False），与 GBC 的方式一致
    OA_s, F1_s, miou_s, precision_s, recall_s, features_s, labels_s = \
        feature_label_and_index_all_batch(it_src, model_device, model, one_batch=False, by_pred=by_pred)

    rows = []

    # 目标域按 batch 遍历，与 FD/DS 保持同样风格（i==0 记录一次）
    for i in tqdm(range(len(val_loader_target)), desc='otce_fotce_logme_batches'):
        OA_t, F1_t, miou_t, precision_t, recall_t, features_t, labels_t = \
            feature_label_and_index_all_batch(it_tar, model_device, model, one_batch=True, by_pred=by_pred)

        # 计算三种度量
        # 1) OTCE
        otce = compute_otce_from_features(features_s, labels_s.reshape(-1, 1), features_t, labels_t.reshape(-1, 1))
        # 2) F-OTCE
        fotce = compute_f_otce(features_s, labels_s.reshape(-1, 1), features_t, labels_t.reshape(-1, 1))
        # 3) LogME（以目标域为主，同时返回源域参考分）
        logme = compute_logme(features_s, labels_s.reshape(-1, 1), features_t, labels_t.reshape(-1, 1))

        if i != 0:
            append_pre = ["", f"{i+1}", "", ""]
        else:
            append_pre = ["source", "target", "class_index", "class_name"]

        row = append_pre + [
            OA_s, F1_s, miou_s, precision_s, recall_s,
            OA_t, F1_t, miou_t, precision_t, recall_t,
            OA_s - OA_t, F1_s - F1_t, miou_s - miou_t, precision_s - precision_t, recall_s - recall_t,
            otce.get('OTCE_score'), otce.get('OTCE_W'), otce.get('OTCE_CE'),
            fotce.get('score'), fotce.get('W'), fotce.get('CE'),
            logme.get('score'), logme.get('logme_src'), logme.get('logme_tar'),
        ]
        rows.append(row)

        # 只取一个 batch 的示例（与其他脚本一致的做法可通过 target_domain_all 控制，这里用简单版本）
        break

    # 清理 hook
    handle.remove()

    # 写出 CSV
    import csv
    out_csv = os.path.join(result_path, 'otce_fotce_logme.csv')
    header = [
        "source", "target", "class_index", "class_name",
        "OA_s", "F1_s", "miou_s", "precision_s", "recall_s",
        "OA_t", "F1_t", "miou_t", "precision_t", "recall_t",
        "OA_delta", "F1_delta", "miou_delta", "precision_delta", "recall_delta",
        "OTCE_score", "OTCE_W", "OTCE_CE",
        "FOTCE_score", "FOTCE_W", "FOTCE_CE",
        "LogME_score", "LogME_src", "LogME_tar",
    ]
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved: {out_csv}")

    # 同时输出日志
    args_like = type('obj', (object,), dict(result_path=result_path, log_name=log_name))
    save_log(args_like)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run OTCE / F-OTCE / LogME with the SAME data calling style as FCDTM')
    parser.add_argument('--model_path', required=True, help='Path to .pth model')
    parser.add_argument('--model_chdir', required=True, help='Chdir before loading model (keep same as existing scripts)')
    parser.add_argument('--data_path_source', required=True)
    parser.add_argument('--data_path_target', required=True)
    parser.add_argument('--result_path', required=True)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--binary_class_index', type=int, default=1)
    parser.add_argument('--label_1_percent', type=float, default=0.2)
    parser.add_argument('--feature_layer_name', type=str, default='up4')
    parser.add_argument('--by_pred', action='store_true')
    parser.add_argument('--dataset_is_train', type=int, default=1)
    parser.add_argument('--log_name', type=str, default='otce_fotce_logme.log')
    args = parser.parse_args()

    run_otce_fotce_logme(
        model_path=args.model_path,
        model_chdir=args.model_chdir,
        data_path_source=args.data_path_source,
        data_path_target=args.data_path_target,
        result_path=args.result_path,
        batch_size=args.batch_size,
        binary_class_index=args.binary_class_index,
        label_1_percent=args.label_1_percent,
        feature_layer_name=args.feature_layer_name,
        by_pred=args.by_pred,
        dataset_is_train=args.dataset_is_train,
        log_name=args.log_name,
    )
