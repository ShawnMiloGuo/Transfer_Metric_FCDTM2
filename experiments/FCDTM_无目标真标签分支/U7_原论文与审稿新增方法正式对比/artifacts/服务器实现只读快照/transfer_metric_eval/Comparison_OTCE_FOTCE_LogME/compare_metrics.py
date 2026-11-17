import argparse
import os
import numpy as np
import csv
from typing import Tuple, Dict, Any

# 复用项目内已有 OTCE 组件
from feature.metric_otce import compute_coupling, compute_CE

# 可选：若存在 csv 格式特征/标签

def load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.csv':
        return np.loadtxt(path, delimiter=',')
    else:
        raise ValueError(f'Unsupported file type: {ext}')


def ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def compute_otce(src_x: np.ndarray, src_y: np.ndarray, tar_x: np.ndarray, tar_y: np.ndarray) -> Dict[str, Any]:
    """
    OTCE: 通过最优传输耦合矩阵 P 与条件熵 CE（基于标签）来度量跨域迁移风险。
    复用项目内 compute_coupling / compute_CE。
    返回：score（示例以 CE 作为主分数，可根据论文替换），并同时返回 W/CE 供对比。
    """
    P, W = compute_coupling(src_x, tar_x)
    CE = compute_CE(P, src_y, tar_y)
    score = CE  # 占位：若需以其他组合形式作为 OTCE 分数（如 W+CE 或加权），可替换此处
    return {"method": "OTCE", "score": float(score), "W": float(W), "CE": float(CE)}


# F-OTCE：在 metric_f_otce.py 中实现，这里仅做统一入口调用
from .metric_f_otce import compute_f_otce


# LogME：在 metric_logme.py 中实现，这里仅做统一入口调用
from .metric_logme import compute_logme


def write_csv(rows: Tuple[Dict[str, Any], ...], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description='Compare OTCE / F-OTCE / LogME metrics without modifying source code')
    parser.add_argument('--src_x', required=True, help='Source features (.npy or .csv)')
    parser.add_argument('--src_y', required=True, help='Source labels (.npy or .csv)')
    parser.add_argument('--tar_x', required=True, help='Target features (.npy or .csv)')
    parser.add_argument('--tar_y', required=True, help='Target labels (.npy or .csv)')
    parser.add_argument('--out', required=True, help='Output CSV path')
    args = parser.parse_args()

    src_x = ensure_2d(load_array(args.src_x))
    tar_x = ensure_2d(load_array(args.tar_x))
    src_y = ensure_2d(load_array(args.src_y)).astype(int)
    tar_y = ensure_2d(load_array(args.tar_y)).astype(int)

    rows = []
    # OTCE
    rows.append(compute_otce(src_x, src_y, tar_x, tar_y))
    # F-OTCE
    rows.append(compute_f_otce(src_x, src_y, tar_x, tar_y))
    # LogME
    rows.append(compute_logme(src_x, src_y, tar_x, tar_y))

    write_csv(tuple(rows), args.out)
    print(f'Wrote results to: {args.out}')


if __name__ == '__main__':
    main()
