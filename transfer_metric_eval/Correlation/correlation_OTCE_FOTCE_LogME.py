#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation analysis for OTCE / F-OTCE / LogME vs target F1.
- Scans recursively for otce_fotce_logme.csv under a result root directory
- Aggregates rows, computes Pearson & Spearman correlations against F1_t
- Saves a summary CSV and plots (scatter + best-fit line, and a bar chart)

Usage:
  python transfer_metric_eval/Correlation/correlation_OTCE_FOTCE_LogME.py \
    --result_root /abs/path/results/20260116_otce_logme \
    --out_dir /abs/path/results/20260116_otce_logme/Correlation

Output:
  - correlation_summary.csv
  - scatter_<metric>_vs_F1_t.png
  - bar_correlations.png
"""
import argparse
import os
import glob
import math
import csv
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


METRICS = [
    'OTCE_score', 'OTCE_W', 'OTCE_CE',
    'FOTCE_score', 'FOTCE_W', 'FOTCE_CE',
    'LogME_score', 'LogME_src', 'LogME_tar',
]
TARGET_COL = 'F1_t'
CSV_BASENAME = 'otce_fotce_logme.csv'


def find_all_csv(result_root: str) -> List[str]:
    pattern = os.path.join(result_root, '**', CSV_BASENAME)
    return glob.glob(pattern, recursive=True)


def read_concat(csv_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            # Add context columns: direction and class index inferred from path
            parts = p.replace('\\', '/').split('/')
            direction = 'unknown'
            for part in parts:
                if part in ('src_to_tar', 'tar_to_src'):
                    direction = part
                    break
            # class index if present like cls_3
            cls = None
            for part in parts:
                if part.startswith('cls_'):
                    try:
                        cls = int(part.split('_')[1])
                    except Exception:
                        cls = None
            df['__direction'] = direction
            if cls is not None:
                df['__class_index'] = cls
            df['__source_file'] = p
            frames.append(df)
        except Exception as e:
            print(f'[WARN] Failed reading {p}: {e}')
    if not frames:
        raise RuntimeError('No CSV files loaded; please check result_root')
    return pd.concat(frames, ignore_index=True)


def safe_corr(x: pd.Series, y: pd.Series):
    # Drop NaNs and infs
    df = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3:
        return np.nan, np.nan
    px, _ = pearsonr(df['x'], df['y'])
    sx, _ = spearmanr(df['x'], df['y'])
    return float(px), float(sx)


def plot_scatter(x: pd.Series, y: pd.Series, metric: str, out_dir: str):
    df = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return
    plt.figure(figsize=(5.5, 4.2))
    plt.scatter(df['x'], df['y'], s=16, alpha=0.7, edgecolors='none')
    # best-fit line
    try:
        coeffs = np.polyfit(df['x'], df['y'], deg=1)
        xs = np.linspace(df['x'].min(), df['x'].max(), 100)
        ys = coeffs[0]*xs + coeffs[1]
        plt.plot(xs, ys, color='red', linewidth=1.5, label='Linear fit')
    except Exception:
        pass
    plt.xlabel(metric)
    plt.ylabel('F1_t (target)')
    plt.title(f'{metric} vs F1_t')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    out_path = os.path.join(out_dir, f'scatter_{metric}_vs_F1_t.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_bar(corr_rows: List[Dict], out_dir: str):
    labels = [r['metric'] for r in corr_rows]
    pearsons = [r['pearson_r'] for r in corr_rows]
    spearmans = [r['spearman_r'] for r in corr_rows]

    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(max(6, len(labels)*0.8), 4.2))
    plt.bar(x - w/2, pearsons, width=w, label='Pearson r')
    plt.bar(x + w/2, spearmans, width=w, label='Spearman ρ')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylim(-1.0, 1.0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.title('Correlation with F1_t')
    out_path = os.path.join(out_dir, 'bar_correlations.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_root', required=True, help='Root directory containing otce_fotce_logme.csv files')
    ap.add_argument('--out_dir', required=True, help='Directory to save correlation outputs')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csvs = find_all_csv(args.result_root)
    if not csvs:
        raise SystemExit(f'No {CSV_BASENAME} found under: {args.result_root}')

    df = read_concat(csvs)

    # Summary correlations
    rows = []
    for m in METRICS:
        if m not in df.columns:
            print(f'[WARN] Metric not found in data: {m}')
            continue
        pr, sr = safe_corr(df[m], df[TARGET_COL])
        rows.append({'metric': m, 'pearson_r': pr, 'spearman_r': sr, 'n': int(df[[m, TARGET_COL]].dropna().shape[0])})
        # scatter plot
        plot_scatter(df[m], df[TARGET_COL], m, args.out_dir)

    # Save summary CSV
    out_csv = os.path.join(args.out_dir, 'correlation_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'pearson_r', 'spearman_r', 'n'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Saved: {out_csv}')

    # Bar chart
    if rows:
        plot_bar(rows, args.out_dir)
        print(f'Saved: {os.path.join(args.out_dir, "bar_correlations.png")}')


if __name__ == '__main__':
    main()
