import numpy as np
from typing import Dict, Any

"""
一个简化版 LogME 实现（原型）：
- 参考思想：对线性/岭回归的贝叶斯证据（log marginal likelihood）进行近似，
  以衡量给定特征在目标域上拟合标签的可解释性/可迁移性。
- 处理分类标签：采用 one-vs-rest，将类别标签转为 {0,1} 并分别计算证据，最后取平均。
- 说明：该实现为轻量近似版，便于无外部依赖情况下快速对比；若需严格复现论文版 LogME，
  我们可以替换为更严谨的数值实现或接入现有库。
"""


def _ridge_log_evidence(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, sigma2: float = 1.0) -> float:
    """
    近似计算岭回归的对数边际似然（证据）。
    C = sigma^2 I + X X^T / alpha
    log p(y | X) ≈ -0.5 * [ n log(2π) + log|C| + y^T C^{-1} y ]
    """
    n = X.shape[0]
    # C 为 n×n，适合中小规模；大规模可改用 Woodbury 恒等式降低维度
    Xt = X
    C = sigma2 * np.eye(n) + Xt @ Xt.T / alpha
    # 使用稳定的线性求解代替显式逆
    # y^T C^{-1} y = (C^{-1} y)·y
    sol = np.linalg.solve(C, y)
    quad = float(y.T @ sol)
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        # 数值退化时回退一个稳定值
        logdet = np.log(np.finfo(float).eps)
    return -0.5 * (n * np.log(2 * np.pi) + logdet + quad)


def _one_vs_rest_logme(X: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    scores = []
    for c in classes:
        yc = (y == c).astype(float)
        # 中心化以增强数值稳定性
        yc = yc - yc.mean()
        scores.append(_ridge_log_evidence(X, yc, alpha=1.0, sigma2=1.0))
    return float(np.mean(scores))


def compute_logme(src_x: np.ndarray, src_y: np.ndarray, tar_x: np.ndarray, tar_y: np.ndarray) -> Dict[str, Any]:
    """
    比较设计：以目标域特征/标签计算 LogME 作为主要评分（更贴近迁移落地表现），
    同时提供源域评分供参考。
    返回字段：{"method": "LogME", "score": <target_score>, "logme_src": <src_score>, "logme_tar": <tar_score>}
    """
    # 目标域为主
    tar_score = _one_vs_rest_logme(tar_x, tar_y.flatten())
    src_score = _one_vs_rest_logme(src_x, src_y.flatten())
    return {"method": "LogME", "score": float(tar_score), "logme_src": float(src_score), "logme_tar": float(tar_score)}
