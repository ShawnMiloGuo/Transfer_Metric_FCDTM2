import numpy as np
import math
from typing import Dict, Any
from feature.metric_otce import compute_coupling, compute_CE


def compute_f_otce(src_x: np.ndarray, src_y: np.ndarray, tar_x: np.ndarray, tar_y: np.ndarray) -> Dict[str, Any]:
    """
    F-OTCE（原型实现）：
    - 思路：结合耦合代价 W 与条件熵 CE，构造一个归一化/稳健的 OTCE 变体。
    - 公式（占位）：score = CE / (W + eps)
      该形式体现：当跨域代价 W 较大时（域差异大），相同 CE 的风险相对降低；反之亦然。
    - 若需严格对齐某论文的 F-OTCE 定义，请提供公式，我们将替换为精确定义。
    """
    eps = 1e-8
    P, W = compute_coupling(src_x, tar_x)
    CE = compute_CE(P, src_y, tar_y)
    score = CE / (W + eps)
    return {"method": "F-OTCE", "score": float(score), "W": float(W), "CE": float(CE)}
