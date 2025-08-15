# -*- coding: utf-8 -*-
"""
Created on 2025/08/15 11:34:34

@File -> s1_multiscale_decomp_example.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 多尺度分解示例
"""

from itertools import permutations
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

def _gen_embed_series(x: np.ndarray, idxs: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    生成m维时延嵌入序列，返回 shape=(len(idxs), m) 的嵌入矩阵
    """
    N = len(idxs)
    X_embed = np.empty((N, m), dtype=x.dtype)
    for i in range(m):
        X_embed[:, i] = x[idxs - (m - 1 - i) * tau]
    return X_embed


def symbolize(x: np.ndarray, tau: int, m: int, tau_max: int) -> np.ndarray:
    """
    符号化：将嵌入后的序列映射为排列符号索引
    """
    patterns = list(permutations(range(1, m + 1)))
    dict_pattern_index = {p: i for i, p in enumerate(patterns)}

    idxs = np.arange((m - 1) * tau_max, len(x))
    X_embed = _gen_embed_series(x, idxs, m, tau)

    # 按行排序并转为tuple作为pattern
    X = np.array([dict_pattern_index[tuple(np.argsort(row) + 1)] for row in X_embed])
    return X


if __name__ == "__main__":
    y = np.load("runtime/y_series_a_[0.9, 0.97]_w_[0.1, 1.0].npy")

    N2plot = 500

    # 符号化参数
    m = 3
    taus = [1, 10 ,100, 1000]

    # 画图
    plt.figure(figsize=(4, 4))
    plt.subplot(len(taus) + 1, 1, 1)
    plt.plot(y[:N2plot], linewidth=1, c="k", label="$y$")
    plt.legend(loc="upper right")

    tau_max = max(taus)
    for i, tau in enumerate(taus):
        y_symbol = symbolize(y, tau, m, tau_max)

        plt.subplot(len(taus) + 1, 1, i + 2)
        plt.plot(y_symbol[:N2plot], linewidth=1, c="k", label=r"$y_{\text{symbol}}$ " +  f"($m_e = {m}, \\tau_e={tau}$)")
        plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.xlabel("$t$")
    plt.savefig("asset/多尺度符号化分解示例.png", dpi=600, bbox_inches="tight")

