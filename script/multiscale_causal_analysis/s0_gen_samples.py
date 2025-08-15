# -*- coding: utf-8 -*-
"""
Created on 2025/07/25 14:44:24

@File -> s0_coupled_Lorenz_system.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 耦合Lorenz系统
"""

from itertools import permutations
from typing import List
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
    """生成m维时延嵌入序列，返回 shape=(len(idxs), m) 的嵌入矩阵"""
    N = len(idxs)
    X_embed = np.empty((N, m), dtype=x.dtype)
    for i in range(m):
        X_embed[:, i] = x[idxs - (m - 1 - i) * tau]
    return X_embed


def symbolize(x: np.ndarray, tau: int, m: int, tau_max: int) -> np.ndarray:
    """符号化：将嵌入后的序列映射为排列符号索引"""
    patterns = list(permutations(range(1, m + 1)))
    dict_pattern_index = {p: i for i, p in enumerate(patterns)}

    idxs = np.arange((m - 1) * tau_max, len(x))
    X_embed = _gen_embed_series(x, idxs, m, tau)
    # 按行排序并转为tuple作为pattern
    X = np.array([dict_pattern_index[tuple(np.argsort(row) + 1)] for row in X_embed])
    return X


# **************************************************************************************************
# 项目工具
# **************************************************************************************************

def gen_samples(size: int, lags: List[int], a: List[float], w: List[float], show: bool = False):
    """
    生成样本数据

    Args:
        size: 样本数量
        lags: x对y作用的时延，单位为采样点数
        a: 平滑系数，0<alpha<1，越大则输出的响应越慢
    Returns:
        x: 输入信号
        y: 输出信号
    """
    N_preheat = 500  # 预热长度，避免初始条件对结果的影响

    x = np.random.normal(size=size + N_preheat)

    y_1 = np.zeros_like(x)
    for i in range(len(x)):
        if i < lags[0]:
            y_1[i] = x[i]
        else:
            y_1[i] = a[0] * y_1[i - 1] + (1 - a[0]) * x[i - lags[0]]

    y_2 = np.zeros_like(x)
    for i in range(len(x)):
        if i < lags[1]:
            y_2[i] = x[i]
        else:
            y_2[i] = a[1] * y_2[i - 1] + (1 - a[1]) * x[i - lags[1]]

    y = w[0] * y_1 + w[1] * y_2  # 每个时刻上的加和耦合

    # 截去预热部分
    x = x[N_preheat:]
    y = y[N_preheat:]
    y_1 = y_1[N_preheat:]
    y_2 = y_2[N_preheat:]

    if show:
        N2plot = 500
        plt.figure(figsize=(3, 4))
        plt.suptitle(f"$a={a}, w={w}$", fontsize=12)
        plt.subplot(2, 1, 1)
        plt.plot(x[-N2plot:], "k", linewidth=1.5, label="$X$")
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.plot(y[-N2plot:], "k", linewidth=1.5, label="$Y$")
        plt.plot(y_1[-N2plot:], "b--", linewidth=1.5, label="$Y_1$", alpha=0.6)
        plt.plot(y_2[-N2plot:], "r--", linewidth=1.5, label="$Y_2$", alpha=0.6)
        plt.legend(loc="upper right")
        
        plt.xlabel("$t$")
        plt.tight_layout()
        plt.savefig(f"asset/samples_a_{a}_w_{w}.png", dpi=600)
        plt.show()

    return x, y


if __name__ == "__main__":
    a = [0.9, 0.99]
    w = [0.1, 1.0]  # 不同的耦合强度
    x_series, y_series = gen_samples(size=10000, lags=[5, 10], a=a, w=w, show=True)

    # 保存数据
    if not os.path.exists("runtime"):
        os.makedirs("runtime")

    np.save(f"runtime/x_series_a_{a}_w_{w}.npy", x_series)
    np.save(f"runtime/y_series_a_{a}_w_{w}.npy", y_series)