# -*- coding: utf-8 -*-
"""
Created on 2025/07/25 16:17:59

@File -> test_symbolization.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 测试符号化
"""

from itertools import permutations
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from script.multiscale_causal_analysis.s0_gen_samples import gen_samples


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

# def _gen_embed_series(x: np.ndarray, idxs: np.ndarray, m: int, tau: int) -> np.ndarray:
#     """
#     生成m维时延嵌入序列，返回 shape=(len(idxs), m) 的嵌入序列矩阵
#     """
#     N = len(idxs)
#     X_embed = np.empty((N, m), dtype=x.dtype)
#     for i in range(m):
#         X_embed[:, i] = x[idxs - (m - 1 - i) * tau]
#     return X_embed


# def symbolize(x: np.ndarray, tau: int, m: int, tau_max: int) -> np.ndarray:
#     """
#     符号化：将嵌入后的序列映射为排列符号索引
    
#     Args:
#         x: 输入的一维时间序列
#         tau: 时间延迟
#         m: 嵌入维度
#         tau_max: 最大时间延迟
    
#     Note:
#         嵌入序列的起始对应索引为 (m - 1) * tau_max
#     """
#     patterns = list(permutations(range(m)))
#     dict_pattern_index = {p: i for i, p in enumerate(patterns)}

#     idxs = np.arange((m - 1) * tau_max, len(x))
#     X_embed = _gen_embed_series(x, idxs, m, tau)

#     # 按行排序并转为tuple作为pattern
#     X = np.array([dict_pattern_index[tuple(np.argsort(row))] for row in X_embed])
#     return X


class SeriesSymbolizer(object):
    """一维序列符号化"""

    def __init__(self, m: int, tau: int):
        self.m = m
        self.tau = tau

        # 随机初始模式符号关系
        self.init_patterns = list(permutations(range(m)))
        self.init_pattern_idx_dict = {p: i for i, p in enumerate(self.init_patterns)}

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """
        生成m维时延嵌入窗口样本

        Args:
            x: 输入的一维时间序列

        Returns:
            X_embed: shape=(len(x), m) 的嵌入序列矩阵
        """
        # 对应于索引t的窗口序列为：[x_t, x_{t + tau}, ..., x_{t + (m-1)*tau}]
        X_embed = np.empty((len(x), self.m), dtype=x.dtype)
        for i in range(self.m - 1, -1, -1):
            X_embed[:, i] = np.roll(x, -(self.m - 1 - i) * self.tau)  # 负号表示向左向上
        
        return X_embed
    
    def symbolize(self, x: np.ndarray) -> np.ndarray:
        """
        符号化：将嵌入后的序列映射为排列符号索引
        """
        x = x.flatten()
        x_embed = self._embed(x)

        # 按行排序并转为tuple作为pattern
        x_symbol = np.array([self.init_pattern_idx_dict[tuple(np.argsort(row))] for row in x_embed])

        # 截断因平移产生的无效样本
        x_symbol = x_symbol[:-(self.m - 1) * self.tau]
        return x_symbol


if __name__ == "__main__":
    a = [0.9, 0.99]
    w = [0.06, 1.0]  # 不同的耦合强度
    _, y_series = gen_samples(size=100000, lags=[5, 10], a=a, w=w, show=True)

    # ---- 测试 -------------------------------------------------------------------------------------

    self = SeriesSymbolizer(m=3, tau=10000)

    y_symbol = self.symbolize(y_series)

    # TODO：MHG重编码
    df = pd.DataFrame(np.c_[y_series[:len(y_symbol)], y_symbol], columns=["Symbolized", "Original"])

    # 画图
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(y_series[:2000], label="Original Series")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(y_symbol[:2000], label="Symbolized Series")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # ---- 测试 -------------------------------------------------------------------------------------

    # x = x_series

    # tau = 1000
    # m = 3
    # tau_max = tau

    # x_symbol = symbolize(x, tau, m, tau_max)
    # x_orig = x[(m - 1) * tau_max:]
    # plt.scatter(x_symbol, x_orig, s=1, alpha=0.5)

    # # TODO：MHG编码，使得均值越小的符号化序列对应的点越靠近原点
    # df = pd.DataFrame(np.c_[x_symbol, x_orig], columns=["Symbolized", "Original"])

    # # 计算每组符号化序列的均值
    # df_mean = df.groupby("Symbolized").mean().reset_index()

    # # 重新分配符号
    # df["Symbolized"] = df["Symbolized"].map(df_mean.set_index("Symbolized")["Original"].rank().astype(int))

    # x_symbol = df["Symbolized"].values
    # x_orig = df["Original"].values

    # plt.figure()
    # plt.scatter(x_symbol, x_orig, s=1, alpha=0.5)

    # plt.figure()
    # plt.plot(x[(m - 1) * tau_max:(m - 1) * tau_max+100], label="Original Series")
    # plt.plot(x_symbol[:100], label="Symbolized Series")

    # plt.legend()
    # plt.show()

