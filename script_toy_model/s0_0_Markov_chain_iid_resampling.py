# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 16:58:28

@File -> s0_0_Markov_chain_iid_resampling.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 从Markov链中进行独立同分布重采样
"""

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from script_toy_model.util import gen_samples
from core.knn_prob_est import build_tree, cal_knn_prob_dens


class MarkovChainIIDResampler(object):
    """马尔可夫链IID重采样器"""

    def __init__(self, *x, metric: str = "euclidean") -> None:
        """
        初始化，x为可变参数
        """
        self.metric = metric

        # 将所有x合并为一个数组
        self.arr = np.column_stack([np.ravel(item) for item in x])

        # 归一化
        self.scaler = MinMaxScaler()
        self.arr_norm = self.scaler.fit_transform(self.arr)

        # 计算概率密度
        self.__cal_prob_density()

    def __cal_prob_density(self):
        """
        计算概率密度
        """
        tree = build_tree(self.arr_norm, metric=self.metric)

        prob_dens = []
        for i in range(len(self.arr_norm)):
            prob_dens.append(cal_knn_prob_dens(self.arr_norm[i], tree=tree, k=5))

        prob_dens = np.array(prob_dens)
        prob_dens /= np.sum(prob_dens)  # 归一化为概率分布

        self.prob_dens = prob_dens

    def resample(self, N: int = None):
        if N is None:
            N = len(self.arr)

        idxs = np.random.choice(len(self.arr), size=N, replace=True, p=self.prob_dens)
        arr_resampled = self.arr[idxs]

        # 可变解析
        if arr_resampled.ndim == 1:
            return arr_resampled
        else:
            return [arr_resampled[:, i] for i in range(arr_resampled.shape[1])]

if __name__ == "__main__":

    # ---- 生成样本 ---------------------------------------------------------------------------------

    tau = 0
    X_series, Y_series = gen_samples(tau=tau, N=1000, show=True)

    X_series = X_series[0:]
    Y_series = Y_series[0:]
    Z_series = X_series.copy()

    plt.scatter(X_series, Y_series, alpha=0.1, s=6)

    # ---- 测试 -------------------------------------------------------------------------------------

    self = MarkovChainIIDResampler(X_series, Y_series, Z_series, metric="euclidean")
    X_resampled, Y_resampled, Z_resampled = self.resample(N=200)

    # ---- 散点图对比 --------------------------------------------------------------------------------

    plt.figure(figsize=(5, 5))
    plt.scatter(Z_resampled, Y_resampled, alpha=0.7, s=20, c="r", label="Resampled", edgecolors='none')
    plt.scatter(Z_series, Y_series, alpha=0.3, s=10, c="b", label="Original", edgecolors='none')
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.legend()
    plt.tight_layout()
    plt.show()

