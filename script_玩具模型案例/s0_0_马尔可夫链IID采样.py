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
from script_玩具模型案例.util import gen_samples
from core.knn_prob_est import build_tree, cal_knn_prob_dens


class MarkovChainIIDResampler(object):
    """马尔可夫链IID重采样器"""

    def __init__(self, metric: str = "euclidean"):
        self.metric = metric

    def set_samples(self, *x) -> None:
        """
        初始化，x为可变参数
        """
        # 将所有x合并为一个数组
        self.arr = np.column_stack([np.ravel(item) for item in x])

        # 归一化
        self.scaler = MinMaxScaler()
        self.arr_norm = self.scaler.fit_transform(self.arr)

    def __cal_prob_density(self):
        """
        计算概率密度
        """
        tree = build_tree(self.arr_norm, metric=self.metric)

        # <<----------------------------------------------------------------------------------------
        prob_dens = []
        for i in range(len(self.arr_norm)):
            prob_dens.append(cal_knn_prob_dens(self.arr_norm[i], tree=tree, k=5))

        prob_dens = np.array(prob_dens)
        prob_dens /= np.sum(prob_dens)  # 归一化为概率分布

        self.prob_dens = prob_dens
        # ------------------------------------------------------------------------------------------
        # N = len(self.arr_norm)

        # prob_dens = np.zeros(N)
        # for i in range(N):
        #     prob_dens[i] = cal_knn_prob_dens(self.arr_norm[i], tree=tree, k=5)

        # # 去重
        # arr = np.concatenate([self.arr_norm, prob_dens.reshape(-1, 1)], axis=1)
        # # 返回去重后的原始索引
        # _, unique_idxs = np.unique(arr, axis=0, return_index=True)

        # _, probs = arr[:, :-1], arr[:, -1]
        # probs /= np.sum(probs)  # 归一化为概率分布

        # self.prob_dens = probs[unique_idxs]  # 只保留去重后的概率密度
        # >>----------------------------------------------------------------------------------------

    def resample(self, N: int = None, method: str = None):
        """重采样"""
        if N is None:
            N = len(self.arr)
        
        method = method or "direct"

        if method == "pi":
            # 使用平稳分布方法
            self.__cal_prob_density()
            idxs = np.random.choice(len(self.arr), size=N, replace=True, p=self.prob_dens)
        elif method == "direct":
            # 使用直接采样方法
            idxs = np.random.choice(len(self.arr), size=N, replace=True)
        else:
            raise ValueError(f"Unsupported resampling method: {method}")

        arr_resampled = self.arr[idxs]

        # 检查重复索引比例
        repeat_ratio = 1 - np.unique(idxs).size / len(idxs)

        if not hasattr(self, "_repeat_warned"):
            if repeat_ratio > 0.1:
                print(f"警告: 重复索引比例过高为{repeat_ratio * 100:.2f}% ，可能影响概率密度估计的准确性")
                self._repeat_warned = True

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

    metric = "chebyshev"
    method = "direct"
    self = MarkovChainIIDResampler(X_series, Y_series, Z_series, metric=metric)
    X_resampled, Y_resampled, Z_resampled = self.resample(N=200, method=method)

    # ---- 散点图对比 --------------------------------------------------------------------------------

    plt.figure(figsize=(5, 5))
    plt.scatter(Z_resampled, Y_resampled, alpha=0.7, s=20, c="r", label="Resampled", edgecolors='none')
    plt.scatter(Z_series, Y_series, alpha=0.3, s=10, c="b", label="Original", edgecolors='none')
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.legend()
    plt.tight_layout()
    plt.show()

