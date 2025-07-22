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


class MarkovChainIIDResampler(object):
    """马尔可夫链IID重采样器"""

    def __init__(self):
        pass

    def set_samples(self, *x) -> None:
        """
        初始化，x为可变参数
        """
        # 将所有x合并为一个数组
        self.arr = np.column_stack([np.ravel(item) for item in x])
        self.arr = np.atleast_2d(self.arr)
        self.N, self.D = self.arr.shape

    def resample(self, N: int = None, k4rep_warn: int = 1):
        """
        重采样

        Params:
        -------
        N: 重采样的样本数量，默认为None，表示与原样本数量相同
        """
        if N is None:
            N = self.N
        
        # 使用直接重采样方法
        idxs = np.random.choice(self.N, size=N, replace=True)

        # 统计重复索引占比
        _, counts = np.unique(idxs, return_counts=True)
        rep_ratio = sum(counts[counts > k4rep_warn]) / N

        if not hasattr(self, "_repeat_warned"):
            if rep_ratio > 0.0:
                print(f"警告: 重复索引比例过高为{rep_ratio * 100:.2f}% ，可能影响概率密度估计的准确性")
                self._repeat_warned = True
        
        # 采样
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

    method = "direct"
    self = MarkovChainIIDResampler()
    self.set_samples(X_series, Y_series, Z_series)

    # ---- 重采样 -----------------------------------------------------------------------------------

    N = 100
    k = 2

    X_resampled, Y_resampled, Z_resampled = self.resample(N=N, k4rep_warn=k)

    # ---- 散点图对比 --------------------------------------------------------------------------------

    plt.figure(figsize=(5, 5))
    plt.scatter(Z_resampled, Y_resampled, alpha=0.7, s=20, c="r", label="Resampled", edgecolors='none')
    plt.scatter(Z_series, Y_series, alpha=0.3, s=10, c="b", label="Original", edgecolors='none')
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.legend()
    plt.tight_layout()
    plt.show()

