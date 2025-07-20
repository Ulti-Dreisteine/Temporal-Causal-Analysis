# -*- coding: utf-8 -*-
"""
Created on 2025/07/16 15:43:23

@File -> s0_main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 主程序
"""

from collections import defaultdict
from typing import Callable
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
# from core.cit_entropy import kraskov_mi
from core.dit_entropy import cal_mi


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

class MarkovChainGenerator(object):
    """
    马尔可夫链生成器。用于根据给定的转移概率矩阵，生成一对状态序列（X和Y），每个序列独立地按照马尔可夫过程演化。
    """

    def __init__(self, Pi: np.ndarray):
        """
        初始化

        Params:
        -------
        Pi: 转移概率矩阵
        """
        self.Pi = Pi
        self.Dx = Pi.shape[0]  # 状态空间的大小

    def gen_a_trial(self, N_steps: int):
        """
        进行一次试验
        """
        X_series = np.zeros(N_steps, dtype=int)
        Y_series = np.zeros(N_steps, dtype=int)

        for i in range(N_steps):
            X_series[i] = np.random.choice(np.arange(self.Dx), p=self.Pi[X_series[i - 1]] if i > 0 else self.Pi[np.random.randint(self.Dx)])
            Y_series[i] = np.random.choice(np.arange(self.Dx), p=self.Pi[Y_series[i - 1]] if i > 0 else self.Pi[np.random.randint(self.Dx)])

        return X_series, Y_series
    
    def exec_multi_trials(self, N_trials: int, N_steps: int):
        """
        执行多次试验

        Params:
        -------
        N_trials: 试验次数
        N_steps: 每次试验的递推步数
        """
        X_samples = np.zeros((N_trials, N_steps), dtype=int)
        Y_samples = np.zeros((N_trials, N_steps), dtype=int)

        for i in range(N_trials):
            X_samples[i], Y_samples[i] = self.gen_a_trial(N_steps)

        return X_samples, Y_samples
    
    def show(self, X_samples: np.ndarray, Y_samples: np.ndarray):
        """
        显示生成的样本序列
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 4))
        plt.subplot(2, 1, 1)
        plt.plot(X_samples[0], label="$X$", color="k", linewidth=1.0)
        plt.ylabel("state")
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.plot(Y_samples[0], label="$Y$", color="k", linewidth=1.0)
        plt.xlabel("step")
        plt.ylabel("state")
        plt.legend(loc="upper right")

        plt.tight_layout()


def exec_bootstrap_test(x: np.ndarray, y: np.ndarray, N_perm: int, stat_func: Callable):
    """
    执行排列置换检验
    """
    # 计算原始统计量
    stat_orig = stat_func(x, y)

    # 生成所有bootstrap索引
    idxs = np.random.randint(0, len(x), size=(N_perm, len(x)))
    x_bootstrap = x[idxs]
    y_bootstrap = y[idxs]

    # 计算所有bootstrap统计量（向量化）
    stat_bootstrap = np.array([stat_func(xb, yb) for xb, yb in zip(x_bootstrap, y_bootstrap)])

    # 计算p值
    p_value = np.mean(stat_bootstrap >= stat_orig)

    return stat_orig, stat_bootstrap, p_value



# **************************************************************************************************
# 项目工具
# **************************************************************************************************

def gen_Pi() -> np.ndarray:
    """
    转移概率矩阵的意义：每个当前状态开始，有一半的概率停留在当前状态，25%的概率转移到前一个状态，25%的概率转移到后一个状态。
    """
    Pi = np.array([
        [1 / 6] * 6,
        [1 / 6] * 6,
        [1 / 6] * 6,
        [1 / 6] * 6,
        [1 / 6] * 6,
        [1 / 6] * 6,
    ])
    return Pi
    

if __name__ == "__main__":

    # ---- 设置转移概率矩阵 ---------------------------------------------------------------------------

    Pi = gen_Pi()

    # ---- 样本生成 ----------------------------------------------------------------------------------

    self = MarkovChainGenerator(Pi)

    N_trials = 100
    N_steps = 100
    X_samples, Y_samples = self.exec_multi_trials(N_trials, N_steps)

    # 画图
    self.show(X_samples, Y_samples)
    plt.savefig("runtime/order_0.png", dpi=600)

    # ---- 单组独立性检验 ----------------------------------------------------------------------------

    i = 0
    Xi = X_samples[i]
    Yi = Y_samples[i]

    mi_i = cal_mi(Xi, Yi)

    # 置换检验
    N_perm = 1000
    
    stat_orig, stat_bootstrap, p_value = exec_bootstrap_test(Xi, Yi, N_perm, cal_mi)

    # 画图
    plt.figure(figsize=(5, 3))
    plt.hist(stat_bootstrap, bins=50, alpha=0.7, label="MI Permuted", color="red", density=True)
    plt.axvline(stat_orig, color="blue", linestyle="dashed", linewidth=1.5, label="MI Original")
    plt.xlabel("MI")
    plt.ylabel("Prob. Density")
    plt.title(f"Bootstrap Test: p-value = {p_value:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"runtime/order_0_bootstrap_test_{i}.png", dpi=600)

    # ---- 多组独立性检验 ----------------------------------------------------------------------------

    N_perm = 100
    p_values = np.zeros(N_trials)

    for i in range(N_trials):
        print(f"\rTrial {i}", end="")
        sys.stdout.flush()
        Xi = X_samples[i]
        Yi = Y_samples[i]

        mi_i = cal_mi(Xi, Yi)

        # 置换检验
        stat_orig, stat_bootstrap, p_value = exec_bootstrap_test(Xi, Yi, N_perm, cal_mi)

        p_values[i] = p_value

    # 统计I类错误率
    alpha = 0.01
    type_I_error_rate = np.sum(p_values < alpha) / N_trials

    print(f"\nType I Error Rate: {type_I_error_rate:.4f}")



