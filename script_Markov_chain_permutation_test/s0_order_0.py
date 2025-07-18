# -*- coding: utf-8 -*-
"""
Created on 2025/07/16 15:43:23

@File -> s0_main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 主程序
"""

from collections import defaultdict
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


def gen_Markov_surrogate(series: np.ndarray, order=1):
    """
    生成马尔可夫代理序列。根据给定的时间序列，生成一个新的序列，使其尽可能遵循与原序列相同的马尔可夫性质。
    """
    # 统计转移频率
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(series) - order):
        state = tuple(series[i : i + order])
        next_val = series[i + order]
        counts[state][next_val] += 1

    # 生成代理序列
    surrogate = list(series[:order])
    for _ in range(len(series) - order):
        state = tuple(surrogate[-order:])
        try:
            next_vals, counts_ = zip(*counts[state].items())
            probs = np.array(counts_) / sum(counts_)
            surrogate.append(np.random.choice(next_vals, p=probs))
        except:
            # 如果当前状态没有下一个值的统计信息，则随机选择
            surrogate.append(np.random.choice(series))

    surrogate = np.array(surrogate)
    return surrogate
    

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

    # ---- 设置转移概率矩阵 -------------------------------------------------------------------------

    Pi = gen_Pi()

    # ---- 样本生成 ---------------------------------------------------------------------------------

    self = MarkovChainGenerator(Pi)

    N_trials = 1000
    N_steps = 200
    X_samples, Y_samples = self.exec_multi_trials(N_trials, N_steps)

    # 画图
    self.show(X_samples, Y_samples)
    plt.savefig("runtime/samples.png", dpi=600)

    # # ---- 独立检验 -------------------------------------------------------------------------------

    # # 真实的背景互信息分布
    # mi_true = np.zeros(N_trials)
    # for i in range(N_trials):
    #     mi_true[i] = cal_mi(X_samples[i], Y_samples[i])

    # # 基于置换的互信息分布
    # mi_perm_dict = defaultdict(list)

    # for i in range(N_trials):
    #     X_perm_rand = np.random.permutation(X_samples[i])
    #     mi_perm_dict["rand"].append(cal_mi(X_perm_rand, Y_samples[i]))
    
    # for i in range(N_trials):
    #     X_perm_Markov = gen_Markov_surrogate(X_samples[8], order=1)
    #     mi_perm_dict["Markov"].append(cal_mi(X_perm_Markov, Y_samples[i]))

    # # ---- 绘图 -------------------------------------------------------------------------------------

    # bins = 50
    # range_ = (-0.3, 0.4)
    # density = True
    # histtype = "step"
    # linewidth = 1.5

    # plt.figure(figsize=(5, 3))
    # plt.hist(
    #     mi_true, 
    #     bins=bins, 
    #     alpha=1, 
    #     range=range_, 
    #     label="True MI", 
    #     color="blue", 
    #     density=density, 
    #     histtype=histtype, 
    #     linewidth=linewidth)
    # plt.hist(
    #     mi_perm_dict["rand"],
    #     bins=bins, 
    #     alpha=1, 
    #     range=range_, 
    #     label="Permuted MI (Random)", 
    #     color="red", 
    #     density=density, 
    #     histtype=histtype, 
    #     linewidth=linewidth,
    #     linestyle="dashed"
    # )
    # plt.hist(
    #     mi_perm_dict["Markov"],
    #     bins=bins, 
    #     alpha=1, 
    #     range=range_, 
    #     label="Permuted MI (Markov)", 
    #     color="green", 
    #     density=density,
    #     histtype=histtype, 
    #     linewidth=linewidth,
    #     linestyle="dashed"
    # )
    # plt.xlabel("Mutual Information")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()



