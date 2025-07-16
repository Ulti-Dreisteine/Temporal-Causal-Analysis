# -*- coding: utf-8 -*-
"""
Created on 2025/07/16 13:37:26

@File -> s0_main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 掷骰子模型
"""

from collections import defaultdict
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.cit_entropy import kraskov_mi


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

    def gen_a_trial(self, N_steps: int):
        """
        进行一次试验
        """
        X_series = np.zeros(N_steps, dtype=int)
        Y_series = np.zeros(N_steps, dtype=int)

        for i in range(N_steps):
            X_series[i] = np.random.choice(np.arange(6), p=self.Pi[X_series[i - 1]] if i > 0 else self.Pi[np.random.randint(0, 6)])
            Y_series[i] = np.random.choice(np.arange(6), p=self.Pi[Y_series[i - 1]] if i > 0 else self.Pi[np.random.randint(0, 6)])

        return X_series, Y_series
    
    def exec_multi_trials(self, N_trials: int, N_steps: int):
        """
        执行多次试验

        Params:
        -------
        N_trials: 试验次数
        N_steps: 每次试验的步数
        """
        X_samples = np.zeros((N_trials, N_steps), dtype=int)
        Y_samples = np.zeros((N_trials, N_steps), dtype=int)

        for i in range(N_trials):
            X_samples[i], Y_samples[i] = self.gen_a_trial(N_steps)

        return X_samples, Y_samples
    

def gen_Markov_surrogate(series: np.ndarray, order=1):
    # 统计转移频率
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(series) - order):
        state = tuple(series[i:i+order])
        next_val = series[i+order]
        counts[state][next_val] += 1

    # 生成代理序列
    surrogate = series[:order].copy()
    for i in range(len(series) - order):
        current_state = tuple(surrogate[-order:])
        next_options = list(counts[current_state].keys())
        probs = np.array(list(counts[current_state].values())) / sum(counts[current_state].values())
        next_val = np.random.choice(next_options, p=probs)
        surrogate = np.append(surrogate, next_val)

    return surrogate
        


if __name__ == "__main__":

    # ---- 设置转移概率矩阵 ---------------------------------------------------------------------------

    """
    转移概率矩阵的意义：每个当前状态开始，有一半的概率停留在当前状态，25%的概率转移到前一个状态，25%的概率转移到后一个状态。
    """

    # 马尔可夫性
    Pi = np.array([
        [0.5, 0.25, 0, 0, 0, 0.25],
        [0.25, 0.5, 0.25, 0, 0, 0],
        [0, 0.25, 0.5, 0.25, 0, 0],
        [0, 0, 0.25, 0.5, 0.25, 0],
        [0, 0, 0, 0.25, 0.5, 0.25],
        [0.25, 0, 0, 0, 0.25, 0.5],
    ])

    # 非马尔可夫性
    # Pi = np.array([
    #     [1 / 6] * 6,
    #     [1 / 6] * 6,
    #     [1 / 6] * 6,
    #     [1 / 6] * 6,
    #     [1 / 6] * 6,
    #     [1 / 6] * 6,
    # ])

    # ---- 样本生成 ---------------------------------------------------------------------------------

    self = MarkovChainGenerator(Pi)

    N_trials = 1000
    N_steps = 200
    X_samples, Y_samples = self.exec_multi_trials(N_trials, N_steps)

    # ---- 独立检验 ----------------------------------------------------------------------------------

    # 计算真实的背景互信息
    mi_true = np.zeros(N_trials)
    for i in range(N_trials):
        mi_true[i] = kraskov_mi(X_samples[i], Y_samples[i])

    # 计算基于置换的互信息
    mi_permuted = np.zeros(N_trials)
    for i in range(N_trials):
        # 对X样本进行置换
        X_permuted = gen_Markov_surrogate(X_samples[i])

        # 计算置换后的互信息
        mi_permuted[i] = kraskov_mi(X_permuted, Y_samples[i])

    # ---- 绘图 -------------------------------------------------------------------------------------

    plt.figure(figsize=(5, 5))
    plt.hist(mi_true, bins=30, alpha=1, range=(-0.3, 0.4), label="True MI", color="blue", density=True, histtype="step", linewidth=1.5)
    plt.hist(mi_permuted, bins=30, alpha=1, range=(-0.3, 0.4), label="Permuted MI", color="red", density=True, histtype="step", linewidth=1.5)
    plt.xlabel("Mutual Information")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # ---- 绘图 -------------------------------------------------------------------------------------

    # plt.figure(figsize=(5, 5))
    # plt.subplot(2, 1, 1)
    # plt.plot(X_samples, label="X samples", color="blue", linewidth=1.0)
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(Y_samples, label="Y samples", color="red", linewidth=1.0)
    # plt.xlabel("Trial")
    # plt.legend()
    
    # # ---- 互信息独立性检验 ---------------------------------------------------------------------------

    # bt_rounds = 100

    # bt_values = np.zeros(bt_rounds)
    # bg_values = np.zeros(bt_rounds)

    # for i in range(bt_rounds):
    #     idxs = np.random.choice(N_trials, size=N_trials, replace=True)
    #     bt_values[i] = kraskov_mi(X_samples[idxs], Y_samples[idxs])
    #     bg_values[i] = kraskov_mi(X_samples[idxs], Y_samples[idxs])

    # plt.figure(figsize=(5, 5))
    # plt.hist(bt_values, bins=30, alpha=0.5, label="Bootstrap MI", color="blue")
    # plt.hist(bg_values, bins=30, alpha=0.5, label="Bagging MI", color="red")
    # plt.xlabel("Mutual Information")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

