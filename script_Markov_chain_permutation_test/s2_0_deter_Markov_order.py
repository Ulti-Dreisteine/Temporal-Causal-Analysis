# -*- coding: utf-8 -*-
"""
Created on 2025/07/18 16:20:54

@File -> s2_0_deter_Markov_order.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Markov链的定阶
"""

from collections import defaultdict
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


class MarkovChainGenerator(object):
    """
    马尔可夫链生成器
    """

    def __init__(self, Pi: dict) -> None:
        self.Pi = Pi

    def gen_a_trial(self, N_steps: int):
        X_series = []
        # 初始化状态
        x_t_k = 0
        x_t_k1 = 0
        for _ in range(N_steps):
            # 根据转移概率生成下一个状态
            x_new = np.random.choice([0, 1, 2], p=self.Pi[(x_t_k1, x_t_k)])
            X_series.append(x_new)
            
            # 更新状态
            x_t_k, x_t_k1 = x_new, x_t_k
        return X_series
    

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
    

def cal_block_entropy(X_series: list, k: int) -> float:
    """
    计算k阶马尔可夫链的块熵
    """
    counts = {}
    for i in range(len(X_series) - k):
        key = tuple(X_series[i:i + k + 1])
        if key not in counts:
            counts[key] = 0
        counts[key] += 1

    total_count = sum(counts.values())
    block_entropy = -sum((count / total_count) * np.log2(count / total_count) for count in counts.values())
    
    return block_entropy

if __name__ == "__main__":

    # ---- 样本生成 ----------------------------------------------------------------------------------

    Pi = {
        (0, 0): [0.7, 0.2, 0.1],  # P(X_t | X_{t-2}=0, X_{t-1}=0)
        (0, 1): [0.1, 0.6, 0.3],
        (0, 2): [0.2, 0.2, 0.6],
        (1, 0): [0.3, 0.4, 0.3],
        (1, 1): [0.1, 0.8, 0.1],
        (1, 2): [0.0, 0.1, 0.9],
        (2, 0): [0.5, 0.5, 0.0],
        (2, 1): [0.2, 0.3, 0.5],
        (2, 2): [0.1, 0.1, 0.8]
    }

    self = MarkovChainGenerator(Pi)
    
    X_series = self.gen_a_trial(1000)

    plt.plot(X_series, label='Markov Chain Sample', color='blue')

    # ---- 块熵计算 ----------------------------------------------------------------------------------

    k = 0  # 马尔可夫阶数
    block_entropy = cal_block_entropy(X_series, k+1)

    # 计算代理数据的块熵分布
    N_trials = 100

    block_entropy_srg = np.zeros(N_trials)
    for i in range(N_trials):
        X_srg = gen_Markov_surrogate(X_series, order=k)
        block_entropy_srg[i] = cal_block_entropy(X_srg, k+1)

    # ---- 结果展示 ----------------------------------------------------------------------------------

    plt.figure(figsize=(5, 3))
    plt.hist(block_entropy_srg, bins=30, density=True, alpha=0.6, color='g', label='Surrogate Block Entropy')
    plt.axvline(block_entropy, color='r', linestyle='dashed', linewidth=1, label='Original Block Entropy')
    plt.xlabel('Block Entropy')
    plt.ylabel('Density')
    plt.title(f'Block Entropy Distribution (k={k})')
    plt.legend()
    plt.tight_layout()
    plt.show()
