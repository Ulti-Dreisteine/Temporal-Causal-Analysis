# -*- coding: utf-8 -*-
"""
Created on 2025/07/20 16:03:20

@File -> s2_2阶投掷样本定阶.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 二阶投掷样本定阶
"""

from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def cal_block_entropy(X_series: list, k: int) -> float:
    """
    计算k阶马尔可夫链的块熵
    """
    X_series = X_series.copy()

    counts = {}
    for i in range(len(X_series) - k):
        key = tuple(X_series[i:i + k + 1])

        if key not in counts:
            counts[key] = 0

        counts[key] += 1

    total_count = len(X_series) - k
    block_entropy = -sum((count / total_count) * np.log2(count / total_count) for count in counts.values())
    
    return block_entropy

    # counts = defaultdict(int)
    # for i in range(len(X_series)-k+1):
    #     counts[tuple(X_series[i:i+k])] += 1
    # probs = np.array(list(counts.values()))/sum(counts.values())
    # return -np.sum(probs * np.log2(probs))  # type: ignore


# def cal_block_entropy(data, k):
#     """
#     Calculates the block entropy of a sequence.

#     Args:
#         data: The input sequence (list or NumPy array).
#         k: The block size.

#     Returns:
#         The block entropy value.
#     """
#     counts = {}
#     n = len(data)
#     for i in range(n - k + 1):
#         block = tuple(data[i:i+k])
#         counts[block] = counts.get(block, 0) + 1

#     probabilities = np.array(list(counts.values())) / (n - k + 1)
#     return entropy(probabilities, base=2)


# Markov链检验（Markov_chain）

def _gen_Markov_surrogate(series: np.ndarray, order: int):
    """
    生成马尔可夫代理序列。根据给定的时间序列，生成一个新的序列，使其尽可能遵循与原序列相同的马尔可夫性质。
    """
    series = series.copy()

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


if __name__ == "__main__":

    # ---- 载入样本 ---------------------------------------------------------------------------------

    X_samples = np.load("runtime/2阶投掷_X_samples.npy")[0, :30000]

    # ---- 计算块熵 ---------------------------------------------------------------------------------

    k2test = range(10)
    N_k = len(k2test)

    delta_H_kp1_k = []
    for k in k2test:
        X_series = X_samples[:]  # 取一组样本
        H_k = cal_block_entropy(X_series, k)
        H_kp1 = cal_block_entropy(X_series, k + 1)
        delta_H_kp1_k.append(H_kp1 - H_k)

    plt.plot(k2test, delta_H_kp1_k, marker="o", linestyle="-")

    # N_resample = 100

    # Markov_orders_results = defaultdict(dict)

    # for sample_idx in range(X_samples.shape[0]):
    #     print(f"\r正在处理第 {sample_idx + 1} 组样本...", end="")
    #     sys.stdout.flush()

    #     X_series = X_samples[13, :]  # 取一组样本

    #     if sample_idx == 0:
    #         plt.figure(figsize=(5, 6))

    #     for k in k2test:
    #         H_kp1_surrog = np.zeros(N_resample)

    #         for i in range(N_resample):
    #             X_srg = _gen_Markov_surrogate(X_series, k)
    #             H_kp1_surrog[i] = cal_block_entropy(X_srg, k+1)

    #         H_kp1 = cal_block_entropy(X_series, k + 1)

    #         # 计算p值：注意方向
    #         p_value = np.sum(H_kp1_surrog <= H_kp1) / N_resample

    #         alpha = 0.05
    #         if (p_value > alpha) & (sample_idx not in Markov_orders_results):
    #             Markov_orders_results[sample_idx] = k

    #         # 画图
    #         if sample_idx == 0:
    #             plt.subplot(N_k // 2, 2, k + 1)
    #             plt.hist(
    #                 H_kp1_surrog,
    #                 bins=50,
    #                 alpha=0.7, 
    #                 label=f"{k+1}阶块熵零假设分布", 
    #                 color="blue", 
    #                 density=True, 
    #                 histtype="step",
    #                 linewidth=1.5,
    #                 linestyle="-",
    #             )
    #             plt.axvline(H_kp1, color="red", linestyle="dashed", linewidth=1.5, label=f"{k+1}阶块熵观测值")
    #             plt.title(f"p值: {p_value:.4f}")
    #             plt.legend()
    #             plt.xlabel("块熵值")
    #             plt.ylabel("概率密度")

    #     if sample_idx == 0:
    #         plt.tight_layout()
    #         plt.show()
    #         plt.pause(1)

    #     if sample_idx == 10:
    #         break