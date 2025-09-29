# -*- coding: utf-8 -*-
"""
Created on 2025/09/29 14:22:19

@File -> s0_gen_samples.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 样本生成
"""

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt


def generate_fourth_order_system(n_steps=1000, random_seed=42):
    """
    生成四阶线性随机差分方程系统的样本数据

    Ref:
    ----
    https://medium.com/causality-in-data-science/hands-on-causal-discovery-with-python-e4fb2488c543
    
    Params:
    -------
    n_steps: 生成的时间步数
    random_seed: 随机种子，用于重现结果
    
    Returns:
    --------
    DataFrame: 包含四个变量时间序列的数据框
    """
    np.random.seed(random_seed)
    
    # 系统参数
    burn_in = 100
    max_lag = 3
    total_steps = n_steps + burn_in + max_lag
    
    # 预分配数组
    X = np.zeros((4, total_steps))
    eta = np.random.normal(0, 1, (4, total_steps))
    
    # 随机初始化
    X[:, :max_lag] = np.random.normal(0, 0.1, (4, max_lag))
    
    # 生成时间序列
    for t in range(max_lag, total_steps):
        X[0, t] = 0.7 * X[0, t-1] - 0.8 * X[1, t-1] + eta[0, t]
        X[1, t] = 0.8 * X[1, t-1] + 0.8 * X[3, t-1] + eta[1, t]
        X[2, t] = 0.5 * X[2, t-1] + 0.5 * X[1, t-2] + 0.6 * X[3, t-3] + eta[2, t]
        X[3, t] = 0.7 * X[3, t-1] + eta[3, t]
    
    # 提取稳定后的数据
    start_idx = burn_in + max_lag
    samples_df = pd.DataFrame({
        "x0": X[0, start_idx:start_idx + n_steps],
        "x1": X[1, start_idx:start_idx + n_steps],
        "x2": X[2, start_idx:start_idx + n_steps],
        "x3": X[3, start_idx:start_idx + n_steps],
        "t": range(n_steps)
    })
    
    return samples_df


if __name__ == "__main__":
    samples_df = generate_fourth_order_system(n_steps=2000, random_seed=42)

    if not os.path.exists("runtime"):
        os.makedirs("runtime")

    # 保存数据
    samples_df.to_csv("runtime/s0_样本数据.csv", index=False)

    # 画图
    plt.figure(figsize=(6, 3))

    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(samples_df["t"], samples_df[f"x{i}"], "k", linewidth=0.75, label=f"X{i}")
        plt.ylabel(f"$X_{i}$")
        
        # 去掉边框
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        
        if i < 3:  # 上面三张图只保留纵轴和刻度
            ax.spines["left"].set_visible(True)
            ax.tick_params(bottom=False, labelbottom=False)
        else:  # 最底下一张图保留横纵轴和刻度
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)

    plt.xlabel("$t$")
    plt.savefig("runtime/s0_样本数据.png", dpi=450, bbox_inches="tight")