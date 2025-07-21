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

if __name__ == "__main__":

    # ---- 生成样本 ---------------------------------------------------------------------------------

    tau = 0
    X_series, Y_series = gen_samples(tau=tau, N=10000, show=True)

    X_series = X_series[0:]
    Y_series = Y_series[0:]

    plt.scatter(X_series, Y_series, alpha=0.1, s=6)

    # ---- 构建样本 ---------------------------------------------------------------------------------

    arr = np.column_stack((X_series, Y_series))

    # 归一化
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(arr)

    # ---- 计算概率密度 ------------------------------------------------------------------------------

    tree = build_tree(arr, metric="euclidean")
    prob_dens = []
    for i in range(len(arr)):
        prob_dens.append(cal_knn_prob_dens(arr[i], tree=tree, k=5))

    prob_dens = np.array(prob_dens)
    prob_dens /= np.sum(prob_dens)  # 归一化为概率分布

    # ---- 按照概率密度进行重采样 ----------------------------------------------------------------------

    idxs = np.arange(len(arr))
    idxs_resample = np.random.choice(idxs, size=1000, replace=True, p=prob_dens)

    arr_resample = arr[idxs_resample]

    # 反归一化
    arr_resample = scaler.inverse_transform(arr_resample)

    # ---- 画图 -------------------------------------------------------------------------------------

    X_resample, Y_resample = arr_resample[:, 0], arr_resample[:, 1]

    plt.figure(figsize=(5, 5))

    plt.subplot(2, 1, 1)
    plt.plot(X_resample, "k", linewidth=1.0, label="$X$")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(Y_resample, "k", linewidth=1.0, label="$Y$")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ---- 散点图对比 --------------------------------------------------------------------------------

    plt.figure(figsize=(5, 5))
    plt.scatter(X_resample, Y_resample, alpha=0.3, s=20, label="Resampled", edgecolors='none')
    plt.scatter(X_series, Y_series, alpha=0.1, s=3, label="Original", edgecolors='none')
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.legend()
    plt.tight_layout()
    plt.show()

