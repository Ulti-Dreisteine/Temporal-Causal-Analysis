# -*- coding: utf-8 -*-
"""
Created on 2025/07/22 10:43:08

@File -> s0_0_基于KNN的概率密度估计.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于KNN的概率密度估计
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.knn_prob_est import build_tree, cal_knn_prob_dens
from core.knn_prob_est import query_neighbors_dist, get_unit_ball_volume

if __name__ == "__main__":
    # 生成正太分布样本
    X = np.random.normal(loc=0, scale=1, size=(90,))

    # 选择第一个样本点重复10次，加入样本数据里
    X_rep = np.repeat(X[0], 10)
    X = np.concatenate([X, X_rep])

    plt.figure(figsize=(5, 5))
    plt.plot(X)
    plt.xlabel("索引")
    plt.ylabel("数值")
    plt.tight_layout()
    plt.show()

    # ---- 构建树 -----------------------------------------------------------------------------------

    tree = build_tree(x=X, metric="euclidean")

    # ---- 计算概率密度 ------------------------------------------------------------------------------

    prob_dens = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        prob_dens[i] = cal_knn_prob_dens(X[i], tree=tree, k=5)

    # 去重
    arr = np.concatenate([X.reshape(len(X), -1), prob_dens.reshape(-1, 1)], axis=1)
    arr = np.unique(arr, axis=0)

    X, probs = arr[:, :-1], arr[:, -1]
    probs /= np.sum(probs)  # 归一化为概率分布
