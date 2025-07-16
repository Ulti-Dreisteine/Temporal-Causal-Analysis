# -*- coding: utf-8 -*-
"""
Created on 2025/05/07 13:47:50

@File -> dit.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: dit
"""

import dit.multivariate
import numpy as np
import dit


def cal_mi(X: np.ndarray, Y: np.ndarray) -> float:
    """
    计算两个一维或多维变量之间的互信息
    """

    X = X.reshape(len(X), -1)
    Y = Y.reshape(len(Y), -1)

    # X和Y的值都为整数
    assert np.issubdtype(X.dtype, np.integer), "X的值必须为整数"
    assert np.issubdtype(Y.dtype, np.integer), "Y的值必须为整数"

    assert len(X) == len(Y), "X和Y的长度不一致"
    Dx, Dy = X.shape[1], Y.shape[1]

    # 总体样本
    samples = np.c_[X, Y]

    # 统计不同模式的概率
    modes, counts = np.unique(samples, axis=0, return_counts=True)
    probs = counts / np.sum(counts)

    # 制作成分布字典
    dist_dict = {tuple(modes[i, :]): probs[i] for i in range(modes.shape[0])}

    # 计算互信息
    dist = dit.Distribution(dist_dict)
    # mi = dit.multivariate.coinformation(
    #     dist, 
    #     rvs=[list(range(Dx)), list(range(Dx, Dx + Dy))],
    #     rv_mode="indexes",
    #     )
    mi = dit.multivariate.caekl_mutual_information(dist, [list(range(Dx)), list(range(Dx, Dx + Dy))])

    return mi


def SU(x: np.ndarray, y: np.ndarray) -> float:
    """对称不确定性系数"""
    a = cal_mi(x, y)
    b = cal_mi(x, x) + cal_mi(y, y)
    return 2 * a / b


def cal_cmi(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """
    计算一维或多维变量之间的条件互信息
    """

    X = X.reshape(len(X), -1)
    Y = Y.reshape(len(Y), -1)
    Z = Z.reshape(len(Z), -1)

    # X、Y和Z的值都为整数
    assert np.issubdtype(X.dtype, np.integer), "X的值必须为整数"
    assert np.issubdtype(Y.dtype, np.integer), "Y的值必须为整数"
    assert np.issubdtype(Z.dtype, np.integer), "Z的值必须为整数"

    assert len(X) == len(Y) == len(Z), "X、Y和Z的长度不一致"

    Dx, Dy, Dz = X.shape[1], Y.shape[1], Z.shape[1]

    # 总体样本
    samples = np.c_[X, Y, Z]

    # 统计不同模式的概率
    modes, counts = np.unique(samples, axis=0, return_counts=True)
    probs = counts / np.sum(counts)

    # 制作成分布字典
    dist_dict = {tuple(modes[i, :]): probs[i] for i in range(modes.shape[0])}

    # 计算条件互信息
    dist = dit.Distribution(dist_dict)
    cmi = dit.multivariate.coinformation(
        dist, 
        rvs=[list(range(Dx)), list(range(Dx, Dx + Dy))], 
        crvs=[list(range(Dx + Dy, Dx + Dy + Dz))],
        rv_mode="indexes",
        )
    
    return cmi


if __name__ == "__main__":
    # 生成两个随机变量的样本
    X = np.random.randint(0, 2, size=(1000, 2))
    Y = np.random.randint(0, 2, size=(1000, 2))
    Z = np.random.randint(0, 2, size=(1000, 2))

    mi = cal_mi(X, Y)
    su = SU(X, Y)
    cmi = cal_cmi(X, Y, Z)