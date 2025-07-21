# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 14:01:33

@File -> s0_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 分析
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt

# **************************************************************************************************
# 项目工具
# **************************************************************************************************

def gen_samples(N: int, show: bool):
    """
    生成样本数据

    Params:
    -------
    N: 样本量
    show: 是否显示图形
    """
    # 初始化数组
    X_series = np.ones(N) * np.nan
    Y_series = np.ones(N) * np.nan

    # 稳态段初始
    ss_len = 20
    X_series[:ss_len] = 0
    Y_series[:ss_len] = 0

    a = 0.8
    b = 0.8
    c = 0.6
    tau = 0
    
    assert tau >= 0, "tau必须大于等于0"
    assert tau <= ss_len, "tau不能大于稳态段长度"

    for i in range(ss_len, N):
        X_series[i] = a * X_series[i - 1] + 0.01 * np.random.randn()
        Y_series[i] = b * Y_series[i - 1] + c * X_series[i - tau] + 0.01 * np.random.randn()

    X_series = X_series[ss_len:]
    Y_series = Y_series[ss_len:]
    
    # 绘图
    if show:
        plt.figure(figsize=(5, 5))

        plt.subplot(2, 1, 1)
        plt.plot(X_series, "k", linewidth=1.0, label="$X$")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(Y_series, "k", linewidth=1.0, label="$Y$")
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    return X_series, Y_series

if __name__ == "__main__":
    
    # ---- 生成样本 ---------------------------------------------------------------------------------

    X_series, Y_series = gen_samples(N=100000, show=True)

    X_series = X_series[0:]
    Y_series = Y_series[0:]