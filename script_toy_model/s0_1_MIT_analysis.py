# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 14:01:33

@File -> s0_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: MIT分析
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from script_toy_model.util import gen_samples


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

class MIT(object):
    """基于瞬时信息传输（Momentary Information Transfer, MIT）识别变量之间的时延"""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()

        """
        注意:
        - 将tau_x设置为1，可提升MIT检测准确性
        - 将tau_y设置为1，可提升时延检测准确性
        """
        
        self.tau_x = 1
        self.tau_y = 1

        assert len(self.x) == len(self.y), f"len(x) = {len(self.x)} != len(y) = {len(self.y)}"
        self.N = len(self.x)


if __name__ == "__main__":
    
    # ---- 生成样本 ---------------------------------------------------------------------------------

    tau = 0
    X_series, Y_series = gen_samples(tau=tau, N=1000, show=True)

    X_series = X_series[0:]
    Y_series = Y_series[0:]

    # ---- 进行测试 ---------------------------------------------------------------------------------

    self = MIT(X_series, Y_series)

    # ---- 单时延检验 --------------------------------------------------------------------------------

    lag = 0

    size_bt = 100
    rounds_bt = 50
    show = True

    # 构造对应的时延样本
    x_t_lag = np.roll(self.x.copy(), lag)  # 对应于 x_{t-lag} 样本
    x_t_lag_tau_x = np.roll(self.x.copy(), lag + self.tau_x)  # 对应于 x_{t-tau_x} 样本
    y_t = self.y.copy()  # 对应于 y_t 样本
    y_t_tau_y = np.roll(self.y.copy(), self.tau_y)  # 对应于 y_{t-tau_y} 样本

    # TODO：截断，避免引入无关样本对TE构成误差
    max_lag = max([lag, lag + self.tau_x, self.tau_y])

