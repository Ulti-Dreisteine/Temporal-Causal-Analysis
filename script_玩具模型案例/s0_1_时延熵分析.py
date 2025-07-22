# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 14:01:33

@File -> s0_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: MIT分析
"""

from collections import defaultdict
from typing import List, Union
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import show_results
from core.cit_entropy import cal_cmi
from script_玩具模型案例.util import gen_samples
from script_玩具模型案例.s0_0_马尔可夫链IID采样 import MarkovChainIIDResampler

# KSG互信息估计参数
K = 5
ALPHA = 0.0


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

class TemporalCausalEntropyAnalysis(object):
    """基于瞬时信息传输（Momentary Information Transfer, MIT）识别变量之间的时延"""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()

        """
        注意:
        - 将tau_x设置为1，可提升MIT检测准确性，屏蔽掉更早的X信息
        - 将tau_y设置为1，可提升时延检测准确性，屏蔽掉更早的Y信息，且无信息混淆现象
        """
        
        self.tau_x = 1
        self.tau_y = 1

        assert len(self.x) == len(self.y), f"len(x) = {len(self.x)} != len(y) = {len(self.y)}"
        self.N = len(self.x)

    def __build_lagged_cutoff_samples(self, lag: int):
        """ 构建时延样本并截断"""
        # 构造对应的时延样本
        x_t_lag = np.roll(self.x.copy(), lag)  # 对应于 x_{t-lag} 样本
        x_t_lag_tau_x = np.roll(self.x.copy(), lag + self.tau_x)  # 对应于 x_{t-tau_x} 样本
        y_t = self.y.copy()  # 对应于 y_t 样本
        y_t_tau_y = np.roll(self.y.copy(), self.tau_y)  # 对应于 y_{t-tau_y} 样本

        # 计算截断区间索引
        lags = [0, lag, lag + self.tau_x, self.tau_y]
        min_lag = min(lags)
        max_lag = max(lags)

        # 截断，避免引入无关样本对计算构成误差
        x_t_lag = x_t_lag[max_lag:self.N + min_lag]
        x_t_lag_tau_x = x_t_lag_tau_x[max_lag:self.N + min_lag]
        y_t = y_t[max_lag:self.N + min_lag]
        y_t_tau_y = y_t_tau_y[max_lag:self.N + min_lag]

        return x_t_lag, x_t_lag_tau_x, y_t, y_t_tau_y
    
    def cal_lagged_MIT(self, lag: int, size_bt: int, rounds_bt: int):
        x_t_lag, x_t_lag_tau_x, y_t, y_t_tau_y = self.__build_lagged_cutoff_samples(lag)

        # 检查样本长度是否满足要求
        assert len(y_t) >= size_bt, f"len(y_t) = {len(y_t)} < size_bt = {size_bt}"

        # 初始化记录数组
        bt_records, bg_records = np.zeros(rounds_bt), np.zeros(rounds_bt)

        sampler = MarkovChainIIDResampler()
        sampler.set_samples(x_t_lag, x_t_lag_tau_x, y_t, y_t_tau_y)
        for round in range(rounds_bt):
            # 重采样
            x_t_lag_bt, x_t_lag_tau_x_bt, y_t_bt, y_t_tau_y_bt = sampler.resample(N=size_bt, k4rep_warn=K)

            # 计算CMI
            z_bt = np.c_[x_t_lag_tau_x_bt, y_t_tau_y_bt]
            cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, z_bt, k=K, alpha=ALPHA)

            # 计算背景CMI
            idxs_bg = np.random.permutation(np.arange(len(y_t_bt)))
            x_t_lag_bg = x_t_lag_bt[idxs_bg]
            x_t_lag_tau_x_bg = x_t_lag_tau_x_bt[idxs_bg]

            z_bg = np.c_[x_t_lag_tau_x_bg, y_t_tau_y_bt]
            cmi_bg = cal_cmi(x_t_lag_bg, y_t_bt, z_bg, k=K, alpha=ALPHA)

            bt_records[round] = cmi_bt
            bg_records[round] = cmi_bg
        
        return bt_records, bg_records
    
    def cal_lagged_TE(self, lag: int, size_bt: int, rounds_bt: int):
        """
        计算时延的传递熵（Transfer Entropy）
        """
        x_t_lag, x_t_lag_tau_x, y_t, y_t_tau_y = self.__build_lagged_cutoff_samples(lag)

        # 检查样本长度是否满足要求
        assert len(y_t) >= size_bt, f"len(y_t) = {len(y_t)} < size_bt = {size_bt}"

        # 初始化记录数组
        bt_records, bg_records = np.zeros(rounds_bt), np.zeros(rounds_bt)

        sampler = MarkovChainIIDResampler()
        sampler.set_samples(x_t_lag, x_t_lag_tau_x, y_t, y_t_tau_y)
        for round in range(rounds_bt):
            # 重采样
            x_t_lag_bt, _, y_t_bt, y_t_tau_y_bt = sampler.resample(N=size_bt, k4rep_warn=K)

            # 计算CMI
            z_bt = np.c_[y_t_tau_y_bt]
            cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, z_bt, k=K, alpha=ALPHA)

            # 计算背景CMI
            idxs_bg = np.random.permutation(np.arange(len(y_t_bt)))
            x_t_lag_bg = x_t_lag_bt[idxs_bg]

            z_bg = np.c_[y_t_tau_y_bt]
            cmi_bg = cal_cmi(x_t_lag_bg, y_t_bt, z_bg, k=K, alpha=ALPHA)

            bt_records[round] = cmi_bt
            bg_records[round] = cmi_bg
        
        return bt_records, bg_records
    
    def exec_td_analysis(self, lags2test: Union[list, np.ndarray], size_bt: int, rounds_bt: int, method: str, show: bool = True, correct_base: bool = True):
        """
        执行时间延迟分析

        Params:
        -------
        lags2test: 待检测的时延列表或数组
        size_bt: 每次重采样的样本大小
        rounds_bt: 重采样的轮数
        method: 使用的方法，支持 "MIT" 或 "TE"
        show: 是否显示结果
        correct_base: 是否对结果进行基准校正
        """
        cmi_lag_records = defaultdict(dict)

        for lag in lags2test:

            if method == "MIT":
                bt_records, bg_records = self.cal_lagged_MIT(lag=lag, size_bt=size_bt, rounds_bt=rounds_bt)
            elif method == "TE":
                bt_records, bg_records = self.cal_lagged_TE(lag=lag, size_bt=size_bt, rounds_bt=rounds_bt)
            else:
                raise ValueError(f"不被支持的方法: {method}")

            cmi_lag_records["bt_records"][lag] = bt_records
            cmi_lag_records["bg_records"][lag] = bg_records

        if correct_base:
            base_value = np.mean([bg.mean() for bg in cmi_lag_records["bg_records"].values()])
            for lag in cmi_lag_records["bt_records"]:
                cmi_lag_records["bt_records"][lag] -= base_value
                cmi_lag_records["bg_records"][lag] -= base_value

        if show:
            show_results(cmi_lag_records)

        return cmi_lag_records


if __name__ == "__main__":
    
    # ---- 生成样本 ---------------------------------------------------------------------------------

    tau = 10
    X_series, Y_series = gen_samples(tau=tau, N=1000, show=True)

    X_series = X_series[0:]
    Y_series = Y_series[0:]

    # ---- 进行测试 ---------------------------------------------------------------------------------

    self = TemporalCausalEntropyAnalysis(X_series, Y_series)

    # ---- 单时延检验 --------------------------------------------------------------------------------

    size_bt = 500
    rounds_bt = 100
    method = "MIT"
    show = True
    
    lags2test = np.arange(-30, 31, 1)

    cmi_lag_records = self.exec_td_analysis(
        lags2test, size_bt=size_bt, rounds_bt=rounds_bt, method=method, show=show, correct_base=True)




