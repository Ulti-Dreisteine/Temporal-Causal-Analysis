# -*- coding: utf-8 -*-
"""
Created on 2025/07/23 15:06:23

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 玩具模型熵分析
"""

from collections import defaultdict
from typing import Literal
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import show_results
from core.cit_entropy import cal_cmi
from script.toy_model.util import gen_samples, cal_tau


# **************************************************************************************************
# 通用工具
# **************************************************************************************************


SUPPORTED_METHODS = Literal["TE", "MIT"]

class CausalEntropyAnalysis(object):
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

    def set_params(self, rounds_bt: int, size_bt: int, method: SUPPORTED_METHODS, **kwargs):
        """
        设置计算参数

        Params:
        -------
        rounds_bt: int,  # 迭代轮数
        size_bt: int,    # 每轮样本大小
        method: str      # 计算方法
        """
        self.rounds_bt = rounds_bt
        self.size_bt = size_bt
        self.method = method

        # Kraskov互信息参数
        self.k = kwargs.get("k", 3)  
        self.alpha = kwargs.get("alpha", 0.0)

    def _build_lagged_cutoff_samples(self, lag: int):
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
    
    def cal_entropy(self, x_t_lag_bt, x_t_lag_tau_x_bt, y_t_bt, y_t_tau_y_bt, **kwargs):

        # TODO：替代样本构造还是有问题
        idxs_bg = np.random.permutation(np.arange(len(y_t_bt)))
        x_t_lag_bg = x_t_lag_bt[idxs_bg]
        x_t_lag_tau_x_bg = x_t_lag_tau_x_bt[idxs_bg]

        if self.method == "TE":
            z_bt = np.c_[y_t_tau_y_bt]
            z_bg = np.c_[y_t_tau_y_bt]
        elif self.method == "MIT":
            z_bt = np.c_[x_t_lag_tau_x_bt, y_t_tau_y_bt]
            z_bg = np.c_[x_t_lag_tau_x_bg, y_t_tau_y_bt]
        else:
            raise ValueError(f"不支持的method: {self.method}")
        
        cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, z_bt, k=self.k, alpha=self.alpha)
        cmi_bg = cal_cmi(x_t_lag_bg, y_t_bt, z_bg, k=self.k, alpha=self.alpha)

        return cmi_bt, cmi_bg
    
    def cal_lagged_entropy(self, lag: int):
        x_t_lag, x_t_lag_tau_x, y_t, y_t_tau_y = self._build_lagged_cutoff_samples(lag)

        # 计算条件互信息值
        sampler = MarkovChainSampler()
        sampler.set_samples(x_t_lag, x_t_lag_tau_x, y_t, y_t_tau_y)

        warns_bt = 0
        bt_records, bg_records = np.zeros(self.rounds_bt), np.zeros(self.rounds_bt)
        for round in range(self.rounds_bt):
            # 重采样
            x_t_lag_bt, x_t_lag_tau_x_bt, y_t_bt, y_t_tau_y_bt = sampler.resample(N=self.size_bt, k_thres=1)
            warns_bt += sampler.warn_num

            # 计算熵指标
            cmi_bt, cmi_bg = self.cal_entropy(
                x_t_lag_bt, x_t_lag_tau_x_bt, y_t_bt, y_t_tau_y_bt, method=self.method)

            # 记录结果
            bt_records[round] = cmi_bt
            bg_records[round] = cmi_bg

        warning_rate = warns_bt / self.rounds_bt

        return bt_records, bg_records, warning_rate


class MarkovChainSampler(object):
    """马尔可夫链采样器"""

    def __init__(self):
        pass

    def set_samples(self, *x) -> None:
        """
        初始化，x为可变参数
        """
        # 将所有x合并为一个数组
        self.arr = np.column_stack([np.ravel(item) for item in x])
        self.arr = np.atleast_2d(self.arr)
        self.N, self.D = self.arr.shape

    def resample(self, N: int = None, k_thres: int = 1):
        """
        重采样

        Params:
        -------
        N: 重采样的样本数量，默认为None，表示与原样本数量相同
        """
        if N is None:
            N = self.N
        
        # 使用直接重采样方法
        idxs = np.random.choice(self.N, size=N, replace=True)

        # 统计重复索引占比，如果超过10%，则记录警告
        _, counts = np.unique(idxs, return_counts=True)
        rep_ratio = sum(counts[counts > k_thres]) / N
        self.warn_num = 1 if rep_ratio > 0.1 else 0
        
        # 采样
        arr_resampled = self.arr[idxs]

        # 可变解析
        if arr_resampled.ndim == 1:
            return arr_resampled
        else:
            return [arr_resampled[:, i] for i in range(arr_resampled.shape[1])]
    

if __name__ == "__main__":

    # ---- 生成样本 ---------------------------------------------------------------------------------

    fig_savepath = "fig/xy_series.png"
    x_series, y_series = gen_samples(taus=[0, 10], N=1000, show=True, fig_savepath=fig_savepath)

    x_series, y_series = y_series, x_series

    # ---- 预计算 -----------------------------------------------------------------------------------

    # taus = np.arange(1, 100, 1)

    # size_bt = 50
    # rounds_bt = 10
    # thres = 1 / np.e
    
    # fig_savepath_x = "fig/tau_x.png"
    # fig_savepath_y = "fig/tau_y.png"
    # tau_x = cal_tau(x_series, taus, size_bt, rounds_bt, thres, show=True, fig_savepath=fig_savepath_x)
    # tau_y = cal_tau(y_series, taus, size_bt, rounds_bt, thres, show=True, fig_savepath=fig_savepath_y)

    # ---- 时延熵分析 --------------------------------------------------------------------------------

    self = CausalEntropyAnalysis(x_series, y_series)

    method = "MIT"
    size_bt = 100
    rounds_bt = 50

    self.set_params(rounds_bt=rounds_bt, size_bt=size_bt, method=method, k=5)

    # ---- 计算时延熵 --------------------------------------------------------------------------------

    lags2test = np.arange(-30, 31, 1)
    correct_base = True
    show = True

    cmi_lag_records = defaultdict(dict)
    for lag in lags2test:
        print(f"\rProcessing lag: {lag}", end="")
        sys.stdout.flush()

        bt_records, bg_records, warning_rate = self.cal_lagged_entropy(lag=lag)

        cmi_lag_records["bt_records"][lag] = bt_records
        cmi_lag_records["bg_records"][lag] = bg_records
        cmi_lag_records["warning_rate"][lag] = warning_rate

    if correct_base:
        base_value = np.mean([bg.mean() for bg in cmi_lag_records["bg_records"].values()])
        for lag in cmi_lag_records["bt_records"]:
            cmi_lag_records["bt_records"][lag] -= base_value
            cmi_lag_records["bg_records"][lag] -= base_value

    warning_rates = list(cmi_lag_records["warning_rate"].values())
    avg_warning_rate = np.mean(warning_rates)
    if avg_warning_rate > 0.01:
        print("\n警告: 重采样警告率过高，可能影响结果的可靠性！考虑增加样本量或k值。")

    if show:
        show_results(cmi_lag_records)

    plt.figure(figsize=(5, 3))
    plt.plot(cmi_lag_records["warning_rate"].values())
    plt.xlabel("Lag")
    plt.ylabel("Warning Rate")
    plt.title("Warning Rate by Lag")
    plt.tight_layout()
    plt.show()



    




