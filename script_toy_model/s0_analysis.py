# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 10:35:40

@File -> s0_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 分析
"""

from collections import defaultdict
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from script_toy_model.util import cal_tau
from core.cit_entropy import cal_cmi
# from core.dit_entropy import cal_cmi


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

class TDTEAnalysis(object):
    """基于TE识别变量之间的时延"""

    def __init__(self, x: np.ndarray, y: np.ndarray, tau_x: int, tau_y: int) -> None:
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()
        self.tau_x = int(tau_x)
        self.tau_y = int(tau_y)

        assert len(self.x) == len(self.y), f"len(x) = {len(self.x)} != len(y) = {len(self.y)}"
        self.N = len(self.x)


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

    # ---- 时间常数计算 ------------------------------------------------------------------------------

    recal = False  # 是否重新计算时间常数

    if recal:
        taus = np.arange(1, 100, 1)
        tau_x, _ = cal_tau(X_series, taus, bt_size=100, bt_rounds=100, thres=0.9, show=True)
        tau_y, _ = cal_tau(Y_series, taus, bt_size=100, bt_rounds=100, thres=0.9, show=True)
    else:
        tau_x = 1
        tau_y = 1
    
    # ---- 分析 -------------------------------------------------------------------------------------

    self = TDTEAnalysis(X_series, Y_series, tau_x, tau_y)

    # ---- 测试 -------------------------------------------------------------------------------------

    lags2test = np.arange(-60 * 1, 61 * 1, 1)

    size_bt = 100
    rounds_bt = 50
    show = True

    cmi_lag_records = defaultdict(dict)

    for lag in lags2test:
        print(f"\rProcessing lag = {lag} ...", end="")
        sys.stdout.flush()

        # 构造对应的时延样本
        x_t_lag = np.roll(self.x.copy(), lag)  # 对应于 x_{t-lag} 样本
        x_t_lag_tau_x = np.roll(self.x.copy(), lag + self.tau_x)  # 对应于 x_{t-tau_x} 样本
        y_t = self.y.copy()  # 对应于 y_t 样本
        y_t_tau_y = np.roll(self.y.copy(), self.tau_y)  # 对应于 y_{t-tau_y} 样本

        # 截断，避免引入无关样本对TE构成误差
        if lag > 0:
            x_t_lag = x_t_lag[lag:]
            x_t_lag_tau_x = x_t_lag_tau_x[lag:]
            y_t = y_t[lag:]
            y_t_tau_y = y_t_tau_y[lag:]
        elif lag < 0:
            x_t_lag = x_t_lag[:lag]
            x_t_lag_tau_x = x_t_lag_tau_x[:lag]
            y_t = y_t[:lag]
            y_t_tau_y = y_t_tau_y[:lag]
        else:
            pass

        # ---- 基于Bootstrap的CMI测试 ----------------------------------------------------------------

        N = len(y_t)
        assert N >= size_bt, f"样本长度不足：len(y_t) = {len(y_t)} < size_bt = {size_bt}"

        # 生成rounds_bt组起止索引
        bt_idx_tuples = []
        for _ in range(rounds_bt):
            idx_s = np.random.randint(0, N - size_bt)
            idx_e = idx_s + size_bt
            bt_idx_tuples.append((idx_s, idx_e))

        # 随机打乱
        bg_idx_tuples = bt_idx_tuples.copy()
        np.random.shuffle(bg_idx_tuples)
        # for _ in range(rounds_bt):
        #     idx_s = np.random.randint(0, N - size_bt)
        #     idx_e = idx_s + size_bt
        #     bg_idx_tuples.append((idx_s, idx_e))

        bt_records, bg_records = np.zeros(rounds_bt), np.zeros(rounds_bt)

        # 计算CMI
        k = 3
        alpha = 0.0

        for i, (idx_s, idx_e) in enumerate(bt_idx_tuples):
            x_t_lag_bt = x_t_lag[idx_s:idx_e]
            x_t_lag_tau_x_bt = x_t_lag_tau_x[idx_s:idx_e]
            y_t_bt = y_t[idx_s:idx_e]
            y_t_tau_y_bt = y_t_tau_y[idx_s:idx_e]

            x_t_lag_bg = x_t_lag[bg_idx_tuples[i][0]:bg_idx_tuples[i][1]]
            x_t_lag_tau_x_bg = x_t_lag_tau_x[bg_idx_tuples[i][0]:bg_idx_tuples[i][1]]

            # 计算CMI
            z_bt = np.c_[x_t_lag_tau_x_bt, y_t_tau_y_bt]
            # z_bt = y_t_tau_y_bt
            cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, z_bt, k=k, alpha=alpha)
            bt_records[i] = cmi_bt

            # 计算零假设分布指标
            z_bg = np.c_[x_t_lag_tau_x_bg, y_t_tau_y_bt]
            # z_bg = y_t_tau_y_bt
            cmi_bg = cal_cmi(x_t_lag_bg, y_t_bt, z_bg, k=k, alpha=alpha)
            bg_records[i] = cmi_bg
        
        cmi_lag_records["bt_records"][lag] = bt_records
        cmi_lag_records["bg_records"][lag] = bg_records

    # ---- 画图 -------------------------------------------------------------------------------------

    plt.figure(figsize=(5, 4))
    plt.plot(lags2test, [np.mean(cmi_lag_records["bt_records"][lag]) for lag in lags2test], "b", label="bt_mean")
    plt.plot(lags2test, [np.mean(cmi_lag_records["bg_records"][lag]) for lag in lags2test], "r", label="bg_mean")
    plt.legend()
    plt.axvline(0, color="k", linestyle="--", linewidth=0.5)
    plt.xlabel("Lag")
    plt.ylabel("CMI")
    plt.title("CMI vs Lag")
    plt.show()