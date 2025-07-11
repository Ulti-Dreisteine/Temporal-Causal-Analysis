# -*- coding: utf-8 -*-
"""
Created on 2025/07/11 09:45:17

@File -> s0_gen_samples.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据生成
"""

from sklearn.utils import resample, shuffle
from joblib import Parallel, delayed
from collections import defaultdict
from itertools import permutations
from scipy.stats import pearsonr
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import show_results
from core.cit_entropy import cal_cmi


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

def _gen_embed_series(x, idxs, m, tau):
    X_embed = x.copy()[idxs]
    for i in range(1, m):
        X_embed = np.c_[x[idxs - i * tau], X_embed]
    return X_embed


def symbolize(x: np.ndarray, tau, m, tau_max):
    """符号化"""
    # 确定所有模式集合
    patterns = list(permutations(np.arange(m) + 1))
    dict_pattern_index = {patterns[i]: i for i in range(len(patterns))}
    
    idxs = np.arange((m - 1) * tau_max, len(x))  # 连续索引
    X_embed = _gen_embed_series(x, idxs, m, tau)
    
    X = np.argsort(X_embed) + 1  # NOTE: 滚动形成m维时延嵌入样本  一个时刻成为一个标签
    X = np.array([dict_pattern_index[tuple(p)] for p in X])  # 对应映射到符号上
    return X


def cal_tau_x(x, taus, bt_size: int, bt_rounds: int, thres: float, show: bool = False):
    """
    计算时延关联系数

    Params:
    -------
    x: 输入序列，shape = (n_samples,)
    taus: 时延序列
    bt_size: Bootstrap采样大小
    bt_rounds: Bootstrap重复次数
    thres: 阈值
    show: 是否显示结果
    """
    
    # 计算时延关联系数
    tau2r = {}
    for tau in taus:
        x1 = np.roll(x, tau)
        x2 = x

        idxs = np.arange(len(x1))
        idxs = idxs[::10]

        r = []
        for i in range(bt_rounds):
            idxs_bt = np.random.choice(idxs, bt_size, replace=True)

            # 计算关联系数
            r_bt = pearsonr(x1[idxs_bt], x2[idxs_bt])[0]

            r.append(r_bt)
        
        tau2r[tau] = np.abs(np.mean(r))

    # 以tau2r第一次低于阈值的tau作为时间常数
    taus = list(tau2r.keys())
    rs = list(tau2r.values())

    try:
        tau_x = taus[np.where(np.array(rs) < thres)[0][0]]
    except:
        print("Warning: no tau_x found, use the last tau instead.")
        tau_x = taus[-1]

    if show:
        plt.figure(figsize = (4, 3))
        plt.plot(list(tau2r.keys()), list(tau2r.values()), label = "r(tau)")
        plt.xlabel("tau")
        plt.ylabel("r(tau)")
        plt.title(f"r(tau) = {tau_x}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return tau_x, tau2r


class TDTEAnalysis(object):
    """基于TE识别变量之间的时延"""

    def __init__(self, x: np.ndarray, y: np.ndarray, tau_x: int, tau_y: int) -> None:
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()
        self.tau_x = int(tau_x)
        self.tau_y = int(tau_y)

        assert len(self.x) == len(self.y), f"len(x) = {len(self.x)} != len(y) = {len(self.y)}"
        self.N = len(self.x)

    @staticmethod
    def __shuffle(x: np.ndarray):
        x_srg = shuffle(x)  # 无放回抽样
        return x_srg

    def exec_td_te_analysis(self, lags2test: np.ndarray, size_bt: int, rounds_bt: int, show: bool = True, **kwargs):
        """
        执行时延分析，挖掘x对y的作用时延

        Params:
        -------
        lags2test: 待测试的时延范围
        size_bt: Bootstrap样本大小
        rounds_bt: Bootstrap重复次数
        cmi_kwargs: CMI计算参数
        show: 是否显示图像

        Kwargs:
        -------
        x_col: x变量名称
        y_col: y变量名称
        """

        cmi_lag_records = defaultdict(dict)

        for lag in lags2test:

            # 构造对应的时延样本
            x_t_lag = np.roll(self.x, lag)  # 对应于 x_{t-lag} 样本
            y_t = self.y  # 对应于 y_t 样本
            y_t_tau_y = np.roll(self.y, self.tau_y)  # 对应于 y_{t-tau_y} 样本

            # 截断，避免引入无关样本对TE构成误差
            if lag > 0:
                x_t_lag = x_t_lag[lag:]
                y_t = y_t[lag:]
                y_t_tau_y = y_t_tau_y[lag:]
            elif lag < 0:
                x_t_lag = x_t_lag[:lag]
                y_t = y_t[:lag]
                y_t_tau_y = y_t_tau_y[:lag]

            # 间隔采样
            tau_max = max(self.tau_x, self.tau_y)
            idxs = np.arange(0, len(y_t), tau_max)
            x_t_lag = x_t_lag[idxs]
            y_t = y_t[idxs]
            y_t_tau_y = y_t_tau_y[idxs]

            # 基于Bootstrap的CMI测试
            assert len(y_t) >= size_bt, f"len(y_t) = {len(y_t)} < size_bt = {size_bt}"

            idxs = np.arange(len(y_t))
            bt_records = np.zeros(rounds_bt)
            bg_records = np.zeros(rounds_bt)

            def compute_cmi():
                idxs_bt = resample(idxs, n_samples=size_bt, replace=False)  # 无放回抽样
                y_t_bt = y_t[idxs_bt]
                x_t_lag_bt = x_t_lag[idxs_bt]
                y_t_tau_y_bt = y_t_tau_y[idxs_bt]

                # 计算CMI
                k = 3
                alpha = 0.1
                cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, y_t_tau_y_bt, k=k, alpha=alpha)

                # 计算背景CMI
                x_t_lag_srg = self.__shuffle(x_t_lag_bt)  # 随机打乱样本
                cmi_bg = cal_cmi(x_t_lag_srg, y_t_bt, y_t_tau_y_bt, k=k, alpha=alpha)

                return cmi_bt, cmi_bg

            results = Parallel(n_jobs=-1)(delayed(compute_cmi)() for round in range(rounds_bt))
            bt_records, bg_records = zip(*results)
            bt_records = np.array(bt_records)
            bg_records = np.array(bg_records)

            cmi_lag_records["bt_records"][lag] = bt_records
            cmi_lag_records["bg_records"][lag] = bg_records

        # 扣除背景基线均值，进行校准
        base_value = np.array([p for p in cmi_lag_records["bg_records"].values()]).mean()
        # base_value = 0.0

        for lag in cmi_lag_records["bt_records"].keys():
            bt_records = cmi_lag_records["bt_records"][lag]
            bg_records = cmi_lag_records["bg_records"][lag]

            bt_records = bt_records - base_value
            bg_records = bg_records - base_value

            cmi_lag_records["bt_records"][lag] = bt_records
            cmi_lag_records["bg_records"][lag] = bg_records

        # 画图
        if show:
            show_results(cmi_lag_records, **kwargs)

        return cmi_lag_records
    

# **************************************************************************************************
# 项目工具
# **************************************************************************************************

def gen_samples(N: int, show: bool):
    # 初始化数组
    X_series = np.ones(N) * np.nan
    Y_series = np.ones(N) * np.nan

    # 稳态段初始
    ss_len = 10
    X_series[:ss_len] = 0
    Y_series[:ss_len] = 0

    a = 0.95
    b = 0.9
    c = 0.1
    tau = 0

    assert tau >= 0, "tau必须大于等于0"
    assert tau <= ss_len, "tau不能大于稳态段长度"

    for i in range(ss_len, N):
        X_series[i] = a * X_series[i - 1] + 0.1 * np.random.randn()
        Y_series[i] = b * Y_series[i - 1] + c * X_series[i - tau] + 0.1 * np.random.randn()

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
    
    return X_series[ss_len:], Y_series[ss_len:]


if __name__ == "__main__":
    # 生成样本
    X_series, Y_series = gen_samples(N=100000, show=True)

    # 时间常数计算
    taus = np.arange(1, 100, 1)
    tau_x, _ = cal_tau_x(X_series, taus, bt_size=100, bt_rounds=100, thres=0.3, show=True)
    tau_y, _ = cal_tau_x(Y_series, taus, bt_size=100, bt_rounds=100, thres=0.3, show=True)

    # 时延因果分析
    taus = np.arange(10, 11)
    for i, tau in enumerate(taus):
        # 执行时延分析
        # X_sym = symbolize(X_series, tau=tau_x, m=3, tau_max=100)
        # Y_sym = symbolize(Y_series, tau=tau_y, m=3, tau_max=100)
        # tdte_analysis = TDTEAnalysis(X_sym, Y_sym, tau_x=tau_x, tau_y=tau_y)

        tau_x = tau_x
        tau_y = tau_y
        tdte_analysis = TDTEAnalysis(X_series, Y_series, tau_x=tau_x, tau_y=tau_y)

        lags2test = np.arange(-30 * 10, 31 * 10, 10)

        size_bt = 500
        rounds_bt = 100
        show = True

        cmi_lag_records = tdte_analysis.exec_td_te_analysis(lags2test, size_bt, rounds_bt, show=show)

