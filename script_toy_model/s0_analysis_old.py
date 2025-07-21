# -*- coding: utf-8 -*-
"""
Created on 2025/07/11 13:54:43

@File -> s0_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 分析
"""

from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KernelDensity
from collections import defaultdict
from sklearn.utils import shuffle
from scipy.stats import pearsonr
import numpy as np
import random
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import show_results
from core.cit_entropy import cal_cmi
from core.knn_prob_est import cal_knn_prob_dens, build_tree


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

def cal_tau(x, taus, bt_size: int, bt_rounds: int, thres: float, show: bool = False):
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

    # @staticmethod
    # def __shuffle(x: np.ndarray):
    #     x_srg = shuffle(x)  # 无放回抽样
    #     return x_srg
    
    # def tmp(self, lags2test: np.ndarray, size_bt: int, rounds_bt: int, show: bool = True, **kwargs):
    #     cmi_lag_records = defaultdict(dict)

    #     x_t_lag, y_t, y_t_tau_y = None, None, None
    #     for lag in lags2test:

    #         # 构造对应的时延样本
    #         x_t_lag = np.roll(self.x, lag)  # 对应于 x_{t-lag} 样本
    #         y_t = self.y  # 对应于 y_t 样本
    #         y_t_tau_y = np.roll(self.y, self.tau_y)  # 对应于 y_{t-tau_y} 样本

    #         # 截断，避免引入无关样本对TE构成误差
    #         if lag > 0:
    #             x_t_lag = x_t_lag[lag:]
    #             y_t = y_t[lag:]
    #             y_t_tau_y = y_t_tau_y[lag:]
    #         elif lag < 0:
    #             x_t_lag = x_t_lag[:lag]
    #             y_t = y_t[:lag]
    #             y_t_tau_y = y_t_tau_y[:lag]
    #         else:
    #             pass

    #         break

    #     return x_t_lag, y_t, y_t_tau_y
    
    # @staticmethod
    # def __resample_iid(x_t_lag, y_t, y_t_tau_y):
    #     samples = np.c_[x_t_lag, y_t, y_t_tau_y]
    #     samples_norm = Normalizer().fit_transform(samples)

    #     # 使用KNN估算密度
    #     tree = build_tree(samples_norm, metric="chebyshev")

    #     dens_series = np.zeros(samples_norm.shape[0])
    #     for i in range(samples_norm.shape[0]):
    #         dens = cal_knn_prob_dens(samples_norm[i], tree=tree, k=3, metric="chebyshev")
    #         dens_series[i] = dens

    #     weights = dens_series / np.sum(dens_series)

    #     # 使用np.random.Generator提高采样效率
    #     rng = np.random.default_rng()
    #     idxs_rs = rng.choice(len(samples), size=len(samples), p=weights, replace=True)

    #     x_t_lag = x_t_lag[idxs_rs]
    #     y_t = y_t[idxs_rs]
    #     y_t_tau_y = y_t_tau_y[idxs_rs]
    #     return x_t_lag, y_t, y_t_tau_y

    def compute_cmi(self, x_t_lag, y_t, x_t_lag_tau_x, y_t_tau_y, size_bt):
        idxs = np.arange(len(y_t))
        
        # 无放回抽样
        idxs_bt = np.random.choice(idxs.copy(), size_bt, replace=True)

        x_t_lag_bt = x_t_lag[idxs_bt]
        y_t_bt = y_t[idxs_bt]
        x_t_lag_tau_x_bt = x_t_lag_tau_x[idxs_bt]
        y_t_tau_y_bt = y_t_tau_y[idxs_bt]

        # 计算CMI
        k = 3
        alpha = 0

        z_bt = np.c_[x_t_lag_tau_x_bt, y_t_tau_y_bt]
        cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, z_bt, k=k, alpha=alpha)

        # 计算背景CMI
        idxs = np.arange(len(z_bt))
        idxs_bg = np.random.permutation(idxs.copy())
        x_t_lag_srg = x_t_lag_bt[idxs_bg]
        x_t_lag_tau_x_srg = x_t_lag_tau_x_bt[idxs_bg]
        z_srg = np.c_[x_t_lag_tau_x_srg, y_t_tau_y_bt]
        cmi_bg = cal_cmi(x_t_lag_srg, y_t_bt, z_bt, k=k, alpha=alpha)

        return cmi_bt, cmi_bg
    
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

            # 重采样，使样本满足iid条件
            # <<------------------------------------------------------------------------------------
            # idxs_iid = random.sample(sorted(range(len(x_t_lag))), len(x_t_lag))  # 无放回抽样
            # x_t_lag = x_t_lag[idxs_iid]
            # y_t = y_t[idxs_iid]
            # y_t_tau_y = y_t_tau_y[idxs_iid]
            # <<------------------------------------------------------------------------------------
            # tau_max = max(self.tau_x, self.tau_y)
            # idxs = np.arange(0, len(y_t), tau_max)
            # x_t_lag = x_t_lag[idxs]
            # y_t = y_t[idxs]
            # y_t_tau_y = y_t_tau_y[idxs]
            # <<------------------------------------------------------------------------------------
            # x_t_lag, y_t, y_t_tau_y = self.__resample_iid(x_t_lag, y_t, y_t_tau_y)
            # >>------------------------------------------------------------------------------------

            # 基于Bootstrap的CMI测试
            assert len(y_t) >= size_bt, f"len(y_t) = {len(y_t)} < size_bt = {size_bt}"

            bt_records, bg_records = np.zeros(rounds_bt), np.zeros(rounds_bt)
            for round in range(rounds_bt):
                cmi_bt, cmi_bg = self.compute_cmi(x_t_lag, y_t, x_t_lag_tau_x, y_t_tau_y, size_bt)
                bt_records[round] = cmi_bt
                bg_records[round] = cmi_bg

            bt_records = np.array(bt_records)
            bg_records = np.array(bg_records)

            cmi_lag_records["bt_records"][lag] = bt_records
            cmi_lag_records["bg_records"][lag] = bg_records

        # 扣除背景基线均值，进行校准
        base_value = np.array([p for p in cmi_lag_records["bg_records"].values()]).mean()

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
    c = 10
    tau = 0
    
    assert tau >= 0, "tau必须大于等于0"
    assert tau <= ss_len, "tau不能大于稳态段长度"

    for i in range(ss_len, N):
        X_series[i] = a * X_series[i - 1] + 0.01 * np.random.randn()
        Y_series[i] = b * Y_series[i - 1] + c * X_series[i - tau] + 0.0 * np.random.randn()

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

    X_series, Y_series = gen_samples(N=10000, show=True)

    X_series = X_series[:1000]
    Y_series = Y_series[:1000]

    # ---- 时间常数计算 ------------------------------------------------------------------------------

    recal = False

    if recal:
        taus = np.arange(1, 100, 1)
        tau_x, _ = cal_tau(X_series, taus, bt_size=100, bt_rounds=100, thres=1/np.e, show=True)
        tau_y, _ = cal_tau(Y_series, taus, bt_size=100, bt_rounds=100, thres=0.9, show=True)
    else:
        tau_x = 1
        tau_y = 1

    # ---- 时序分析 ----------------------------------------------------------------------------------

    self = TDTEAnalysis(X_series, Y_series, tau_x, tau_y)

    lags2test = np.arange(-60 * 1, 61 * 1, 1)

    size_bt = 50
    rounds_bt = 100
    show = True

    cmi_lag_records = self.exec_td_te_analysis(lags2test, size_bt, rounds_bt, show=show)

    # ---- 临时 -------------------------------------------------------------------------------------

    # x_t_lag, y_t, y_t_tau_y = self.tmp(lags2test, size_bt, rounds_bt, show=show)

    # # 合并样本并归一化
    # samples = np.c_[x_t_lag, y_t, y_t_tau_y]
    # samples_norm = Normalizer().fit_transform(samples)

    # # 使用KNN估算密度
    # tree = build_tree(samples_norm, metric="chebyshev")

    # dens_lst = []
    # for i in range(samples_norm.shape[0]):
    #     dens = cal_knn_prob_dens(samples_norm[i], tree=tree, k=3, metric="chebyshev")
    #     dens_lst.append(dens)

    # weights = np.array(dens_lst) / np.sum(dens_lst)

    # # 使用np.random.Generator提高采样效率
    # rng = np.random.default_rng()
    # idxs_rs = rng.choice(len(samples), size=len(samples), p=weights, replace=True)

    # x_t_lag = x_t_lag[idxs_rs]
    # y_t = y_t[idxs_rs]
    # y_t_tau_y = y_t_tau_y[idxs_rs]