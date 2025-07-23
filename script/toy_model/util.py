# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 10:39:47

@File -> util.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 工具函数
"""

from scipy.stats import pearsonr
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def cal_tau(x, taus, bt_size: int, bt_rounds: int, thres: float, show: bool = False, fig_savepath: str = None):
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
        plt.plot(list(tau2r.keys()), list(tau2r.values()), "k", linewidth=1.0, label = "r(tau)")
        plt.axhline(y=thres, color="r", linestyle="--", label=f"Threshold = {thres:.2f}")
        plt.xlabel(f"$\\tau$")
        plt.ylabel("$r(\\tau)$")
        plt.title(f"$\\tau$ = {tau_x}")
        plt.legend(loc="upper right")
        plt.tight_layout()

        if fig_savepath is not None:
            plt.savefig(fig_savepath, dpi=450)

        plt.show()

    return tau_x, tau2r


def gen_samples(taus: list, N: int, show: bool = False, fig_savepath: str = None):
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

    a = 0.7
    b = 0.2
    c = 0.5
    d = 0.5

    for tau in taus:
        assert tau >= 0, "tau必须大于等于0"
        assert tau <= ss_len, "tau不能大于稳态段长度"

    for i in range(ss_len, N):
        X_series[i] = a * X_series[i - 1] + 0.01 * np.random.randn()
        Y_series[i] = b * Y_series[i - 1] + c * X_series[i - taus[0]] + d * X_series[i - taus[1]] + 0.01 * np.random.randn()

    X_series = X_series[ss_len:]
    Y_series = Y_series[ss_len:]

    if show:

        plt.figure(figsize=(5, 4))

        plt.subplot(2, 1, 1)
        plt.plot(X_series, "k", linewidth=1.0, label="$X$")
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.plot(Y_series, "k", linewidth=1.0, label="$Y$")
        plt.legend(loc="upper right")
        plt.xlabel("$t$")

        plt.tight_layout()

        if fig_savepath is not None:
            plt.savefig(fig_savepath, dpi=450)

    return X_series, Y_series