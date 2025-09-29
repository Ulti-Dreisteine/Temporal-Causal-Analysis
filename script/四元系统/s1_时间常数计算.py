# -*- coding: utf-8 -*-
"""
Created on 2025/09/29 15:08:48

@File -> s1_时间常数计算.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 时间常数计算
"""

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.cit_entropy import SU


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
            r_bt = SU(x1[idxs_bt], x2[idxs_bt])

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
        plt.figure(figsize = (5, 4))
        plt.plot(list(tau2r.keys()), list(tau2r.values()), label = "r(tau)")
        plt.xlabel("tau")
        plt.ylabel("r(tau)")
        plt.title(f"r(tau) = {tau_x}")
        plt.legend()
        plt.show()

    return tau_x, tau2r


if __name__ == "__main__":
    samples_df = pd.read_csv("runtime/s0_样本数据.csv")

    taus = list(range(1, 101))
    bt_size = 100
    bt_rounds = 100
    thres = 1 / np.e
    tau_xs = []

    for i in range(4):
        x = samples_df[f"x{i}"].values
        tau_x, tau2r = cal_tau_x(x, taus, bt_size, bt_rounds, thres, show=True)
        tau_xs.append(tau_x)
        print(f"X{i}: tau_x = {tau_x}")