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