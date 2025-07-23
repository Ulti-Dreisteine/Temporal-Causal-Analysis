# -*- coding: utf-8 -*-
"""
Created on 2025/07/10 15:41:00

@File -> s1_ste_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: STE分析
"""

from sklearn.utils import resample, shuffle
from joblib import Parallel, delayed
from collections import defaultdict
from itertools import permutations
import pandas as pd
import numpy as np
import random
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import show_results
from core.cit_entropy import cal_cmi


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
    def _shuffle(x: np.ndarray):
        x_srg = np.random.choice(x, len(x), replace=True)
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

            """
            注意：这里不对数据进行截断，理论上平移超出因果作用范围的样本所得结果为0，不影响最终结果
            """

            # 基于Bootstrap的CMI测试
            assert len(y_t) >= size_bt, f"len(y_t) = {len(y_t)} < size_bt = {size_bt}"

            idxs = np.arange(len(y_t))
            bt_records = np.zeros(rounds_bt)
            bg_records = np.zeros(rounds_bt)

            def compute_cmi(round):
                # idxs_bt = random.choices(idxs, k=size_bt)  # 无放回抽样
                idxs_bt = resample(idxs, n_samples=size_bt, replace=False)
                y_t_bt = y_t[idxs_bt]
                x_t_lag_bt = x_t_lag[idxs_bt]
                y_t_tau_y_bt = y_t_tau_y[idxs_bt]

                # 计算CMI
                cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, y_t_tau_y_bt)

                # 计算背景CMI
                # x_t_lag_srg = self._shuffle(x_t_lag_bt)
                x_t_lag_srg = shuffle(x_t_lag_bt)  # 随机打乱样本
                cmi_bg = cal_cmi(x_t_lag_srg, y_t_bt, y_t_tau_y_bt)

                return cmi_bt, cmi_bg

            results = Parallel(n_jobs=-1)(delayed(compute_cmi)(round) for round in range(rounds_bt))
            bt_records, bg_records = zip(*results)  # type: ignore
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


if __name__ == "__main__":
    df = pd.read_csv("runtime/s0_data.csv")
    df = df.iloc[500:]

    x = df["全球温度异常"].to_numpy()
    y = df["全球降水量"].to_numpy()

    x_sym = symbolize(x, tau=100, m=3, tau_max=200)

    # plt.figure(figsize=(5, 5))
    # plt.subplot(2, 1, 1)
    # plt.plot(x, label="原序列")
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(x_sym, label="符号化序列")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # ---- 变尺度检验 --------------------------------------------------------------------------------

    taus = np.arange(1, 21)
    for i, tau in enumerate(taus):
        x_sym = symbolize(x, tau=tau, m=3, tau_max=100)
        y_sym = symbolize(y, tau=tau, m=3, tau_max=100)

        # 执行时延分析
        tdte_analysis = TDTEAnalysis(x_sym, y_sym, tau_x=tau, tau_y=tau)
        lags2test = np.arange(-10 * tau, (10 + 1) * tau, tau)
        
        size_bt = 50
        rounds_bt = 10
        show = True

        cmi_lag_records = tdte_analysis.exec_td_te_analysis(lags2test, size_bt, rounds_bt, show)

